from typing import Optional

import torch
from pytorch_lightning import LightningModule
from torch.nn.functional import l1_loss, mse_loss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, LambdaLR

from geoformer.model.modeling_geoformer import create_model
import os
import numpy as np
import pandas as pd
import itertools
from spectrum.loss import gmm_loss, fc_loss
from spectrum.write import save_spectrum

def warmup_exponential_decay(step: int, hparams):
    alpha = min( 1.0, float(step)/float(hparams.lr_warmup_steps) )
    lr_scale = hparams.lr_warmup_factor * (1.0-alpha) + alpha
    lr_exp = hparams.decay_rate**(step/hparams.decay_step)
    return lr_scale * lr_exp

class LNNP(LightningModule):
    def __init__(self, config) -> None:
        super(LNNP, self).__init__()

        self.save_hyperparameters(config)
        self.model = create_model(self.hparams)
        self._reset_losses_dict()

        self._id_buffer = []
        self._pred_buffer: list[torch.Tensor] = []
        self._label_buffer: list[torch.Tensor] = []

    def configure_optimizers(self) -> Optional[AdamW]:
        optimizer = AdamW(
            params=self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        if self.hparams.lr_scheduler == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.lr_cosine_length,
                eta_min=self.hparams.lr_min,
            )
            lr_scheduler = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        elif self.hparams.lr_scheduler == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                "min",
                factor=self.hparams.lr_factor,
                patience=self.hparams.lr_patience,
                min_lr=self.hparams.lr_min,
            )
            lr_scheduler = {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            }

        elif self.hparams.lr_scheduler == "exponential":
            scheduler = LambdaLR(
                optimizer,
                lr_lambda=lambda step: warmup_exponential_decay(step, self.hparams),
            )
            
            lr_scheduler = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }

        else:
            raise NotImplementedError(
                f"Unknown lr_schedule: {self.hparams.lr_scheduler}"
            )

        return [optimizer], [lr_scheduler]

    def forward(self, batch):
        return self.model(z=batch["z"], pos=batch["pos"])

    def training_step(self, batch, batch_idx):
        if self.hparams.spec_loss_type == 'GMM':
            return self.spectrum_step(batch, gmm_loss, "train")
        elif self.hparams.spec_loss_type == 'FC':
            return self.spectrum_step(batch, fc_loss, "train")

        if self.hparams.loss_type == "MSE":
            return self.step(batch, mse_loss, "train")
        elif self.hparams.loss_type == "MAE":
            return self.step(batch, l1_loss, "train")
        else:
            NotImplementedError(f"Unknown loss type: {self.hparams.loss_type}")

    def validation_step(self, batch, batch_idx):
        if self.hparams.spec_loss_type == 'GMM':
            return self.spectrum_step(batch, gmm_loss, "val")
        elif self.hparams.spec_loss_type == 'FC':
            return self.spectrum_step(batch, fc_loss, "val")
        else:
            # return self.step(batch, l1_loss, "val")
            if self.hparams.loss_type == "MSE":
                return self.step(batch, mse_loss, "val")
            elif self.hparams.loss_type == "MAE":
                return self.step(batch, l1_loss, "val")

    def test_step(self, batch, batch_idx):
        if self.hparams.spec_loss_type == 'GMM':
            loss = self.spectrum_step(batch, gmm_loss, "test")
        elif self.hparams.spec_loss_type == 'FC':
            loss = self.spectrum_step(batch, fc_loss, "test")
        else:
            # loss = self.step(batch, l1_loss, "test")
            if self.hparams.loss_type == "MSE":
                loss = self.step(batch, mse_loss, "test")
            elif self.hparams.loss_type == "MAE":
                loss = self.step(batch, l1_loss, "test")

        with torch.no_grad():
            pred = self(batch)
            tgt = batch["labels"]
            if tgt.ndim == 1:
                tgt = tgt.unsqueeze(1)
            molid = batch['name']
            self._pred_buffer.append(pred.cpu())
            self._label_buffer.append(tgt.cpu())
            self._id_buffer.append(molid)
        return loss

    def step(self, batch, loss_fn, stage):
        with torch.set_grad_enabled(stage == "train"):
            pred = self(batch)

        loss = 0

        if "labels" in batch:
            if batch["labels"].ndim == 1:
                batch["labels"] = batch["labels"].unsqueeze(1)

            loss = loss_fn(pred, batch["labels"])
            self.losses[stage].append(loss.detach())

        self.losses[stage].append(loss.detach())

        return loss

    def spectrum_step(self, batch, loss_fn, stage):
        with torch.set_grad_enabled(stage == "train"):
            pred = self(batch)

        loss = loss_fn(batch["spec_x"], batch["spec_y"], pred, 
                loss_type=self.hparams.loss_type, line_shape=self.hparams.lineshape,
                beta=self.hparams.beta)
        self.losses[stage].append(loss.detach())

        return loss

    def optimizer_step(self, *args, **kwargs):
        optimizer = kwargs["optimizer"] if "optimizer" in kwargs else args[2]
        super().optimizer_step(*args, **kwargs)
        optimizer.zero_grad()

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            result_dict = {
                "epoch": float(self.current_epoch),
                "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
                "train_loss": torch.stack(self.losses["train"]).mean(),
                "val_loss": torch.stack(self.losses["val"]).mean(),
            }

            # add test loss if available
            if len(self.losses["test"]) > 0:
                result_dict["test_loss"] = torch.stack(
                    self.losses["test"]
                ).mean()

            self.log_dict(result_dict, prog_bar=True, sync_dist=True)

        self._reset_losses_dict()

    def on_test_epoch_end(self):
        result_dict = {}
        if len(self.losses["test"]) > 0:
            result_dict["test_loss"] = torch.stack(self.losses["test"]).mean()

        if self._pred_buffer:
            preds = torch.cat(self._pred_buffer, 0)
            ids = list(itertools.chain.from_iterable(self._id_buffer))
            save_spectrum(preds, ids, os.path.join(self.hparams.log_dir, "p_spec.csv"), 
                    spectrum_type=self.hparams.spec_loss_type,
                    kernel_kind=self.hparams.lineshape,
                    beta=self.hparams.beta)
            preds = preds.numpy()
            if isinstance(self.hparams.dataset_arg, (list, tuple)):
                col_names = list(self.hparams.dataset_arg)
            else:
                col_names = [self.hparams.dataset_arg]
            
            preds_df = pd.DataFrame(preds, columns=col_names)
            preds_df.insert(0, "molecule_id", ids)
            preds_df.to_csv(os.path.join(self.hparams.log_dir, "p.csv"), index=False)

        self.log_dict(result_dict, sync_dist=True)
        self._reset_losses_dict()
        
        self._pred_buffer.clear()
        self._label_buffer.clear()
        self._id_buffer.clear()

    def _reset_losses_dict(self):
        self.losses = {
            "train": [],
            "val": [],
            "test": [],
        }
