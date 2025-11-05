from types import SimpleNamespace
import argparse
import datetime
import itertools
import pickle
import subprocess
import time
import torch
from torch import nn
import numpy as np
from torch_geometric.loader import DataLoader
from torch_cluster import radius_graph
from torch_scatter import scatter_mean

import os
from pathlib import Path

from contextlib import suppress
from timm.utils import NativeScaler

from dataset.IrDB import IrDB
from optim_factory import create_optimizer
from logger import FileLogger

from engine import train_one_step, evaluate, compute_stats
from torch.optim.lr_scheduler import LambdaLR
from fairchem.core.models.painn.painn import PaiNN as _PaiNN

# distributed training
import utils as utils
import pandas as pd
from spectrum.write import save_spectrum

class OneBatchLoader:
    def __init__(self, batch): self.batch = batch
    def __iter__(self):
        yield self.batch
    def __len__(self):
        return 1

def _rbf(dist, K, cutoff):
    centers = torch.linspace(0, cutoff, K, device=dist.device)
    gamma = 1.0 / ((cutoff / max(K,1))**2 + 1e-9)
    return torch.exp(-gamma * (dist.unsqueeze(-1) - centers)**2)

class PaiNN(nn.Module):
    def __init__(
        self,
        out_channels=1,          
        cutoff=5.0,              
        hidden_channels=128,     
        num_layers=6,      
        num_rbf=64,              
        **kwargs,                
    ):
        super().__init__()
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        self.backbone = _PaiNN(
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_rbf=num_rbf,
            cutoff=cutoff,
            out_channels = out_channels,
        )

    @torch.no_grad()
    def _edge_index(self, pos, batch):
        return radius_graph(pos, r=self.cutoff, batch=batch, loop=False, max_num_neighbors=512)

    def forward(
        self,
        f_in,              
        pos,               
        batch,             
        node_atom,         
        **kwargs,
    ):
        edge_index = self._edge_index(pos, batch)
        rij = pos[edge_index[0]] - pos[edge_index[1]]
        dist = rij.norm(dim=-1)
        edge_attr = _rbf(dist, self.num_rbf, self.cutoff)
        data = SimpleNamespace(
                pos=pos,
                z=node_atom.long(),
                atomic_numbers=node_atom.long(),
                edge_index=edge_index,
                edge_attr=edge_attr,
                batch=batch,
                natoms=torch.bincount(batch),
                )
        B = int(batch.max())+1
        data.pbc = torch.zeros(B, 3, dtype=torch.bool, device=pos.device)
        data.cell = torch.zeros(B, 3, 3, dtype=pos.dtype, device=pos.device)
        pred = self.backbone(data)

        return pred

def build_painn(args, out_channels=1):
    return PaiNN(
        out_channels=out_channels,
        cutoff=args.radius,
        num_rbf=args.num_basis,
        hidden_channels=args.embed_dim,
        num_layers=args.num_layers,
    )

def get_args_parser():
    parser = argparse.ArgumentParser('Training equivariant networks', add_help=False)
    parser.add_argument('--output-dir', type=str, default=None)
    # network architecture
    parser.add_argument('--radius', type=float, default=5.0)
    parser.add_argument('--num-basis', type=int, default=128)
    parser.add_argument('--embed-dim', type=int, default=512)
    parser.add_argument('--num-layers', type=int, default=6)
    # training hyper-parameters
    parser.add_argument("--batch-size", type=int, default=64)
    # optimizer (timm)
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=1.0, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='weight decay (default: 0.01)')
    # learning rate schedule parameters (timm)
    parser.add_argument('--lr', type=float, default=1.0e-3, metavar='LR',
                        help='learning rate (default: 1.0e-3)')
    parser.add_argument('--min-lr', type=float, default=1.0e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1.0e-6)')

    parser.add_argument('--decay-step', type=int, default=10000, metavar='N',
                        help='step interval to decay LR')
    parser.add_argument('--lr-warmup-steps', type=int, default=1000, metavar='N',
                        help='steps to warmup LR, if scheduler supports')
    parser.add_argument('--lr-warmup-factor', type=float, default=0.1)
    parser.add_argument('--decay-rate', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    # logging
    parser.add_argument("--print-freq", type=int, default=100)
    # task
    parser.add_argument("--data-path", type=str, default='IrDB')
    parser.add_argument('--compute-stats', action='store_true', dest='compute_stats')
    parser.set_defaults(compute_stats=False)
    parser.add_argument('--no-standardize', action='store_false', dest='standardize')
    parser.set_defaults(standardize=True)
    parser.add_argument('--loss', type=str, default='MAE')
    # random
    parser.add_argument("--seed", type=int, default=0)
    # data loader config
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    parser.add_argument('--split-index-npz', type=str, default='IrDB/raw/splits.npz')

    parser.add_argument('--train-steps', type=int, default=10000)
    parser.add_argument('--eval-steps', type=int, default=100)
    
    #spectrum_calculation
    parser.add_argument('--spectrum-type', type=str, default='FC')
    parser.add_argument('--n-mode', type=int, choices=[2,3,4,5,6], default=3, help='number of vibronic modes for FC progression (allowed: 2,3,4,5,6)')
    parser.add_argument('--lineshape', type=str, default='gaussian')
    parser.add_argument('--beta', type=float, default=2.0)

    return parser

def load_split_from_npz(path):
    if not os.path.exists(path):
        raise Exception(f"npz file {path} is not exist")
    data = np.load(path)
    idx_train = data['idx_train']
    idx_val = data['idx_val']
    idx_test = data['idx_test']
    return idx_train.tolist(), idx_val.tolist(), idx_test.tolist()

def warmup_exponential_decay(step: int, hparams):
    alpha = min( 1.0, float(step)/float(hparams.lr_warmup_steps) )
    lr_scale = hparams.lr_warmup_factor * (1.0-alpha) + alpha
    lr_exp = hparams.decay_rate**(step/hparams.decay_step)
    return lr_scale * lr_exp

def save_pred(preds, ids, wrt_file, col_names):
    preds_df = pd.DataFrame(preds, columns=col_names)
    preds_df.insert(0, "molecule_id", ids)
    preds_df.to_csv(wrt_file, index=False)

def main(args):

    utils.init_distributed_mode(args)
    is_main_process = (args.rank == 0)

    _log = FileLogger(is_master=is_main_process, is_rank0=is_main_process, output_dir=args.output_dir)
    _log.info(args)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    ''' Dataset '''
    dataset = IrDB(root=args.data_path, dataset_arg=args.targets)
    idx_train, idx_val, idx_test = load_split_from_npz(args.split_index_npz)
    train_dataset = dataset[idx_train]
    val_dataset = dataset[idx_val]
    test_dataset = dataset[idx_test]

    # calculate dataset stats
    task_mean = [0.0 for _ in args.targets]
    task_std = [1.0 for _ in args.targets]
    if args.standardize:
        task_mean = [train_dataset.mean(i) for i in train_dataset.label_idx]
        task_std = [train_dataset.std(i) for i in train_dataset.label_idx]
    _log.info('Training set mean: {}, std:{}'.format(
        ' '.join(map(str, task_mean)), ' '.join(map(str, task_std))))
    
    # since dataset needs random 
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    task_mean = torch.tensor(task_mean).to(device)
    task_std = torch.tensor(task_std).to(device)
    norm_factor = [task_mean, task_std]
    
    ''' Network '''
    model = build_painn(args, out_channels=len(args.targets))
    _log.info(model)
    model = model.to(device)
    
    # distributed training
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _log.info('Number of params: {}'.format(n_parameters))
    
    ''' Optimizer and LR Scheduler '''
    optimizer = create_optimizer(args, model)
    lr_scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: warmup_exponential_decay(step, args),
    )
 
    criterion = None #torch.nn.L1Loss() # torch.nn.MSELoss() 
    if args.loss == 'MAE':
        criterion = torch.nn.L1Loss()
    elif args.loss == 'MSE':
        criterion = torch.nn.MSELoss()
    else:
        raise ValueError

    ''' Data Loader '''
    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(
                train_dataset, num_replicas=utils.get_world_size(), rank=utils.get_rank(), shuffle=True
            )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
            sampler=sampler_train, num_workers=args.workers, pin_memory=args.pin_mem, 
            drop_last=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
            shuffle=True, num_workers=args.workers, pin_memory=args.pin_mem, 
            drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    ''' Compute stats '''
    if args.compute_stats:
        compute_stats(train_loader, max_radius=args.radius, logger=_log, print_freq=args.print_freq)
        return
    
    train_iter = iter(train_loader)
    best_val_loss = float('inf')

    for step in range(1, args.train_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        train_err = train_one_step(
            model=model, criterion=criterion, norm_factor=norm_factor,
            data_loader=OneBatchLoader(batch),
            optimizer=optimizer, device=device,
            loss_scaler=(loss_scaler if 'loss_scaler' in locals() else None),
            clip_grad=getattr(args, 'clip_grad', None),
            input_step=step,
            print_freq=args.print_freq,
            loss_type=args.loss,
            spec_type=args.spectrum_type,
            line_shape=args.lineshape,
            beta=args.beta,
            logger=_log
        )
        lr_scheduler.step()

        if (step % args.eval_steps == 0) or (step == args.train_steps):
            val_mae, val_loss, _, _ = evaluate(
                model, norm_factor,
                val_loader, device,
                print_freq=args.print_freq, 
                loss_type=args.loss, 
                spec_type=args.spectrum_type,
                line_shape=args.lineshape,
                beta=args.beta,
                logger=_log
            )
            _log.info(f'[step {step}] val MAE={val_mae:.6f}, val loss={val_loss:.6f}, best val={best_val_loss:.6f}')

            test_mae, test_loss, preds, ids = evaluate(
                model, norm_factor,
                test_loader, device,
                print_freq=args.print_freq, 
                loss_type=args.loss, 
                spec_type=args.spectrum_type,
                line_shape=args.lineshape,
                beta=args.beta,
                logger=_log
            )
            _log.info(f'[step {step}] test MAE={test_mae:.6f}, test loss={test_loss:.6f}')

            if val_loss < best_val_loss and getattr(args, 'output_dir', None):
                best_val_loss = val_loss
                os.makedirs(args.output_dir, exist_ok=True)
                save_pred(preds.numpy(), ids, os.path.join(args.output_dir, 'pred.csv'), args.targets)
                save_spectrum(preds, ids, os.path.join(args.output_dir, 'p_spec.csv'), args.spectrum_type,
                        kernel_kind=args.lineshape, beta=args.beta)
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'args': args
                }, os.path.join(args.output_dir, 'checkpoint_best.ckpt'))

    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser('Training PaiNN', parents=[get_args_parser()])
    args = parser.parse_args()  
    if args.spectrum_type == 'Naive':
        args.targets = [f'y{i}' for i in range(800)]
        args.standardize = False
    elif args.spectrum_type == 'GMM':
        args.targets = ['A2','A3','B1','B2','B3','C1','C2','C3']
        args.standardize = True
    elif args.spectrum_type == 'FC':
        args.targets = ['S1','S2','S3','C','E0','h1','h2','h3']
        if args.n_mode != 3:
            args.targets = []
            for i in range(1, args.n_mode+1):
                args.targets.append(f'S{i}({args.n_mode})')
            args.targets += [f'C({args.n_mode})',f'E0({args.n_mode})']
            for i in range(1, args.n_mode+1):
                args.targets.append(f'h{i}({args.n_mode})')
        args.standardize = True
    else:
        raise Exception("Spectrum type Error")
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print (args.targets)
    main(args)
    
