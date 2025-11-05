import torch
import numpy as np
import pandas as pd
from spectrum.loss import spectrum_gmm, spectrum_fc

# SPDX-License-Identifier: LicenseRef-Proprietary
# Copyright (c) 2025 Hasup Lee. All rights reserved.

def save_spectrum(preds, ids, wrt_file, spectrum_type='FC', kernel_kind='gaussian', beta=2.0):
    nm2ev = 1240.0
    spec_x = torch.arange(400.0, 800.0, 0.5)
    spec_x = nm2ev / spec_x
    spec_x = spec_x.unsqueeze(0).expand(len(ids), -1)
    if spectrum_type == 'Naive':
        preds = preds.numpy()
        mins = preds.min(axis=1, keepdims=True)
        maxs = preds.max(axis=1, keepdims=True)
        spec_y = (preds-mins) / (maxs-mins)
    elif spectrum_type == 'GMM':
        spec_y = spectrum_gmm(spec_x, preds[:,0:1], preds[:,1:2], 
                preds[:,2:3], preds[:,3:4], preds[:,4:5], 
                preds[:,5:6], preds[:,6:7], preds[:,7:8])
    elif spectrum_type == 'FC':
        n_S = (preds.shape[1]-2)//2
        spec_y = spectrum_fc(spec_x, preds[:,0:n_S], preds[:,n_S+2:2*n_S+2], 
                preds[:,n_S:n_S+1], preds[:,n_S+1:n_S+2],
                kernel_kind=kernel_kind, beta=beta)
    else:
        raise Exception("Undefined spectrum type")

    x = np.arange(400, 800, 0.5)
    strings = [f"{v:.1f}".rstrip("0").rstrip(".") for v in x]
    line_x = " ".join(strings)

    data = []
    for name, spec in zip(ids, spec_y):
        line_y = " ".join(f"{val:.6f}" for val in spec.tolist())
        data.append((name, line_x, line_y))

    df = pd.DataFrame(data, columns=["molecule_id", "Wavelength(nm)", "Intensity"])
    df.to_csv(wrt_file, index=False)

