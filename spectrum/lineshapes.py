# SPDX-License-Identifier: LicenseRef-Proprietary
# Copyright (c) 2025 Hasup Lee. All rights reserved.

import torch
from math import sqrt, log, pi

_TWO_SQRT_2LN2 = 2.0 * sqrt(2.0 * log(2.0))  # FWHM = _TWO_SQRT_2LN2 * sigma

def _as_t(x, like):
    return torch.as_tensor(x, dtype=like.dtype, device=like.device)

def gaussian(delta, fwhm):
    sigma = fwhm / _TWO_SQRT_2LN2
    return (1.0/(sigma*sqrt(2.0*pi))) * torch.exp(-0.5*(delta/sigma)**2)

def lorentzian(delta, fwhm):
    gamma = fwhm / 2.0
    return (1.0/pi) * (gamma / (delta**2 + gamma**2))

def generalized_gaussian(delta, fwhm, beta=2.0):
    D = delta
    F = _as_t(fwhm, D)
    b = _as_t(beta, D)

    ln2 = _as_t(log(2.0), D)
    alpha = F / (2.0 * torch.exp(torch.log(ln2) / b))

    log_coeff = torch.log(b) - torch.log(2.0*alpha) - torch.lgamma(1.0/b)
    coeff = torch.exp(log_coeff)

    z = torch.abs(D) / alpha
    return coeff * torch.exp(- torch.pow(z, b))

def pearson_vii(delta, fwhm, m=2.0):
    m_t = _as_t(m, delta)
    a = fwhm / (2.0 * torch.sqrt(_as_t(2.0, delta)**(1.0/m) - 1.0))
    log_coeff = torch.lgamma(m_t) - torch.lgamma(m_t - 0.5) - _as_t(pi, delta).sqrt().log() - torch.log(a)
    coeff = torch.exp(log_coeff)
    return coeff * (1.0 + (delta/a)**2)**(-m_t)

def make_kernel(kind: str, w_voigt=0.5, beta=2.0):
    kind = kind.lower()
    def k(delta, fwhm):
        if kind == "gaussian":
            return gaussian(delta, fwhm)
        if kind == "lorentzian":
            return lorentzian(delta, fwhm)
        if kind == "pearson" or kind == "pearson_vii":
            return pearson_vii(delta, fwhm, m=2.0)
        if kind == 'generalized_gaussian':
            return generalized_gaussian(delta, fwhm, beta=beta)
        if kind == "voigt" or kind == "pseudo_voigt":
            return w_voigt*gaussian(delta, fwhm) + (1-w_voigt)*lorentzian(delta, fwhm)
        raise ValueError(f"Unknown lineshape kind: {kind}")
    return k

def sigma_to_fwhm(cc):
    return cc * _TWO_SQRT_2LN2

