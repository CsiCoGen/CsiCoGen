"""Evaluate CSI reconstruction quality from saved reconstructions."""
from pathlib import Path
import numpy as np
import torch
import scipy.io as sio
from modules.diffusion import load_ht_mat, load_raw_hf_all
from utils.metrics import evaluate_crnet_metrics_np


def evaluate_files(reconstruction, sparse_file, raw_file='', device='cpu'):
    archive = sio.loadmat(reconstruction)
    key = 'ldm_ddpm_rec'
    rec = archive[key].transpose(0, 3, 1, 2)
    n = len(rec)
    sparse = load_ht_mat(sparse_file, 32, 2)[:n]
    if len(sparse) != n:
        raise ValueError('Reference CSI must cover every reconstructed sample.')
    raw = load_raw_hf_all(raw_file)[:n] if raw_file and Path(raw_file).is_file() else None
    if raw is not None and len(raw) != n:
        raise ValueError('Frequency references must cover every reconstructed sample.')
    nmse, rho = evaluate_crnet_metrics_np(rec, sparse, raw, torch.device(device), batch_size=500)
    pred = rec.astype(np.float64) - 0.5
    target = sparse.astype(np.float64) - 0.5
    ratios = ((pred-target)**2).sum(axis=(1,2,3)) / (target**2).sum(axis=(1,2,3))
    return {'num_samples': n, 'nmse': nmse, 'rho': rho,
            'nmse_global_db': float(10*np.log10(ratios.mean())),
            'nmse_definition': 'sample-weighted average of 500-sample batch NMSE values in dB'}
