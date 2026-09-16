"""Finite-rate codebook contracts and Gaussian codebook diffusion."""

import hashlib
import os

import numpy as np
import torch

from utils import logger
from modules.diffusion import (
    build_beta_schedule,
    ensure_dir,
    load_torch,
    normalize_gen_mode,
    normalize_pred_mode,
)


def codec_contract_for_cfg(cfg):
    normalize_gen_mode(getattr(cfg, 'GEN_MODE', 'ddpm'))
    return {
        'gen_mode': 'ddpm',
        'codec': 'legacy_ddpm_codebook',
        'codec_update': 'ddpm',
        'encode_decode_mode': 'ddpm',
    }

def codebook_vector_norm_stats(codebook_cpu):
    flat = codebook_cpu.detach().reshape(codebook_cpu.shape[0], codebook_cpu.shape[1], -1).float()
    norms = flat.norm(dim=2)
    return {
        'mean': float(norms.mean().item()),
        'std': float(norms.std(unbiased=False).item()),
        'min': float(norms.min().item()),
        'max': float(norms.max().item()),
    }

def default_shared_codebook_path(cfg, codebook_size=None, codebook_seed=None):
    shared_root = str(getattr(cfg, 'CODEBOOK_SHARED_ROOT', '')).strip()
    if not shared_root:
        return ''
    size = int(cfg.CODEBOOK_SIZE if codebook_size is None else codebook_size)
    seed = int(getattr(cfg, 'CODEBOOK_SEED', -1) if codebook_seed is None else codebook_seed)
    if seed < 0:
        seed = int(cfg.SEED)
    filename = (
        f'codebook_T{int(cfg.DIFF_TIMESTEPS)}_K{size}_'
        f'C{int(cfg.IN_CHANNELS)}_M{int(cfg.MAT_SIZE)}_seed{seed}.pt'
    )
    return os.path.join(shared_root, filename)

def resolve_codebook_path(cfg):
    codebook_path_cfg = str(getattr(cfg, 'CODEBOOK_PATH', '')).strip()
    if codebook_path_cfg:
        return os.path.abspath(codebook_path_cfg)
    shared_path = default_shared_codebook_path(cfg)
    if shared_path:
        return os.path.abspath(shared_path)
    return os.path.abspath(os.path.join(cfg.DIFF_RESULTS_DIR, 'codebook.pt'))

def fingerprint_codebook_tensors(codebook_cpu, init_noise_cpu):
    h = hashlib.sha1()
    cb_flat = codebook_cpu.detach().reshape(-1).cpu().numpy()
    nz_flat = init_noise_cpu.detach().reshape(-1).cpu().numpy()
    h.update(cb_flat[: min(cb_flat.size, 2048)].tobytes())
    h.update(nz_flat[: min(nz_flat.size, 512)].tobytes())
    return h.hexdigest()[:16]

class CodebookGaussianDiffusion:
    def __init__(self, cfg, device):
        betas = build_beta_schedule(
            getattr(cfg, 'DIFF_NOISE_SCHED_INFER', 'linear'),
            int(cfg.DIFF_TIMESTEPS),
            float(cfg.DIFF_BETA_START),
            float(cfg.DIFF_BETA_END),
        )
        alphas = 1.0 - betas
        acp = np.cumprod(alphas, axis=0)
        acp_prev = np.append(1.0, acp[:-1])

        post_var = betas * (1.0 - acp_prev) / (1.0 - acp)
        post_log_var = np.log(np.maximum(post_var, 1e-20))

        self.clip_min = float(cfg.CODEBOOK_CLIP_MIN)
        self.clip_max = float(cfg.CODEBOOK_CLIP_MAX)

        self.acp = torch.tensor(acp, dtype=torch.float32, device=device)
        self.sqrt_acp = torch.tensor(np.sqrt(acp), dtype=torch.float32, device=device)
        self.sqrt_1m_acp = torch.tensor(np.sqrt(np.maximum(1.0 - acp, 1e-20)), dtype=torch.float32, device=device)
        self.sqrt_recip_acp = torch.tensor(np.sqrt(1.0 / acp), dtype=torch.float32, device=device)
        self.sqrt_recipm1_acp = torch.tensor(np.sqrt(1.0 / acp - 1), dtype=torch.float32, device=device)
        self.post_mean_c1 = torch.tensor(betas * np.sqrt(acp_prev) / (1.0 - acp), dtype=torch.float32, device=device)
        self.post_mean_c2 = torch.tensor((1.0 - acp_prev) * np.sqrt(alphas) / (1.0 - acp), dtype=torch.float32, device=device)
        self.post_var = torch.tensor(post_var, dtype=torch.float32, device=device)
        self.post_log_var = torch.tensor(post_log_var, dtype=torch.float32, device=device)

        codebook_path = resolve_codebook_path(cfg)
        codebook_seed = int(getattr(cfg, 'CODEBOOK_SEED', -1))
        if codebook_seed < 0:
            codebook_seed = int(cfg.SEED)
        shared_root = str(getattr(cfg, 'CODEBOOK_SHARED_ROOT', '')).strip()
        expected_codebook_shape = (
            int(cfg.DIFF_TIMESTEPS), int(cfg.CODEBOOK_SIZE), int(cfg.IN_CHANNELS), int(cfg.MAT_SIZE), int(cfg.MAT_SIZE)
        )
        expected_init_shape = (int(cfg.IN_CHANNELS), int(cfg.MAT_SIZE), int(cfg.MAT_SIZE))

        self.codebook_path = codebook_path
        self.codebook_seed = codebook_seed
        self.codebook_shared_root = os.path.abspath(shared_root) if shared_root else ''
        self.codebook_status = 'missing'
        self.codebook_fingerprint = ''
        self.codebook_meta = {}
        self.codebook_vector_norm_stats = {}

        need_regen = True
        if os.path.isfile(codebook_path):
            ck = load_torch(codebook_path, map_location='cpu')
            cb = ck.get('codebook', None)
            nz = ck.get('init_noise', None)
            if cb is not None and nz is not None:
                meta = dict(ck.get('meta', {}) or {})
                meta_seed = meta.get('seed', None)
                meta_norm = str(meta.get('codebook_norm', 'raw_gaussian')).strip().lower()
                seed_ok = meta_seed is None or int(meta_seed) == int(codebook_seed)
                norm_ok = meta_norm in ('', 'raw_gaussian')
                if tuple(cb.shape) == expected_codebook_shape and tuple(nz.shape) == expected_init_shape:
                    if seed_ok and norm_ok:
                        self.codebook = cb.to(device=device, dtype=torch.float32)
                        self.init_noise = nz.to(device=device, dtype=torch.float32)
                        self.codebook_status = 'loaded'
                        self.codebook_fingerprint = fingerprint_codebook_tensors(cb, nz)
                        self.codebook_vector_norm_stats = codebook_vector_norm_stats(cb)
                        meta.setdefault('codebook_norm', 'raw_gaussian')
                        meta.setdefault('unit_norm', False)
                        meta.setdefault('frozen', True)
                        meta.setdefault('trainable', False)
                        meta.setdefault('shape', list(expected_codebook_shape))
                        meta.setdefault('init_noise_shape', list(expected_init_shape))
                        meta.setdefault('vector_norm_stats', self.codebook_vector_norm_stats)
                        meta.setdefault('fingerprint', self.codebook_fingerprint)
                        self.codebook_meta = meta
                        need_regen = False
                    else:
                        logger.warning(
                            f'[codebook] metadata mismatch with current config, regenerate. '
                            f'found seed={meta_seed} norm={meta_norm or "<missing>"} '
                            f'expected seed={codebook_seed} norm=raw_gaussian'
                        )
                else:
                    logger.warning(
                        f'[codebook] shape mismatch with current config, regenerate. found codebook={tuple(cb.shape)}, '
                        f'init={tuple(nz.shape)}, expected codebook={expected_codebook_shape}, init={expected_init_shape}'
                    )
            else:
                logger.warning(f'[codebook] malformed checkpoint at {codebook_path}, regenerate.')

        if need_regen and str(getattr(cfg, 'CODEBOOK_PATH', '')).strip():
            raise ValueError(f'Explicit codebook is missing or incompatible: {codebook_path}')
        if need_regen:
            g = torch.Generator(device='cpu').manual_seed(codebook_seed)
            codebook_cpu = torch.randn(expected_codebook_shape, generator=g, dtype=torch.float32, device='cpu')
            init_noise_cpu = torch.randn(expected_init_shape, generator=g, dtype=torch.float32, device='cpu')
            norm_stats = codebook_vector_norm_stats(codebook_cpu)
            fingerprint = fingerprint_codebook_tensors(codebook_cpu, init_noise_cpu)
            ensure_dir(os.path.dirname(codebook_path))
            meta = {
                'timesteps': int(cfg.DIFF_TIMESTEPS),
                'codebook_size': int(cfg.CODEBOOK_SIZE),
                'in_channels': int(cfg.IN_CHANNELS),
                'mat_size': int(cfg.MAT_SIZE),
                'seed': int(codebook_seed),
                'shared_root': self.codebook_shared_root,
                'codebook_norm': 'raw_gaussian',
                'unit_norm': False,
                'frozen': True,
                'trainable': False,
                'shape': list(expected_codebook_shape),
                'init_noise_shape': list(expected_init_shape),
                'vector_norm_stats': norm_stats,
                'fingerprint': fingerprint,
            }
            torch.save({'codebook': codebook_cpu, 'init_noise': init_noise_cpu, 'meta': meta}, codebook_path)
            self.codebook = codebook_cpu.to(device)
            self.init_noise = init_noise_cpu.to(device)
            self.codebook_status = 'generated'
            self.codebook_meta = meta
            self.codebook_fingerprint = fingerprint
            self.codebook_vector_norm_stats = norm_stats

        self.codebook.requires_grad_(False)
        self.init_noise.requires_grad_(False)

        norm_stats = self.codebook_vector_norm_stats
        logger.info(
            f'[codebook] status={self.codebook_status} path={self.codebook_path} seed={self.codebook_seed} '
            f'shared_root={self.codebook_shared_root or "<none>"} fingerprint={self.codebook_fingerprint} '
            f'norm=raw_gaussian unit_norm=false shape={expected_codebook_shape} '
            f'vector_norm_mean={norm_stats.get("mean", float("nan")):.6f} '
            f'vector_norm_std={norm_stats.get("std", float("nan")):.6f} '
            f'vector_norm_min={norm_stats.get("min", float("nan")):.6f} '
            f'vector_norm_max={norm_stats.get("max", float("nan")):.6f}'
        )

    def _extract(self, a, t, x_shape):
        return a[t].reshape(x_shape[0], 1, 1, 1)

    def predict_start_from_noise(self, xt, t, noise):
        return self._extract(self.sqrt_recip_acp, t, xt.shape) * xt - self._extract(self.sqrt_recipm1_acp, t, xt.shape) * noise

    def q_posterior(self, h0, xt, t):
        mean = self._extract(self.post_mean_c1, t, xt.shape) * h0 + self._extract(self.post_mean_c2, t, xt.shape) * xt
        var = self._extract(self.post_var, t, xt.shape)
        log_var = self._extract(self.post_log_var, t, xt.shape)
        return mean, var, log_var

    def p_sample_with_index(self, pred, x, t, best_idx, clip_denoised=True, pred_mode='eps'):
        pred_mode = normalize_pred_mode(pred_mode)
        h0 = pred if pred_mode == 'h0' else self.predict_start_from_noise(x, t, pred)
        if clip_denoised:
            h0 = torch.clamp(h0, self.clip_min, self.clip_max)

        mean, _, log_var = self.q_posterior(h0, x, t)
        t_idx = int(t[0].item())
        noise = self.codebook[t_idx][best_idx]
        nonzero = (t != 0).float().view(-1, 1, 1, 1)
        x_prev = mean + nonzero * (0.5 * log_var).exp() * noise
        return x_prev, h0

    def p_sample_with_fixed_h0_and_index(self, h0, x_t, t_idx, best_idx, eta=1.0):
        tt = torch.full((x_t.shape[0],), int(t_idx), device=x_t.device, dtype=torch.long)
        mean, _, log_var = self.q_posterior(h0, x_t, tt)
        noise = self.codebook[int(t_idx)][best_idx]
        return mean + float(eta) * (0.5 * log_var).exp() * noise

@torch.no_grad()
def select_codebook_index(gdf, t_idx, resid):
    cb_flat = gdf.codebook[t_idx].reshape(gdf.codebook[t_idx].size(0), -1)
    resid_flat = resid.reshape(resid.size(0), -1)
    score = cb_flat @ resid_flat.t()
    return torch.argmax(score, dim=0)


def predict_h0(gdf, x_t, t_tensor, out, pred_mode, clip_flag):
    pred_mode = normalize_pred_mode(pred_mode)
    h0_pred = out if pred_mode == 'h0' else gdf.predict_start_from_noise(x_t, t_tensor, out)
    if clip_flag:
        h0_pred = torch.clamp(h0_pred, gdf.clip_min, gdf.clip_max)
    return h0_pred
