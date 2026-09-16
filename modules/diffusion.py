"""Shared model, data, and diffusion helpers."""

import json
import os
import random
import re

import numpy as np
import scipy.io as sio
import torch

from models.net import CSI_ResAttnNet

def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def load_torch(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)

def count_model_params(model):
    return int(sum(p.numel() for p in model.parameters()))

def normalize_gen_mode(value):
    mode = str(value).strip().lower()
    if mode in ('', 'ddpm', 'diffusion'):
        return 'ddpm'
    raise ValueError(f'Unsupported GEN_MODE: {value} (use ddpm)')

def normalize_pred_mode(value):
    mode = str(value).strip().lower().replace('-', '_')
    if mode in ('', 'h0', 'x0'):
        return 'h0'
    if mode in ('eps', 'epsilon', 'noise'):
        return 'eps'
    raise ValueError(f'Unsupported prediction mode: {value} (use h0 or eps)')

def checkpoint_names(cfg):
    normalize_gen_mode(getattr(cfg, 'GEN_MODE', 'ddpm'))
    name = cfg.MODEL_NAME
    if name not in ('csicogen', 'csicogen_lite'):
        raise ValueError(f'Unsupported MODEL_NAME: {name}')
    return f'{name}.ckpt', f'{name}_ema.ckpt'

def should_save_periodic_checkpoint(epoch, save_every):
    save_every = int(save_every)
    return save_every > 0 and (int(epoch) % save_every) == 0

def normalize_model_arch(value):
    arch = str(value).strip().lower().replace('-', '_')
    if arch in ('', 'resattn', 'res_attn', 'resattn2d'):
        return 'resattn'
    raise ValueError(f'Unsupported model arch: {value} (use resattn)')


def resolve_model_spec(cfg, prefix='DIFF_MODEL'):
    return {
        'arch': normalize_model_arch(getattr(cfg, f'{prefix}_ARCH', 'resattn')),
        'preset': '',
        'dim': int(getattr(cfg, f'{prefix}_DIM', 128)),
        'blocks': int(getattr(cfg, f'{prefix}_BLOCKS', 8)),
        'attn_every': int(getattr(cfg, f'{prefix}_ATTN_EVERY', 2)),
        'attn_heads': int(getattr(cfg, f'{prefix}_ATTN_HEADS', 4)),
        'attn_downsample': int(getattr(cfg, f'{prefix}_ATTN_DOWNSAMPLE', 2)),
        'attn_max_hw': int(getattr(cfg, f'{prefix}_ATTN_MAX_HW', 8)),
        'expansion': int(getattr(cfg, f'{prefix}_EXPANSION', 2)),
    }

def model_metadata_from_cfg(cfg, prefix='DIFF_MODEL', params=None):
    spec = resolve_model_spec(cfg, prefix=prefix)
    meta = {
        'model_arch': spec['arch'],
        'model_preset': spec['preset'],
        'model_dim': int(spec['dim']),
        'model_blocks': int(spec['blocks']),
        'model_attn_every': int(spec['attn_every']),
        'model_attn_heads': int(spec['attn_heads']),
        'model_attn_downsample': int(spec['attn_downsample']),
        'model_attn_max_hw': int(spec['attn_max_hw']),
        'model_expansion': int(spec['expansion']),
    }
    if params is not None:
        meta['model_params'] = int(params)
    return meta

def normalize_csi_array(x, mean, std):
    return (x - float(mean)) / (float(std) + 1e-7)

def inverse_normalize_csi_array(x, mean, std):
    return x * (float(std) + 1e-7) + float(mean)

def save_norm_stats(path, mean, std, source_file='', tensor_shape=None, gen_mode='ddpm'):
    ensure_dir(os.path.dirname(path))
    tensor_shape = [] if tensor_shape is None else [int(v) for v in tensor_shape]
    np.savez(
        path,
        mean=float(mean),
        std=float(std),
        source_file=str(source_file),
        tensor_shape=np.asarray(tensor_shape, dtype=np.int64),
        tensor_layout='NCHW',
        normalized_tensor='H0',
        normalization_formula='H0=(HT-mean)/(std+1e-7)',
        inverse_formula='HT_hat=H_hat*(std+1e-7)+mean',
        gen_mode=normalize_gen_mode(gen_mode),
    )

def load_norm_stats_file(path):
    with np.load(path, allow_pickle=False) as st:
        mean, std = float(st['mean']), float(st['std'])
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError(f'Invalid normalization statistics: {path}')
    return mean, std

def write_json_manifest(path, payload):
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write('\n')

def array_scalar(md, key, default=None):
    if key not in md:
        return default
    arr = np.asarray(md[key])
    if arr.size == 0:
        return default
    return float(arr.reshape(-1)[0])

def load_ht_mat(path, mat_size, in_ch):
    mat = sio.loadmat(path)
    if 'HT' in mat:
        x = mat['HT']
    else:
        keys = [k for k in mat.keys() if not k.startswith('__')]
        if not keys:
            raise ValueError(f'No data found in MAT: {path}')
        x = mat[keys[0]]
    x = np.asarray(x)
    n = x.shape[0]

    if x.ndim == 2 and x.shape[1] == in_ch * mat_size * mat_size:
        x = x.reshape(n, in_ch, mat_size, mat_size)
        return x.astype(np.float32)

    if x.ndim == 4 and x.shape[1] == in_ch and x.shape[2] == mat_size and x.shape[3] == mat_size:
        return x.astype(np.float32)

    if x.ndim == 4 and x.shape[1] == mat_size and x.shape[2] == mat_size and x.shape[3] == in_ch:
        x = x.transpose(0, 3, 1, 2)
        return x.astype(np.float32)

    raise ValueError(f'Unsupported HT shape {x.shape} for mat_size={mat_size}, in_ch={in_ch}')

def load_raw_hf_all(path):
    mat = sio.loadmat(path)
    if 'HF_all' not in mat:
        keys = [k for k in mat.keys() if not k.startswith('__')]
        if not keys:
            raise ValueError(f'No raw HF data found in MAT: {path}')
        raw = mat[keys[0]]
    else:
        raw = mat['HF_all']

    raw = np.asarray(raw)
    if raw.ndim == 4 and raw.shape[-1] == 2:
        return torch.tensor(raw, dtype=torch.float32)
    if raw.ndim == 3:
        real = torch.tensor(np.real(raw), dtype=torch.float32)
        imag = torch.tensor(np.imag(raw), dtype=torch.float32)
        return torch.stack((real, imag), dim=-1)
    if raw.ndim == 2:
        if raw.shape[1] != 32 * 125:
            raise ValueError(f'Unsupported HF_all flattened shape: {raw.shape}')
        raw = raw.reshape(raw.shape[0], 32, 125)
        real = torch.tensor(np.real(raw), dtype=torch.float32)
        imag = torch.tensor(np.imag(raw), dtype=torch.float32)
        return torch.stack((real, imag), dim=-1)
    raise ValueError(f'Unsupported HF_all shape: {raw.shape}')

def parse_diff_epoch_from_path(path):
    m = re.search(r'ep(\d+)_loss', path)
    return int(m.group(1)) if m else 0

def make_grad_scaler(enabled: bool):
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        return torch.amp.GradScaler('cuda', enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)

def amp_autocast(enabled: bool):
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
        return torch.amp.autocast('cuda', enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)

def build_beta_schedule(schedule, timesteps, beta_start, beta_end):
    schedule = str(schedule).lower()
    timesteps = int(timesteps)
    beta_start = float(beta_start)
    beta_end = float(beta_end)

    if schedule == 'linear':
        return np.linspace(beta_start, beta_end, timesteps, dtype=np.float64)

    if schedule == 'cosine':
        # Improved DDPM cosine schedule
        s = 0.008
        steps = timesteps + 1
        x = np.linspace(0, timesteps, steps, dtype=np.float64)
        acp = np.cos(((x / timesteps) + s) / (1 + s) * np.pi / 2) ** 2
        acp = acp / acp[0]
        betas = 1 - (acp[1:] / acp[:-1])
        return np.clip(betas, 1e-8, 0.999)

    raise ValueError(f'Unsupported diffusion noise schedule: {schedule} (use linear or cosine)')

def build_unet_from_cfg(cfg, device, prefix='DIFF_MODEL'):
    in_channels = int(getattr(cfg, f'{prefix}_IN_CHANNELS', getattr(cfg, 'IN_CHANNELS', 2)))
    out_channels = int(getattr(cfg, f'{prefix}_OUT_CHANNELS', in_channels))
    spec = resolve_model_spec(cfg, prefix=prefix)
    kwargs = dict(
        mat_size=cfg.MAT_SIZE,
        img_channels=in_channels,
        out_channels=out_channels,
        dim=spec['dim'],
        num_blocks=spec['blocks'],
        attn_every=spec['attn_every'],
        attn_heads=spec['attn_heads'],
        attn_downsample=spec['attn_downsample'],
        attn_max_hw=spec['attn_max_hw'],
    )
    net = CSI_ResAttnNet(**kwargs).to(device)
    if device.type == 'cuda' and bool(getattr(cfg, 'DIFF_CHANNELS_LAST', True)):
        net = net.to(memory_format=torch.channels_last)
    return net

class GaussianDiffusion:
    def __init__(self, beta_start, beta_end, timesteps, device, schedule='linear'):
        betas = build_beta_schedule(schedule, timesteps, beta_start, beta_end)
        alphas = 1.0 - betas
        acp = np.cumprod(alphas, axis=0)
        self.sqrt_acp = torch.tensor(np.sqrt(acp), dtype=torch.float32, device=device)
        self.sqrt_1m_acp = torch.tensor(np.sqrt(1.0 - acp), dtype=torch.float32, device=device)

    def _extract(self, a, t, x_shape):
        return a[t].reshape(x_shape[0], 1, 1, 1)

    def q_sample(self, h0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(h0)
        return self._extract(self.sqrt_acp, t, h0.shape) * h0 + self._extract(self.sqrt_1m_acp, t, h0.shape) * noise

class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}

    @torch.no_grad()
    def copy_to(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                p.copy_(self.shadow[name])
