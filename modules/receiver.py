"""Decode index payloads without access to source CSI or evaluation datasets."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

from modules.diffusion import amp_autocast, set_seed

CONTRACT_FIELDS = ['DIFF_TIMESTEPS', 'CODEBOOK_SIZE', 'CODEBOOK_SEED', 'SCENE', 'GEN_MODE', 'DIFF_MODEL_DIM', 'DIFF_MODEL_BLOCKS', 'DIFF_MODEL_ATTN_EVERY', 'DIFF_NOISE_SCHED_INFER', 'DIFF_MACRO_STRIDE', 'DIFF_MACRO_TAIL_STEPS', 'DIFF_MACRO_REFRESH_MODE', 'DIFF_MACRO_REFRESH_T_LIST', 'DIFF_INFER_STEPS', 'DIFF_INFER_SAMPLER', 'DIFF_DDIM_ETA', 'SEED', 'MAT_SIZE', 'IN_CHANNELS', 'DIFF_PRED_MODE', 'DIFF_MODEL_ATTN_HEADS', 'DIFF_MODEL_ATTN_DOWNSAMPLE', 'DIFF_MODEL_ATTN_MAX_HW', 'DIFF_BETA_START', 'DIFF_BETA_END', 'DIFF_MACRO_ETA', 'DIFF_MACRO_MULTI_INDEX', 'DIFF_MACRO_REFRESH_RATIO', 'DIFF_MACRO_REFRESH_COUNT', 'DIFF_MACRO_REFRESH_MIN_SPAN', 'DIFF_INFER_USE_AMP', 'DIFF_CHANNELS_LAST', 'DIFF_ALLOW_TF32']


def save_contract(index_path, cfg):
    from modules.artifacts import file_sha256
    md = sio.loadmat(index_path)
    md = {k: v for k,v in md.items() if not k.startswith('__')}
    contract = {k: getattr(cfg, k) for k in CONTRACT_FIELDS}
    contract['checkpoint_sha256'] = file_sha256(cfg.DIFF_CKPT_PATH)
    contract['normalization_sha256'] = file_sha256(cfg.DIFF_NORM_STATS_PATH)
    if cfg.CODEBOOK_PATH:
        contract['codebook_sha256'] = file_sha256(cfg.CODEBOOK_PATH)
    md['release_contract_json'] = json.dumps(contract, sort_keys=True)
    sio.savemat(index_path, md)


def validate_indices(index_path, cfg):
    from modules.artifacts import file_sha256
    from modules.csicogen_codec import metadata_string
    md = sio.loadmat(index_path)
    idx = np.asarray(md['best_idx_seq'])
    if idx.ndim != 2 or idx.shape[0] != cfg.DIFF_TIMESTEPS or idx.shape[1] == 0:
        raise ValueError(f'Invalid index array shape: {idx.shape}; expected [T,N]')
    if not np.issubdtype(idx.dtype, np.integer) or np.any(idx < 0) or np.any(idx >= cfg.CODEBOOK_SIZE):
        raise ValueError('Indices must be integers in [0,K)')
    if 'release_contract_json' in md:
        contract = json.loads(metadata_string(md, 'release_contract_json'))
        for key in CONTRACT_FIELDS:
            if contract[key] != getattr(cfg, key):
                raise ValueError(f'Index configuration mismatch for {key}: encoded={contract[key]}, decoder={getattr(cfg,key)}')
        for field, path in [('checkpoint_sha256',cfg.DIFF_CKPT_PATH), ('normalization_sha256',cfg.DIFF_NORM_STATS_PATH)]:
            if contract[field] != file_sha256(path):
                raise ValueError(f'Index payload uses a different {field}')
        if contract.get('codebook_sha256') and (not cfg.CODEBOOK_PATH or contract['codebook_sha256'] != file_sha256(cfg.CODEBOOK_PATH)):
            raise ValueError('Index payload uses a different codebook')
    return md


def decode_without_targets(codec, checkpoint, indices, output, prefix_length=None):
    from modules.csicogen_codec import CsiCoGenCodec, CodebookGaussianDiffusion, iter_feedback_blocks, metadata_string, metadata_scalar
    cfg, device = codec.cfg, codec.device
    set_seed(cfg.SEED, deterministic=True)
    md = validate_indices(indices, cfg)
    idx = torch.from_numpy(md['best_idx_seq'].astype(np.int64))
    mean, std = codec._load_norm_stats()
    T, n = idx.shape
    if not isinstance(codec, CsiCoGenCodec):
        if prefix_length is not None:
            raise ValueError('Use --prefix-lengths with Turbo infer for progressive evaluation')
        reconstruction, meta = codec._decode_indices_core(checkpoint, idx, mean, std)
    else:
        length = T-1 if prefix_length is None else int(prefix_length)
        if not 0 <= length <= T-1:
            raise ValueError(f'Prefix length must be between 0 and {T-1}')
        if 'sample_timesteps' in md:
            feedback = [int(x) for x in md['sample_timesteps'].reshape(-1)]
            anchors = [int(x) for x in md['denoise_timesteps'].reshape(-1)]
            sampler = metadata_string(md, 'infer_sampler')
            eta = float(metadata_scalar(md, 'ddim_eta'))
            row = {t: i for i,t in enumerate(feedback)}
        else:
            feedback = anchors = list(range(T-1,-1,-1))
            sampler, eta = cfg.DIFF_INFER_SAMPLER, cfg.DIFF_DDIM_ETA
            row = {t:t for t in feedback}
        if prefix_length is not None and len(anchors) != T:
            raise ValueError('Arbitrary receiver prefixes currently use the full CsiCoGen sampler. Omit --denoiser-calls.')
        net = codec._build_net(checkpoint)
        gdf = CodebookGaussianDiffusion(cfg, device)
        reconstruction = np.empty((n, cfg.IN_CHANNELS, cfg.MAT_SIZE, cfg.MAT_SIZE), dtype=np.float32)
        amp = cfg.DIFF_INFER_USE_AMP and device.type == 'cuda'
        channels_last = cfg.DIFF_CHANNELS_LAST and device.type == 'cuda'
        started = time.perf_counter()
        with torch.inference_mode():
            for st in range(0,n,cfg.DIFF_INFER_BATCH):
                ed = min(st+cfg.DIFF_INFER_BATCH,n)
                samples = gdf.init_noise.unsqueeze(0).repeat(ed-st,1,1,1)
                if channels_last:
                    samples = samples.contiguous(memory_format=torch.channels_last)
                consumed = 0
                for anchor, next_anchor, block in iter_feedback_blocks(anchors,feedback):
                    tt = torch.full((ed-st,),anchor,device=device,dtype=torch.long)
                    with amp_autocast(enabled=amp):
                        out = net(samples,tt)
                    prediction = out.float() if cfg.DIFF_PRED_MODE in ('x0','h0') else gdf.predict_start_from_noise(samples,tt,out.float())
                    if cfg.CODEBOOK_CLIP:
                        prediction = prediction.clamp(gdf.clip_min,gdf.clip_max)
                    if prefix_length is not None and consumed == length:
                        samples = prediction
                        break
                    for i,t in enumerate(block):
                        prev = block[i+1] if i+1 < len(block) else next_anchor
                        tt = torch.full((ed-st,),t,device=device,dtype=torch.long)
                        samples,_ = gdf.sample_with_index(prediction,samples,tt,prev,idx[row[t],st:ed].to(device),sampler,eta,cfg.CODEBOOK_CLIP,'x0')
                        consumed += int(t > 0)
                reconstruction[st:ed] = (samples*(std+1e-7)+mean).cpu().numpy()
        meta = {'num_samples': n, 'prefix_length':length, 'feedback_bits':length*int(math.log2(cfg.CODEBOOK_SIZE)),
                'elapsed_sec':time.perf_counter()-started, 'source_csi_required':False}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    rec_key = 'ldm_ddpm_rec'
    sio.savemat(output, {rec_key:reconstruction.transpose(0,2,3,1)})
    return meta
