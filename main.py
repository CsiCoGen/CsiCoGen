#!/usr/bin/env python3
"""Train, download, encode, decode, and evaluate CsiCoGen and CsiCoGen-Turbo."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODES = ['CsiCoGen', 'CsiCoGen-Lite', 'ddpm', 'ddpm-turbo']


def build_cfg(args):
    from modules.config import CFG
    cfg = CFG()
    cfg.load_yaml(str(ROOT/'configs/base.yaml'), strict=True)
    cfg.load_yaml(str(ROOT/f'configs/scenes/{args.scene}.yaml'), strict=True)
    if args.mode in ('CsiCoGen', 'CsiCoGen-Lite'):
        cfg.load_yaml(str(ROOT/f'configs/models/{args.mode}_T{args.timesteps or 200}.yaml'), strict=True)
    else:
        cfg.load_yaml(str(ROOT/f'configs/modes/{args.mode.replace("-", "_")}.yaml'), strict=True)
    if args.config:
        cfg.load_yaml(args.config, strict=True)
    if args.timesteps is not None:
        cfg.DIFF_TIMESTEPS = args.timesteps
    for attr, field in [('infer_batch','DIFF_INFER_BATCH'), ('epochs','DIFF_EPOCHS'),
                        ('batch','DIFF_BATCH'), ('train_samples','DIFF_TRAIN_SAMPLES'),
                        ('codebook_size','CODEBOOK_SIZE')]:
        value = getattr(args, attr, None)
        if value is not None:
            setattr(cfg, field, value)
    cfg.DIFF_MAX_SAMPLES = getattr(args, 'max_samples', 0)
    if cfg.DIFF_MAX_SAMPLES < 0:
        raise ValueError('--max-samples must be nonnegative')
    cfg.OUTPUT_ROOT = str(Path(args.output_root).expanduser().resolve())
    cfg.RUN_TAG = args.run_tag or f'{args.mode}_T{cfg.DIFF_TIMESTEPS}_{args.command}_{datetime.now():%Y%m%d_%H%M%S}'
    cfg.CODEBOOK_SHARED_ROOT = str(Path(args.checkpoint_root)/'generated_codebooks')
    if args.mode in ('CsiCoGen', 'CsiCoGen-Lite'):
        calls = getattr(args, 'denoiser_calls', None)
        if calls is not None:
            if not 2 <= calls <= cfg.DIFF_TIMESTEPS:
                raise ValueError('--denoiser-calls must be between 2 and T')
            cfg.DIFF_INFER_STEPS = calls
            cfg.DIFF_INFER_FEEDBACK_STEPS = cfg.DIFF_TIMESTEPS
            cfg.DIFF_INFER_SAMPLER = 'ddpm' if calls == cfg.DIFF_TIMESTEPS else 'ddim'
            cfg.DIFF_DDIM_ETA = 0.0 if calls == cfg.DIFF_TIMESTEPS else 1.0
    elif args.command != 'train':
        apply_turbo_schedule(cfg, args.schedule or ('s30' if 'turbo' in args.mode else 'full'))
    if cfg.CODEBOOK_SIZE < 2 or cfg.CODEBOOK_SIZE > 65536 or cfg.CODEBOOK_SIZE & (cfg.CODEBOOK_SIZE - 1):
        raise ValueError('Codebook size must be a power of two between 2 and 65536')
    if getattr(args, 'prefix_lengths', None):
        cfg.DIFF_PREFIX_LENGTHS = args.prefix_lengths
    if getattr(args, 'no_amp', False):
        cfg.DIFF_INFER_USE_AMP = False
    cfg._recompute()
    return cfg


def apply_turbo_schedule(cfg, name):
    from modules.schedules import TurboSchedule
    specs = {'full': f's1:k{cfg.CODEBOOK_SIZE}:t1:none',
             's33': f's33:k{cfg.CODEBOOK_SIZE}:t1:none',
             's25': f's25:k{cfg.CODEBOOK_SIZE}:t1:none',
             's30': f's30:k{cfg.CODEBOOK_SIZE}:t1:c1:r0.50:m8:p84-54-19'}
    if name == 's30' and cfg.DIFF_TIMESTEPS != 100:
        raise ValueError('The s30 preset is defined for T=100. Use a custom schedule specification for other T.')
    schedule = TurboSchedule.from_spec(specs.get(name, name), timesteps=cfg.DIFF_TIMESTEPS)
    for field, value in [('DIFF_MACRO_STRIDE',schedule.stride), ('DIFF_MACRO_TAIL_STEPS',schedule.tail_steps),
                         ('DIFF_MACRO_ETA',schedule.macro_eta), ('DIFF_MACRO_MULTI_INDEX',schedule.multi_index),
                         ('DIFF_MACRO_REFRESH_MODE',schedule.refresh_mode), ('DIFF_MACRO_REFRESH_COUNT',schedule.refresh_count),
                         ('DIFF_MACRO_REFRESH_RATIO',schedule.refresh_ratio), ('DIFF_MACRO_REFRESH_MIN_SPAN',schedule.refresh_min_span),
                         ('DIFF_MACRO_REFRESH_T_LIST',schedule.refresh_t_list), ('CODEBOOK_SIZE',schedule.codebook_size)]:
        setattr(cfg, field, value)


def prepare_assets(cfg, args):
    from modules.artifacts import resolve_checkpoint, resolve_codebook, resolve_dataset
    inference = args.command != 'train'
    if inference:
        if args.ckpt:
            cfg.DIFF_CKPT_PATH = str(Path(args.ckpt).expanduser().resolve())
        if args.norm_stats:
            cfg.DIFF_NORM_STATS_PATH = str(Path(args.norm_stats).expanduser().resolve())
        if not cfg.DIFF_CKPT_PATH:
            family = 'CsiCoGen-Lite' if args.mode == 'CsiCoGen-Lite' else 'CsiCoGen'
            cfg.DIFF_CKPT_PATH, cfg.DIFF_NORM_STATS_PATH = resolve_checkpoint(
                family, cfg.SCENE, cfg.DIFF_TIMESTEPS, args.checkpoint_root, args.local_files_only)
        if not cfg.DIFF_NORM_STATS_PATH:
            raise ValueError('A custom checkpoint requires --norm-stats or DIFF_NORM_STATS_PATH in its config.')
        for path in [cfg.DIFF_CKPT_PATH, cfg.DIFF_NORM_STATS_PATH]:
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        if args.codebook:
            cfg.CODEBOOK_PATH = str(Path(args.codebook).expanduser().resolve())
        elif not cfg.CODEBOOK_PATH and cfg.CODEBOOK_SIZE == 256 and cfg.DIFF_TIMESTEPS in (100,200):
            cfg.CODEBOOK_PATH = resolve_codebook(cfg.DIFF_TIMESTEPS, args.checkpoint_root, args.local_files_only)
    need_test = args.command in ('infer', 'encode') or getattr(args, 'evaluate', False)
    if (need_test and not cfg.TEST_FILE) or (not inference and not cfg.TRAIN_FILE):
        data_root = resolve_dataset(cfg.SCENE, 'test' if inference else 'all', args.data_root, args.local_files_only)
    else:
        data_root = args.data_root
    suffix = 'in' if cfg.SCENE == 'indoor' else 'out'
    cfg.TRAIN_FILE = cfg.TRAIN_FILE or str(Path(data_root)/f'DATA_Htrain{suffix}.mat')
    cfg.TEST_FILE = cfg.TEST_FILE or str(Path(data_root)/f'DATA_Htest{suffix}.mat')
    if not cfg.GT_RAW_FILE and Path(cfg.TEST_FILE).name == f'DATA_Htest{suffix}.mat':
        cfg.GT_RAW_FILE = str(Path(data_root)/f'DATA_HtestF{suffix}_all.mat')
    cfg._recompute()


def command_runtime(args):
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    cfg = build_cfg(args)
    prepare_assets(cfg, args)
    import torch
    import yaml
    from modules.diffusion import set_seed
    from utils import logger
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    set_seed(cfg.SEED, deterministic=True)
    cfg.ensure_dirs()
    logger.set_file(str(Path(cfg.RUN_DIR)/'run.log'))
    (Path(cfg.RUN_DIR)/'config_used.yaml').write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))
    if args.command == 'train':
        from modules.training import DiffusionTrainer
        DiffusionTrainer(cfg, device).train()
        return
    from modules.csicogen_codec import CsiCoGenCodec
    from modules.codec import CodebookCodec
    from modules.receiver import decode_without_targets, save_contract, validate_indices
    codec = CsiCoGenCodec(cfg, device) if args.mode in ('CsiCoGen','CsiCoGen-Lite') else CodebookCodec(cfg, device)
    out = Path(cfg.DIFF_INFERENCE_DIR)
    idx = str(Path(args.indices).expanduser().resolve()) if args.indices else str(out/'indices.mat')
    rec = str(out/'reconstruction.mat')
    metrics = {}
    if args.command in ('infer','encode'):
        if isinstance(codec, CsiCoGenCodec):
            metrics['encode_sec'] = codec.encode_indices(cfg.DIFF_CKPT_PATH, idx, cfg.TEST_FILE, max_samples=args.max_samples)
        else:
            codec.encode_indices(cfg.DIFF_CKPT_PATH, idx, cfg.TEST_FILE)
            metrics.update(json.loads(Path(idx).with_name(Path(idx).stem+'_encode.json').read_text()))
        save_contract(idx, cfg)
    if args.command in ('infer','decode'):
        if args.command == 'decode' and not args.indices:
            raise ValueError('decode requires --indices')
        validate_indices(idx, cfg)
        if args.prefix_length is not None or (args.command == 'decode' and not args.evaluate):
            result = decode_without_targets(codec, cfg.DIFF_CKPT_PATH, idx, rec, args.prefix_length)
        else:
            result = codec.decode_from_indices(cfg.DIFF_CKPT_PATH, idx, rec, return_metrics=True)
        metrics.update(result or {})
        if args.prefix_length is not None and (args.command == 'infer' or args.evaluate):
            from modules.evaluation import evaluate_files
            metrics.update(evaluate_files(rec, cfg.TEST_FILE, cfg.GT_RAW_FILE, device))
    length = args.prefix_length if args.prefix_length is not None else cfg.DIFF_TIMESTEPS-1
    metrics.update(mode=args.mode, scene=cfg.SCENE, timesteps=cfg.DIFF_TIMESTEPS,
                   checkpoint=cfg.DIFF_CKPT_PATH, indices=idx)
    metrics.setdefault('feedback_bits', length*int(math.log2(cfg.CODEBOOK_SIZE)))
    from modules.artifacts import file_sha256
    metrics['checkpoint_sha256'] = file_sha256(cfg.DIFF_CKPT_PATH)
    metrics['environment'] = {'torch':torch.__version__, 'cuda':torch.version.cuda, 'device':str(device)}
    (out/'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(f'Results: {out}')
    print(json.dumps({k:v for k,v in metrics.items() if k in ('nmse','rho','feedback_bits','encode_sec','elapsed_sec')}, indent=2))


def command_download(args):
    from modules.artifacts import resolve_checkpoint, resolve_codebook, resolve_dataset
    if args.asset in ('data','all'):
        resolve_dataset(args.scene, args.split, args.data_root, args.local_files_only)
    if args.asset in ('models','all'):
        for family in (['CsiCoGen','CsiCoGen-Lite'] if args.family == 'all' else [args.family]):
            for scene in (['indoor','outdoor'] if args.scene == 'all' else [args.scene]):
                for t in ([100,200] if args.timesteps is None else [args.timesteps]):
                    resolve_checkpoint(family, scene, t, args.checkpoint_root, args.local_files_only)
                    resolve_codebook(t, args.checkpoint_root, args.local_files_only)
    print('Download and SHA-256 verification complete.')


def asset_args(p):
    p.add_argument('--data-root', default=os.getenv('CSICOGEN_DATA_ROOT', str(ROOT/'data/COST2100')))
    p.add_argument('--checkpoint-root', default=os.getenv('CSICOGEN_CHECKPOINT_ROOT', str(ROOT/'checkpoints')))
    p.add_argument('--local-files-only', action='store_true')


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    download = sub.add_parser('download', help='Download verified model and data assets')
    download.add_argument('asset', choices=['models','data','all'])
    download.add_argument('--family', choices=['CsiCoGen','CsiCoGen-Lite','all'], default='CsiCoGen')
    download.add_argument('--scene', choices=['indoor','outdoor','all'], default='all')
    download.add_argument('--timesteps', type=int, choices=[100,200])
    download.add_argument('--split', choices=['train','val','test','all'], default='all')
    asset_args(download); download.set_defaults(func=command_download)
    for action in ['train','infer','encode','decode']:
        s = sub.add_parser(action)
        s.add_argument('--mode', choices=MODES, default='CsiCoGen')
        s.add_argument('--scene', choices=['indoor','outdoor'], required=True)
        s.add_argument('--timesteps', type=int, choices=[100,200])
        s.add_argument('--config', default='')
        s.add_argument('--gpu', default=None)
        s.add_argument('--cpu', action='store_true')
        s.add_argument('--output-root', default=str(ROOT/'outputs'))
        s.add_argument('--run-tag', default='')
        s.add_argument('--codebook-size', type=int)
        asset_args(s)
        if action == 'train':
            s.add_argument('--epochs', type=int)
            s.add_argument('--batch', type=int)
            s.add_argument('--train-samples', type=int)
        else:
            s.add_argument('--ckpt', default='')
            s.add_argument('--norm-stats', default='')
            s.add_argument('--codebook', default='')
            s.add_argument('--infer-batch', type=int)
            s.add_argument('--max-samples', type=int, default=0, help='0 means the full test set')
            s.add_argument('--denoiser-calls', type=int, help='CsiCoGen D/F schedule; F=T stays fixed')
            s.add_argument('--schedule', default=None, help='Turbo: full, s30, s25, s33, or custom spec')
            s.add_argument('--prefix-lengths', default=None, help='Turbo progressive evaluation: full/all or comma-separated lengths')
            s.add_argument('--prefix-length', type=int, help='Decode a CsiCoGen/CsiCoGen-Lite prefix containing L indices')
            s.add_argument('--indices', default='')
            s.add_argument('--no-amp', action='store_true')
            s.add_argument('--evaluate', action='store_true', help='Compute test metrics in decode-only mode')
        s.set_defaults(func=command_runtime)
    return p


if __name__ == '__main__':
    args = build_parser().parse_args()
    args.func(args)
