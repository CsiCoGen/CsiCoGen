"""Download pinned, checksum-verified model and COST2100 assets."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'configs' / 'artifacts.json'


def manifest():
    return json.loads(MANIFEST.read_text())


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def resolve_file(repo, revision, repo_type, filename, sha256, local_root, offline=False):
    target = Path(local_root).expanduser() / filename
    if target.is_file():
        if file_sha256(target) != sha256:
            raise ValueError(f'Checksum mismatch: {target}. Remove the damaged file and download again.')
        return str(target.resolve())
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id=repo, repo_type=repo_type, revision=revision,
        filename=filename, local_dir=str(Path(local_root).expanduser()),
        local_files_only=offline,
    )
    if file_sha256(path) != sha256:
        raise ValueError(f'Download checksum mismatch: {path}')
    return path


def resolve_checkpoint(family, scene, timesteps, local_root=None, offline=False):
    data = manifest()
    key = f'{family}-{scene}-T{timesteps}'
    if key not in data['checkpoints']:
        raise ValueError(f'No published checkpoint for {key}. Supply --ckpt and --norm-stats.')
    local_root = local_root or os.getenv('CSICOGEN_CHECKPOINT_ROOT', str(ROOT/'checkpoints'))
    paths = {}
    for name, spec in data['checkpoints'][key]['files'].items():
        paths[name] = resolve_file(data['model_repo'], data['model_revision'], 'model',
                                   spec['path'], spec['sha256'], local_root, offline)
    checkpoint = 'csicogen_lite.ckpt' if family == 'CsiCoGen-Lite' else 'csicogen.ckpt'
    return paths[checkpoint], paths['normalization.npz']


def resolve_codebook(timesteps, local_root=None, offline=False):
    data = manifest()
    spec = data['codebooks'][str(timesteps)]
    local_root = local_root or os.getenv('CSICOGEN_CHECKPOINT_ROOT', str(ROOT/'checkpoints'))
    return resolve_file(data['model_repo'], data['model_revision'], 'model',
                        spec['path'], spec['sha256'], local_root, offline)


def resolve_dataset(scene, split='test', local_root=None, offline=False):
    data = manifest()
    local_root = local_root or os.getenv('CSICOGEN_DATA_ROOT', str(ROOT/'data/COST2100'))
    scenes = ['indoor', 'outdoor'] if scene == 'all' else [scene]
    wanted = []
    for sc in scenes:
        suffix = 'in' if sc == 'indoor' else 'out'
        for part in (['train', 'val', 'test'] if split == 'all' else [split]):
            wanted.append(f'DATA_H{part}{suffix}.mat')
            if part == 'test':
                wanted.append(f'DATA_HtestF{suffix}_all.mat')
    for spec in data['dataset_files']:
        if spec['file'] in wanted:
            resolve_file(data['dataset_repo'], data['dataset_revision'], 'dataset',
                         spec['file'], spec['sha256'], local_root, offline)
    return str(Path(local_root).expanduser().resolve())
