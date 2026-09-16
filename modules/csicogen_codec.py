"""CsiCoGen sampler, including the D/F acceleration used in the accepted manuscript.

The denoiser, codebook storage, data loading, and normalization are shared with
Turbo. The DDIM update and denoiser/feedback schedules retain the reference runtime.
"""
import os
import time
import numpy as np
import scipy.io as sio
import torch
from utils import logger
from utils.metrics import evaluate_crnet_metrics_np, evaluate_nmse_crnet_np, init_step_buffers, aggregate_step_metrics
from modules.diffusion import set_seed, ensure_dir, load_ht_mat, load_raw_hf_all, amp_autocast
from modules.codebook import CodebookGaussianDiffusion as SharedCodebookDiffusion, select_codebook_index
from modules.codec import CodebookCodec as SharedCodec

def make_infer_timesteps(total_timesteps, infer_steps=0, spacing='uniform'):
    total_timesteps = int(total_timesteps)
    infer_steps = int(infer_steps or 0)
    if total_timesteps <= 0:
        raise ValueError(f'DIFF_TIMESTEPS must be positive, got {total_timesteps}')
    if infer_steps <= 0 or infer_steps >= total_timesteps:
        return list(range(total_timesteps - 1, -1, -1))
    if infer_steps < 2:
        raise ValueError('DIFF_INFER_STEPS must be at least 2 when using accelerated sampling')

    spacing = str(spacing or 'uniform').lower()
    if spacing == 'uniform':
        ts = np.rint(np.linspace(0, total_timesteps - 1, infer_steps)).astype(np.int64)
    elif spacing == 'quadratic':
        grid = np.linspace(0.0, np.sqrt(total_timesteps - 1), infer_steps)
        ts = np.rint(grid ** 2).astype(np.int64)
    else:
        raise ValueError(f'Unsupported DIFF_INFER_TIMESTEP_SPACING: {spacing} (use uniform or quadratic)')

    ts[0] = 0
    ts[-1] = total_timesteps - 1
    ts = np.unique(np.clip(ts, 0, total_timesteps - 1))
    if ts.size < 2:
        raise ValueError(f'Invalid inference timestep schedule: {ts.tolist()}')
    return ts[::-1].astype(np.int64).tolist()

def make_feedback_timesteps(total_timesteps, denoise_schedule, feedback_steps=0, spacing='uniform'):
    denoise_schedule = [int(t) for t in denoise_schedule]
    if not denoise_schedule:
        raise ValueError('denoise_schedule must not be empty')

    feedback_steps = int(feedback_steps or 0)
    if feedback_steps <= 0:
        return list(denoise_schedule)

    feedback_schedule = make_infer_timesteps(total_timesteps, feedback_steps, spacing)
    merged = sorted(set(feedback_schedule).union(denoise_schedule), reverse=True)
    return [int(t) for t in merged]

def make_infer_schedules(cfg):
    total_timesteps = int(cfg.DIFF_TIMESTEPS)
    denoise_spacing = str(getattr(cfg, 'DIFF_INFER_TIMESTEP_SPACING', 'uniform') or 'uniform').lower()
    feedback_spacing = str(getattr(cfg, 'DIFF_FEEDBACK_TIMESTEP_SPACING', '') or denoise_spacing).lower()
    denoise_schedule = make_infer_timesteps(
        total_timesteps,
        getattr(cfg, 'DIFF_INFER_STEPS', 0),
        denoise_spacing,
    )
    feedback_schedule = make_feedback_timesteps(
        total_timesteps,
        denoise_schedule,
        getattr(cfg, 'DIFF_INFER_FEEDBACK_STEPS', 0),
        feedback_spacing,
    )
    return denoise_schedule, feedback_schedule

def count_feedback_indices(feedback_schedule):
    return sum(1 for t in feedback_schedule if int(t) != 0)

def select_step_metric_timesteps(cfg, feedback_schedule):
    text = str(getattr(cfg, 'DIFF_STEP_METRIC_TIMESTEPS', '') or '').strip()
    return select_step_timesteps_from_text(text or 'all', feedback_schedule)

def select_step_timesteps_from_text(text, feedback_schedule):
    feedback_schedule = [int(t) for t in feedback_schedule]
    text = str(text or '').strip()
    if not text:
        return []

    tl = text.lower()
    if tl in ('all', 'full', 'feedback'):
        return list(feedback_schedule)
    if tl in ('none', 'off', 'false'):
        return []

    feedback_set = set(feedback_schedule)
    selected = []
    seen = set()
    for token in text.split(','):
        token = token.strip()
        if not token:
            continue
        token_l = token.lower()
        if token_l in ('first', 'start'):
            t = feedback_schedule[0]
        elif token_l in ('last', 'final'):
            t = feedback_schedule[-1]
        else:
            t = int(token)
        if t in feedback_set and t not in seen:
            selected.append(t)
            seen.add(t)
    return selected

def get_feedback_spacing(cfg):
    denoise_spacing = str(getattr(cfg, 'DIFF_INFER_TIMESTEP_SPACING', 'uniform') or 'uniform').lower()
    return str(getattr(cfg, 'DIFF_FEEDBACK_TIMESTEP_SPACING', '') or denoise_spacing).lower()

def iter_feedback_blocks(denoise_schedule, feedback_schedule):
    denoise_schedule = [int(t) for t in denoise_schedule]
    feedback_schedule = [int(t) for t in feedback_schedule]
    feedback_set = set(feedback_schedule)
    missing = [t for t in denoise_schedule if t not in feedback_set]
    if missing:
        raise ValueError(f'feedback schedule must contain every denoise timestep, missing={missing[:8]}')

    for anchor_i, anchor_t in enumerate(denoise_schedule):
        next_anchor = denoise_schedule[anchor_i + 1] if (anchor_i + 1) < len(denoise_schedule) else -1
        block = [t for t in feedback_schedule if anchor_t >= t > next_anchor]
        if not block or block[0] != anchor_t:
            raise ValueError(
                f'invalid feedback block for denoise timestep {anchor_t}: '
                f'next_anchor={next_anchor}, block={block[:8]}'
            )
        yield anchor_t, next_anchor, block

def get_infer_sampler(cfg):
    sampler = str(getattr(cfg, 'DIFF_INFER_SAMPLER', 'ddpm') or 'ddpm').lower()
    if sampler not in ('ddpm', 'ddim'):
        raise ValueError(f'Unsupported DIFF_INFER_SAMPLER: {sampler} (use ddpm or ddim)')
    eta = float(getattr(cfg, 'DIFF_DDIM_ETA', 0.0))
    if eta < 0:
        raise ValueError(f'DIFF_DDIM_ETA must be non-negative, got {eta}')
    return sampler, eta

def metadata_string(md, key, default=''):
    if key not in md:
        return default
    arr = np.asarray(md[key])
    if arr.size == 0:
        return default
    if arr.dtype.kind in ('U', 'S'):
        return ''.join(arr.astype(str).reshape(-1)).strip()
    item = arr.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode('utf-8').strip()
    return str(item).strip()

def metadata_scalar(md, key, default=None):
    if key not in md:
        return default
    arr = np.asarray(md[key])
    if arr.size == 0:
        return default
    return arr.reshape(-1)[0].item()

class CodebookGaussianDiffusion(SharedCodebookDiffusion):
    def predict_noise_from_start(self, xt, t, x0):
            denom = torch.clamp(self._extract(self.sqrt_1m_acp, t, xt.shape), min=1e-8)
            return (xt - self._extract(self.sqrt_acp, t, xt.shape) * x0) / denom

    def ddim_sample_with_index(self, pred, x, t, prev_t, best_idx, clip_denoised=True, pred_mode='eps', eta=0.0):
            x0 = pred if pred_mode == 'x0' else self.predict_start_from_noise(x, t, pred)
            if clip_denoised:
                x0 = torch.clamp(x0, self.clip_min, self.clip_max)
            if int(prev_t) < 0:
                return x0, x0

            t_idx = int(t[0].item())
            alpha_t = self._extract(self.acp, t, x.shape)
            alpha_prev = self.acp[int(prev_t)].reshape(1, 1, 1, 1)
            eps = self.predict_noise_from_start(x, t, x0)

            ratio = torch.clamp(1.0 - alpha_t / alpha_prev, min=0.0)
            sigma = float(eta) * torch.sqrt(torch.clamp((1.0 - alpha_prev) / (1.0 - alpha_t) * ratio, min=0.0))
            dir_coeff = torch.sqrt(torch.clamp(1.0 - alpha_prev - sigma ** 2, min=0.0))
            noise = self.codebook[t_idx][best_idx]
            x_prev = torch.sqrt(alpha_prev) * x0 + dir_coeff * eps + sigma * noise
            return x_prev, x0

    def sample_with_index(self, pred, x, t, prev_t, best_idx, sampler='ddpm', eta=0.0, clip_denoised=True, pred_mode='eps'):
            sampler = str(sampler).lower()
            t_idx = int(t[0].item())
            if sampler == 'ddpm' and int(prev_t) == t_idx - 1:
                return self.p_sample_with_index(pred, x, t, best_idx, clip_denoised, pred_mode)
            if sampler == 'ddpm':
                return self.ddim_sample_with_index(pred, x, t, prev_t, best_idx, clip_denoised, pred_mode, eta=1.0)
            if sampler == 'ddim':
                return self.ddim_sample_with_index(pred, x, t, prev_t, best_idx, clip_denoised, pred_mode, eta=eta)
            raise ValueError(f'Unsupported sampler: {sampler}')

class CsiCoGenCodec(SharedCodec):
    def encode_indices(self, checkpoint_path, output_indices, data_file, max_samples=None):
            cfg = self.cfg
            set_seed(cfg.SEED, deterministic=True)
            mean, std = self._load_norm_stats()
            x0_all = load_ht_mat(data_file, cfg.MAT_SIZE, cfg.IN_CHANNELS)
            if max_samples is not None and int(max_samples) > 0:
                x0_all = x0_all[:int(max_samples)]

            n = x0_all.shape[0]
            T = int(cfg.DIFF_TIMESTEPS)
            sampler, eta = get_infer_sampler(cfg)
            denoise_schedule, feedback_schedule = make_infer_schedules(cfg)
            full_schedule = list(range(T - 1, -1, -1))
            legacy_index_layout = sampler == 'ddpm' and denoise_schedule == full_schedule and feedback_schedule == full_schedule
            infer_batch = max(1, int(getattr(cfg, 'DIFF_INFER_BATCH', 256)))
            use_amp = bool(getattr(cfg, 'DIFF_INFER_USE_AMP', True) and self.device.type == 'cuda')
            use_ch_last = bool(getattr(cfg, 'DIFF_CHANNELS_LAST', True) and self.device.type == 'cuda')
            net = self._build_net(checkpoint_path)
            gdf = CodebookGaussianDiffusion(cfg, self.device)

            clip_flag = bool(cfg.CODEBOOK_CLIP)
            pred_mode = str(cfg.DIFF_PRED_MODE)
            idx_rows = T if legacy_index_layout else len(feedback_schedule)
            feedback_row = {int(t): i for i, t in enumerate(feedback_schedule)}
            idx_seq = torch.empty((idx_rows, n), dtype=torch.int64, device='cpu')

            logger.info(
                f'[encode] start N={n} T={T} denoise_steps={len(denoise_schedule)} '
                f'feedback_steps={len(feedback_schedule)} feedback_indices={count_feedback_indices(feedback_schedule)} '
                f'sampler={sampler} eta={eta:g} spacing={getattr(cfg, "DIFF_INFER_TIMESTEP_SPACING", "uniform")} '
                f'feedback_spacing={get_feedback_spacing(cfg)} noise_schedule={getattr(cfg, "DIFF_NOISE_SCHED_INFER", "linear")} '
                f'mode={pred_mode} batch={infer_batch} amp={use_amp}'
            )
            t0 = time.time()

            with torch.inference_mode():
                for st in range(0, n, infer_batch):
                    ed = min(st + infer_batch, n)
                    bsz = ed - st
                    x0 = (x0_all[st:ed] - mean) / (std + 1e-7)
                    x0 = torch.from_numpy(x0).float().to(self.device, non_blocking=True)
                    samples = gdf.init_noise.unsqueeze(0).repeat(bsz, 1, 1, 1).to(self.device)
                    if use_ch_last:
                        x0 = x0.contiguous(memory_format=torch.channels_last)
                        samples = samples.contiguous(memory_format=torch.channels_last)

                    for anchor_t, next_anchor, block in iter_feedback_blocks(denoise_schedule, feedback_schedule):
                        tt_anchor = torch.full((bsz,), anchor_t, device=self.device, dtype=torch.long)
                        with amp_autocast(enabled=use_amp):
                            out = net(samples, tt_anchor)
                        out = out.float()

                        x0_pred = out if pred_mode == 'x0' else gdf.predict_start_from_noise(samples, tt_anchor, out)
                        if clip_flag:
                            x0_pred = torch.clamp(x0_pred, gdf.clip_min, gdf.clip_max)

                        for block_i, t in enumerate(block):
                            prev_t = block[block_i + 1] if (block_i + 1) < len(block) else next_anchor
                            tt = torch.full((bsz,), t, device=self.device, dtype=torch.long)
                            best_idx = select_codebook_index(gdf, t, x0 - x0_pred)
                            idx_row = t if legacy_index_layout else feedback_row[t]
                            idx_seq[idx_row, st:ed] = best_idx.detach().cpu()
                            samples, _ = gdf.sample_with_index(
                                x0_pred, samples, tt, prev_t, best_idx, sampler, eta, clip_flag, 'x0'
                            )

            idx_np = idx_seq.numpy().astype(np.uint8 if int(cfg.CODEBOOK_SIZE) <= 256 else np.uint16)
            ensure_dir(os.path.dirname(output_indices))
            mat = {'best_idx_seq': idx_np}
            if not legacy_index_layout:
                mat.update({
                    'sample_timesteps': np.asarray(feedback_schedule, dtype=np.int32).reshape(1, -1),
                    'denoise_timesteps': np.asarray(denoise_schedule, dtype=np.int32).reshape(1, -1),
                    'diff_timesteps': np.asarray([[T]], dtype=np.int32),
                    'infer_sampler': sampler,
                    'ddim_eta': np.asarray([[eta]], dtype=np.float32),
                    'timestep_spacing': str(getattr(cfg, 'DIFF_INFER_TIMESTEP_SPACING', 'uniform')),
                    'feedback_timestep_spacing': get_feedback_spacing(cfg),
                    'noise_schedule_infer': str(getattr(cfg, 'DIFF_NOISE_SCHED_INFER', 'linear')),
                })
            sio.savemat(output_indices, mat)
            elapsed = time.time() - t0
            logger.info(f'[encode] saved {output_indices} shape={idx_np.shape} dtype={idx_np.dtype} time={elapsed:.2f}s')
            return elapsed

    def decode_from_indices(self, checkpoint_path, index_path, output_reconstruction, gt_data_file=None, gt_raw_file=None, return_metrics=False):
            cfg = self.cfg
            set_seed(cfg.SEED, deterministic=True)
            mean, std = self._load_norm_stats()

            md = sio.loadmat(index_path)
            idx = torch.from_numpy(md['best_idx_seq'].astype(np.int64))
            idx_rows, n = idx.shape
            if 'sample_timesteps' in md:
                feedback_schedule = [int(x) for x in np.asarray(md['sample_timesteps']).reshape(-1)]
                if len(feedback_schedule) != idx_rows:
                    raise ValueError(
                        f'Index/timestep mismatch in {index_path}: idx_rows={idx_rows}, sample_timesteps={len(feedback_schedule)}'
                    )
                if 'denoise_timesteps' in md:
                    denoise_schedule = [int(x) for x in np.asarray(md['denoise_timesteps']).reshape(-1)]
                else:
                    denoise_schedule = list(feedback_schedule)
                sampler = metadata_string(md, 'infer_sampler', default=str(getattr(cfg, 'DIFF_INFER_SAMPLER', 'ddpm'))).lower()
                eta = float(metadata_scalar(md, 'ddim_eta', default=float(getattr(cfg, 'DIFF_DDIM_ETA', 0.0))))
                row_for_feedback = {int(t): i for i, t in enumerate(feedback_schedule)}
                source_layout = 'step'
            else:
                feedback_schedule = list(range(idx_rows - 1, -1, -1))
                denoise_schedule = list(feedback_schedule)
                sampler, eta = get_infer_sampler(cfg)
                row_for_feedback = {int(t): int(t) for t in feedback_schedule}
                source_layout = 'legacy_timestep'

            if not feedback_schedule:
                raise ValueError(f'Empty sampling schedule in {index_path}')
            _ = list(iter_feedback_blocks(denoise_schedule, feedback_schedule))
            max_t = max(feedback_schedule)
            if max_t >= int(cfg.DIFF_TIMESTEPS):
                raise ValueError(
                    f'Index file uses timestep {max_t}, but cfg.DIFF_TIMESTEPS={int(cfg.DIFF_TIMESTEPS)}'
                )

            infer_batch = max(1, int(getattr(cfg, 'DIFF_INFER_BATCH', 256)))
            use_amp = bool(getattr(cfg, 'DIFF_INFER_USE_AMP', True) and self.device.type == 'cuda')
            use_ch_last = bool(getattr(cfg, 'DIFF_CHANNELS_LAST', True) and self.device.type == 'cuda')
            net = self._build_net(checkpoint_path)
            gdf = CodebookGaussianDiffusion(cfg, self.device)

            clip_flag = bool(cfg.CODEBOOK_CLIP)
            pred_mode = str(cfg.DIFF_PRED_MODE)

            gt_data_file = gt_data_file or cfg.GT_DATA_FILE
            gt_raw_file = gt_raw_file if gt_raw_file is not None else cfg.GT_RAW_FILE
            sparse_gt = load_ht_mat(gt_data_file, cfg.MAT_SIZE, cfg.IN_CHANNELS)
            raw_gt_cpu = load_raw_hf_all(gt_raw_file) if gt_raw_file and os.path.isfile(gt_raw_file) else None
            gt_n = int(sparse_gt.shape[0])
            eval_n = min(n, gt_n)
            if gt_n != n:
                logger.warning(f'[decode] GT/data length mismatch: idx_n={n}, gt_n={gt_n}; metrics will be computed on first {eval_n} samples only.')
            if raw_gt_cpu is not None and raw_gt_cpu.shape[0] != gt_n:
                raw_n = int(raw_gt_cpu.shape[0])
                aligned_n = min(eval_n, raw_n)
                logger.warning(f'[decode] RAW GT length mismatch: sparse_gt_n={gt_n}, raw_gt_n={raw_n}; rho will be computed on first {aligned_n} samples only.')
                eval_n = aligned_n

            logger.info(
                f'[decode] start N={n} T={int(cfg.DIFF_TIMESTEPS)} denoise_steps={len(denoise_schedule)} '
                f'feedback_steps={len(feedback_schedule)} feedback_indices={count_feedback_indices(feedback_schedule)} '
                f'sampler={sampler} eta={eta:g} layout={source_layout} mode={pred_mode} batch={infer_batch} amp={use_amp}'
            )
            t0 = time.time()

            rec_nchw = np.empty((n, cfg.IN_CHANNELS, cfg.MAT_SIZE, cfg.MAT_SIZE), dtype=np.float32)
            print_steps = bool(cfg.SAMPLE_PRINT_EVERY_STEP)
            metric_timesteps = select_step_metric_timesteps(cfg, feedback_schedule)
            metric_timestep_set = set(metric_timesteps)
            collect_step_metrics = bool((print_steps or return_metrics) and metric_timesteps)
            step_save_dir = str(getattr(cfg, 'DIFF_STEP_SAVE_DIR', '') or '').strip()
            step_save_text = str(getattr(cfg, 'DIFF_STEP_SAVE_TIMESTEPS', '') or '').strip()
            if step_save_dir and not step_save_text:
                step_save_text = str(getattr(cfg, 'DIFF_STEP_METRIC_TIMESTEPS', '') or '').strip()
            step_save_timesteps = select_step_timesteps_from_text(step_save_text, feedback_schedule) if step_save_dir else []
            step_save_set = set(step_save_timesteps)
            has_rho = raw_gt_cpu is not None
            collect_step_rho = bool(has_rho and getattr(cfg, 'DIFF_STEP_METRIC_RHO', True))
            if collect_step_metrics:
                metric_T = max(int(cfg.DIFF_TIMESTEPS), max_t + 1)
                step_nmse_sum, step_rho_sum, step_cnt = init_step_buffers(metric_T, collect_step_rho)
            if step_save_timesteps:
                ensure_dir(step_save_dir)
                step_rec_buffers = {
                    int(t): np.empty((n, cfg.IN_CHANNELS, cfg.MAT_SIZE, cfg.MAT_SIZE), dtype=np.float32)
                    for t in step_save_timesteps
                }
            else:
                step_rec_buffers = {}

            with torch.inference_mode():
                for st in range(0, n, infer_batch):
                    ed = min(st + infer_batch, n)
                    bsz = ed - st
                    samples = gdf.init_noise.unsqueeze(0).repeat(bsz, 1, 1, 1).to(self.device)
                    if use_ch_last:
                        samples = samples.contiguous(memory_format=torch.channels_last)
                    eval_ed = min(ed, eval_n)
                    eval_bsz = max(0, eval_ed - st)
                    sparse_gt_chunk = sparse_gt[st:eval_ed] if eval_bsz > 0 else None
                    raw_gt_chunk = raw_gt_cpu[st:eval_ed] if (has_rho and eval_bsz > 0) else None

                    for anchor_t, next_anchor, block in iter_feedback_blocks(denoise_schedule, feedback_schedule):
                        tt_anchor = torch.full((bsz,), anchor_t, device=self.device, dtype=torch.long)
                        with amp_autocast(enabled=use_amp):
                            out = net(samples, tt_anchor)
                        out = out.float()

                        pred_x0_anchor = out if pred_mode == 'x0' else gdf.predict_start_from_noise(samples, tt_anchor, out)
                        if clip_flag:
                            pred_x0_anchor = torch.clamp(pred_x0_anchor, gdf.clip_min, gdf.clip_max)

                        for block_i, t in enumerate(block):
                            prev_t = block[block_i + 1] if (block_i + 1) < len(block) else next_anchor
                            tt = torch.full((bsz,), t, device=self.device, dtype=torch.long)
                            best_idx = idx[row_for_feedback[t], st:ed].to(self.device)
                            samples, pred_x0 = gdf.sample_with_index(
                                pred_x0_anchor, samples, tt, prev_t, best_idx, sampler, eta, clip_flag, 'x0'
                            )

                            need_step_snapshot = (
                                (collect_step_metrics and eval_bsz > 0 and t in metric_timestep_set)
                                or t in step_save_set
                            )
                            if need_step_snapshot:
                                sparse_pred_step = (pred_x0 * (std + 1e-7) + mean).detach().cpu().numpy()

                            if t in step_save_set:
                                step_rec_buffers[int(t)][st:ed] = sparse_pred_step

                            if collect_step_metrics and eval_bsz > 0 and t in metric_timestep_set:
                                if collect_step_rho:
                                    nmse_step, rho_step = evaluate_crnet_metrics_np(
                                        sparse_pred_step[:eval_bsz],
                                        sparse_gt_chunk,
                                        raw_gt_chunk,
                                        self.device,
                                        batch_size=min(500, eval_bsz),
                                    )
                                else:
                                    nmse_step = evaluate_nmse_crnet_np(
                                        sparse_pred_step[:eval_bsz],
                                        sparse_gt_chunk,
                                        batch_size=min(500, eval_bsz),
                                    )
                                    rho_step = None
                                step_nmse_sum[t] += float(nmse_step) * eval_bsz
                                if collect_step_rho and rho_step is not None:
                                    step_rho_sum[t] += float(rho_step) * eval_bsz
                                step_cnt[t] += eval_bsz

                    rec_nchw[st:ed] = (samples * (std + 1e-7) + mean).detach().cpu().numpy()

            step_nmse = None
            step_rho = None
            if collect_step_metrics:
                step_nmse = np.zeros((len(metric_timesteps),), dtype=np.float64)
                step_rho = np.zeros((len(metric_timesteps),), dtype=np.float64) if collect_step_rho else None
                for step_i, t in enumerate(metric_timesteps):
                    nmse_t, rho_t = aggregate_step_metrics(step_nmse_sum, step_rho_sum, step_cnt, t, collect_step_rho)
                    step_nmse[step_i] = nmse_t
                    if collect_step_rho and rho_t is not None:
                        step_rho[step_i] = rho_t
                    if not print_steps:
                        continue
                    if rho_t is None:
                        logger.info(f'[decode] t={t} NMSE(dB)={nmse_t:.4f}')
                    else:
                        logger.info(f'[decode] t={t} NMSE(dB)={nmse_t:.4f} rho={rho_t:.4f}')

            rec = rec_nchw.transpose(0, 2, 3, 1)
            ensure_dir(os.path.dirname(output_reconstruction))
            sio.savemat(output_reconstruction, {'ldm_ddpm_rec': rec})
            step_rec_files = {}
            for t in step_save_timesteps:
                step_path = os.path.join(step_save_dir, f'step_rec_t{int(t)}.mat')
                step_rec = step_rec_buffers[int(t)].transpose(0, 2, 3, 1)
                sio.savemat(step_path, {
                    'ldm_ddpm_rec': step_rec,
                    'metric_t': np.asarray([[int(t)]], dtype=np.int32),
                })
                step_rec_files[int(t)] = step_path
                logger.info(f'[decode] saved step reconstruction t={int(t)} {step_path}')
            elapsed = time.time() - t0
            logger.info(f'[decode] saved {output_reconstruction} time={elapsed:.2f}s')

            metrics_t0 = time.time()
            if eval_n > 0:
                nmse, rho = evaluate_crnet_metrics_np(rec_nchw[:eval_n], sparse_gt[:eval_n], (raw_gt_cpu[:eval_n] if raw_gt_cpu is not None else None), self.device, batch_size=500)
            else:
                nmse, rho = float('nan'), None
            metrics_elapsed = time.time() - metrics_t0
            if rho is None:
                logger.info(f'[decode][CRNet-aligned] NMSE(dB)={nmse:.6f}')
            else:
                logger.info(f'[decode][CRNet-aligned] NMSE(dB)={nmse:.6f} rho={rho:.6f}')

            if return_metrics:
                return {
                    'nmse': float(nmse),
                    'rho': (None if rho is None else float(rho)),
                    'step_nmse': (None if step_nmse is None else step_nmse.tolist()),
                    'step_rho': (None if step_rho is None else step_rho.tolist()),
                    'step_timesteps': metric_timesteps if collect_step_metrics else [],
                    'denoise_timesteps': denoise_schedule,
                    'feedback_timesteps': feedback_schedule,
                    'sampler': sampler,
                    'ddim_eta': float(eta),
                    'elapsed_sec': float(elapsed),
                    'metrics_eval_sec': float(metrics_elapsed),
                    'final_metrics_excluded_from_elapsed': True,
                    'step_metrics_included_in_elapsed': bool(collect_step_metrics),
                    'step_rec_files': step_rec_files,
                }
