"""Closed-loop finite-rate encode/decode and Turbo macro schedules."""

import json
import os
import re
import time

import numpy as np
import scipy.io as sio
import torch

from utils import logger
from utils.metrics import evaluate_crnet_metrics_np
from modules.codebook import CodebookGaussianDiffusion, codec_contract_for_cfg, predict_h0, select_codebook_index
from modules.diffusion import (
    amp_autocast,
    build_unet_from_cfg,
    count_model_params,
    ensure_dir,
    inverse_normalize_csi_array,
    load_ht_mat,
    load_norm_stats_file,
    load_raw_hf_all,
    load_torch,
    array_scalar,
    model_metadata_from_cfg,
    normalize_csi_array,
    normalize_pred_mode,
    set_seed,
    write_json_manifest,
)

def build_feedback_timeline(timesteps, macro_stride=1, macro_tail=4, macro_multi=True):
    T = int(timesteps)
    if T <= 1:
        return []
    if int(macro_stride) <= 1:
        return [int(t) for t in range(T - 1, 0, -1)]
    _, transitions, _ = build_macro_sampling_plan(T, macro_stride, macro_tail)
    timeline = []
    for t, s_next in transitions:
        step_iter = range(int(t), int(s_next), -1) if bool(macro_multi) else [int(t)]
        for u in step_iter:
            if int(u) > 0:
                timeline.append(int(u))
    return timeline

def parse_prefix_lengths(value, max_slots):
    max_slots = max(0, int(max_slots))
    text = str(value).strip().lower()
    if not text or text == 'none':
        return []
    if text == 'all':
        return list(range(1, max_slots + 1))
    if text == 'full':
        return [max_slots] if max_slots > 0 else []

    out = []
    parts = re.split(r'[\s,;|:+]+', text)
    for part in parts:
        if not part:
            continue
        length = int(part)
        if length < 1 or length > max_slots:
            raise ValueError(f'Prefix length {length} outside valid range [1,{max_slots}]')
        if length not in out:
            out.append(length)
    return sorted(out)

def build_macro_sampling_plan(timesteps, stride, tail_steps):
    T = max(1, int(timesteps))
    s = max(1, int(stride))
    tail = max(1, int(tail_steps))
    tail = min(tail, T)
    tail_start = tail - 1

    denoise_steps = [T - 1]
    cur = T - 1
    while cur > tail_start:
        nxt = max(cur - s, tail_start)
        if nxt == cur:
            break
        denoise_steps.append(nxt)
        cur = nxt

    for t in range(tail_start - 1, -1, -1):
        if denoise_steps[-1] != t:
            denoise_steps.append(t)

    if denoise_steps[-1] != 0:
        denoise_steps.append(0)

    transitions = []
    for i in range(len(denoise_steps) - 1):
        t = int(denoise_steps[i])
        s_next = int(denoise_steps[i + 1])
        if s_next >= t:
            continue
        transitions.append((t, s_next))

    feedback_slots = int(sum(max(0, t - s_next) for (t, s_next) in transitions))
    return denoise_steps, transitions, feedback_slots

def normalize_macro_refresh_mode(mode):
    m = str(mode).strip().lower()
    if m in ('teacher',):
        return m
    return 'none'

def get_macro_refresh_mid_t(t, s_next, ratio):
    span = int(t) - int(s_next)
    if span <= 1:
        return None
    r = float(ratio)
    if not np.isfinite(r):
        r = 0.5
    r = min(0.9, max(0.1, r))
    first_span = int(round(span * r))
    first_span = max(1, min(span - 1, first_span))
    return int(t - first_span)

def get_macro_refresh_points(t, s_next, ratio, count):
    cnt = max(1, int(count))
    if cnt == 1:
        mid = get_macro_refresh_mid_t(t, s_next, ratio)
        return [] if mid is None else [mid]

    span = int(t) - int(s_next)
    if span <= 1:
        return []

    points = []
    for k in range(1, cnt + 1):
        offset = int(round(span * k / float(cnt + 1)))
        offset = max(1, min(span - 1, offset))
        p = int(t - offset)
        if int(s_next) < p < int(t) and p not in points:
            points.append(p)
    return sorted(points, reverse=True)

def parse_macro_refresh_t_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        parts = value
    else:
        text = str(value).strip()
        if not text:
            return []
        parts = re.split(r'[\s,;|:+-]+', text)

    points = []
    for part in parts:
        if part is None:
            continue
        token = str(part).strip()
        if not token:
            continue
        point = int(token)
        if point not in points:
            points.append(point)
    return sorted(points, reverse=True)

def get_macro_refresh_points_for_transition(t, s_next, ratio, count, refresh_t_list=None):
    explicit_points = parse_macro_refresh_t_list(refresh_t_list)
    if explicit_points:
        points = []
        for point in explicit_points:
            point = int(point)
            if int(s_next) < point < int(t) and point not in points:
                points.append(point)
        return sorted(points, reverse=True)
    return get_macro_refresh_points(t, s_next, ratio, count)

def should_use_macro_refresh(t, s_next, mode, macro_multi, min_span):
    if str(mode) == 'none':
        return False
    if not bool(macro_multi):
        return False
    span = int(t) - int(s_next)
    return span >= max(2, int(min_span))

def count_macro_refresh_calls_with_explicit_points(
    transitions, mode, macro_multi, min_span, ratio, refresh_count, refresh_t_list
):
    cnt = 0
    for t, s_next in transitions:
        if should_use_macro_refresh(t, s_next, mode, macro_multi, min_span):
            cnt += len(get_macro_refresh_points_for_transition(t, s_next, ratio, refresh_count, refresh_t_list))
    return cnt

class CodebookCodec:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device

    def _load_norm_stats(self):
        stats_path_cfg = str(getattr(self.cfg, 'DIFF_NORM_STATS_PATH', '')).strip()
        stats_path = stats_path_cfg if stats_path_cfg else os.path.join(self.cfg.DIFF_RESULTS_DIR, 'normalization.npz')
        return load_norm_stats_file(stats_path)

    def _build_net(self, checkpoint_path):
        if self.device.type == 'cuda':
            if bool(getattr(self.cfg, 'DIFF_FAST_BENCHMARK', True)):
                torch.backends.cudnn.benchmark = True
            if bool(getattr(self.cfg, 'DIFF_ALLOW_TF32', True)):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        net = build_unet_from_cfg(self.cfg, self.device, prefix='DIFF_MODEL')
        net.load_state_dict(load_torch(checkpoint_path, map_location=self.device), strict=True)
        net.eval()
        return net


    def _refresh_segment_prediction(
        self, refresh_net, gdf, x_cur, t_idx, use_amp, pred_mode, clip_flag
    ):
        out_mid, tt_mid = self._run_denoiser(refresh_net, x_cur, t_idx, use_amp)
        return predict_h0(gdf, x_cur, tt_mid, out_mid, pred_mode, clip_flag)

    def _run_denoiser(self, net, samples, t_idx, use_amp):
        tt = torch.full((samples.shape[0],), int(t_idx), device=self.device, dtype=torch.long)
        micro = int(getattr(self.cfg, 'DIFF_DENOISER_MICRO_BATCH', 0))
        if micro <= 0 or micro >= samples.shape[0]:
            with amp_autocast(enabled=use_amp):
                out = net(samples, tt)
            return out.float(), tt

        chunks = []
        for st in range(0, samples.shape[0], micro):
            ed = min(st + micro, samples.shape[0])
            with amp_autocast(enabled=use_amp):
                chunk_out = net(samples[st:ed], tt[st:ed])
            chunks.append(chunk_out.float())
        out = torch.cat(chunks, dim=0)
        return out, tt

    def encode_indices(self, checkpoint_path, output_indices, data_file):
        cfg = self.cfg
        contract = codec_contract_for_cfg(cfg)
        set_seed(cfg.SEED, deterministic=True)
        mean, std = self._load_norm_stats()
        h0_all = load_ht_mat(data_file, cfg.MAT_SIZE, cfg.IN_CHANNELS)
        max_samples = int(getattr(cfg, 'DIFF_MAX_SAMPLES', 0))
        if max_samples > 0:
            h0_all = h0_all[:max_samples]

        n = h0_all.shape[0]
        T = int(cfg.DIFF_TIMESTEPS)
        infer_batch = max(1, int(getattr(cfg, 'DIFF_INFER_BATCH', 256)))
        denoiser_micro = max(0, int(getattr(cfg, 'DIFF_DENOISER_MICRO_BATCH', 0)))
        use_amp = bool(getattr(cfg, 'DIFF_INFER_USE_AMP', True) and self.device.type == 'cuda')
        use_ch_last = bool(getattr(cfg, 'DIFF_CHANNELS_LAST', True) and self.device.type == 'cuda')
        net = self._build_net(checkpoint_path)
        model_meta = model_metadata_from_cfg(cfg, prefix='DIFF_MODEL', params=count_model_params(net))
        gdf = CodebookGaussianDiffusion(cfg, self.device)

        macro_stride = max(1, int(getattr(cfg, 'DIFF_MACRO_STRIDE', 1)))
        macro_tail = max(1, int(getattr(cfg, 'DIFF_MACRO_TAIL_STEPS', 4)))
        macro_eta = float(getattr(cfg, 'DIFF_MACRO_ETA', 1.0))
        macro_multi = bool(getattr(cfg, 'DIFF_MACRO_MULTI_INDEX', True))
        use_macro = macro_stride > 1
        denoise_steps, transitions, feedback_slots = build_macro_sampling_plan(T, macro_stride, macro_tail)
        feedback_timeline = build_feedback_timeline(T, macro_stride, macro_tail, macro_multi)
        refresh_mode = normalize_macro_refresh_mode(getattr(cfg, 'DIFF_MACRO_REFRESH_MODE', 'none'))
        refresh_ratio = float(getattr(cfg, 'DIFF_MACRO_REFRESH_RATIO', 0.5))
        refresh_min_span = max(2, int(getattr(cfg, 'DIFF_MACRO_REFRESH_MIN_SPAN', 6)))
        refresh_count = max(1, int(getattr(cfg, 'DIFF_MACRO_REFRESH_COUNT', 1)))
        refresh_t_list = str(getattr(cfg, 'DIFF_MACRO_REFRESH_T_LIST', '')).strip()
        if not use_macro:
            refresh_mode = 'none'
        refresh_net = None
        if refresh_mode == 'teacher':
            refresh_net = net
        refresh_calls = count_macro_refresh_calls_with_explicit_points(
            transitions, refresh_mode, macro_multi, refresh_min_span, refresh_ratio, refresh_count, refresh_t_list
        )
        denoiser_calls_total = len(denoise_steps) + refresh_calls

        clip_flag = bool(cfg.CODEBOOK_CLIP)
        pred_mode = normalize_pred_mode(cfg.DIFF_PRED_MODE)
        idx_seq = torch.zeros((T, n), dtype=torch.int64, device='cpu')

        bits_per_index = float(np.log2(max(2, int(cfg.CODEBOOK_SIZE))))
        logger.info(
            f'[encode] start N={n} T={T} gen_mode={contract["gen_mode"]} codec={contract["codec"]} '
            f'codec_update={contract["codec_update"]} '
            f'codec_selector={contract.get("codec_selector", "legacy_dot")} '
            f'mode={pred_mode} batch={infer_batch} amp={use_amp} '
            f'denoiser_micro_batch={(denoiser_micro if denoiser_micro > 0 else infer_batch)} '
            f'macro={use_macro} stride={macro_stride} tail={macro_tail} eta={macro_eta:.3f} '
            f'multi_index={macro_multi} refresh={refresh_mode} refresh_ratio={refresh_ratio:.2f} '
            f'refresh_min_span={refresh_min_span} refresh_count={refresh_count} '
            f'refresh_t_list={(refresh_t_list if refresh_t_list else "<auto>")} '
            f'denoiser_calls={denoiser_calls_total} '
            f'base_calls={len(denoise_steps)} refresh_calls={refresh_calls} feedback_slots={feedback_slots} '
            f'feedback_bits/sample={feedback_slots * bits_per_index:.2f}'
        )

        t0 = time.time()

        with torch.inference_mode():
            for st in range(0, n, infer_batch):
                ed = min(st + infer_batch, n)
                bsz = ed - st
                h0 = normalize_csi_array(h0_all[st:ed], mean, std)
                h0 = torch.from_numpy(h0).float().to(self.device, non_blocking=True)
                samples = gdf.init_noise.unsqueeze(0).repeat(bsz, 1, 1, 1).to(self.device)
                if use_ch_last:
                    h0 = h0.contiguous(memory_format=torch.channels_last)
                    samples = samples.contiguous(memory_format=torch.channels_last)

                if not use_macro:
                    for t in reversed(range(T)):
                        out, tt = self._run_denoiser(net, samples, t, use_amp)
                        h0_pred = predict_h0(gdf, samples, tt, out, pred_mode, clip_flag)
                        best_idx = select_codebook_index(gdf, t, h0 - h0_pred)
                        idx_seq[t, st:ed] = best_idx.detach().cpu()
                        samples, _ = gdf.p_sample_with_index(out, samples, tt, best_idx, clip_flag, pred_mode)
                    continue

                for (t, s_next) in transitions:
                    out, tt = self._run_denoiser(net, samples, t, use_amp)
                    h0_pred = predict_h0(gdf, samples, tt, out, pred_mode, clip_flag)
                    use_refresh = should_use_macro_refresh(t, s_next, refresh_mode, macro_multi, refresh_min_span)
                    if not use_refresh:
                        resid = h0 - h0_pred
                        x_cur = samples
                        step_iter = range(t, s_next, -1) if macro_multi else [t]
                        for u in step_iter:
                            best_idx_u = select_codebook_index(gdf, u, resid)
                            idx_seq[u, st:ed] = best_idx_u.detach().cpu()
                            x_cur = gdf.p_sample_with_fixed_h0_and_index(h0_pred, x_cur, u, best_idx_u, eta=macro_eta)
                        samples = x_cur
                        continue

                    refresh_points = get_macro_refresh_points_for_transition(
                        t, s_next, refresh_ratio, refresh_count, refresh_t_list
                    )
                    if not refresh_points:
                        resid = h0 - h0_pred
                        x_cur = samples
                        for u in range(t, s_next, -1):
                            best_idx_u = select_codebook_index(gdf, u, resid)
                            idx_seq[u, st:ed] = best_idx_u.detach().cpu()
                            x_cur = gdf.p_sample_with_fixed_h0_and_index(h0_pred, x_cur, u, best_idx_u, eta=macro_eta)
                        samples = x_cur
                        continue

                    x_cur = samples
                    seg_start = t
                    seg_h0 = h0_pred
                    for boundary in refresh_points + [s_next]:
                        resid = h0 - seg_h0
                        for u in range(seg_start, boundary, -1):
                            best_idx_u = select_codebook_index(gdf, u, resid)
                            idx_seq[u, st:ed] = best_idx_u.detach().cpu()
                            x_cur = gdf.p_sample_with_fixed_h0_and_index(seg_h0, x_cur, u, best_idx_u, eta=macro_eta)

                        if boundary == s_next:
                            break
                        seg_h0 = self._refresh_segment_prediction(
                            refresh_net, gdf, x_cur, boundary, use_amp, pred_mode, clip_flag
                        )
                        seg_start = boundary
                    samples = x_cur

                out0, tt0 = self._run_denoiser(net, samples, 0, use_amp)
                h0_last = predict_h0(gdf, samples, tt0, out0, pred_mode, clip_flag)
                samples = h0_last

        elapsed = time.time() - t0
        feedback_bits = float(feedback_slots * bits_per_index)
        idx_np = idx_seq.numpy().astype(np.uint8 if int(cfg.CODEBOOK_SIZE) <= 256 else np.uint16)
        encode_meta = {
            'timesteps': int(T),
            'num_samples': int(n),
            'encode_time': float(elapsed),
            'denoiser_calls': int(denoiser_calls_total),
            'base_calls': int(len(denoise_steps)),
            'refresh_calls': int(refresh_calls),
            'feedback_slots': int(feedback_slots),
            'feedback_bits': float(feedback_bits),
            'bits_per_index': float(bits_per_index),
            **model_meta,
            'encode_decode_mode': 'ddpm',
            'macro': bool(use_macro),
            'macro_stride': int(macro_stride),
            'macro_tail': int(macro_tail),
            'macro_multi_index': bool(macro_multi),
            'refresh_mode': refresh_mode,
            'refresh_t_list': refresh_t_list,
            'feedback_timeline': feedback_timeline,
            'codebook_status': gdf.codebook_status,
            'codebook_path': gdf.codebook_path,
            'codebook_seed': int(gdf.codebook_seed),
            'codebook_norm': 'raw_gaussian',
            'codebook_unit_norm': False,
            'codebook_fingerprint': gdf.codebook_fingerprint,
            'codebook_vector_norm_stats': gdf.codebook_vector_norm_stats,
            'norm_stats_path': str(getattr(cfg, 'DIFF_NORM_STATS_PATH', '')).strip()
            or os.path.join(cfg.DIFF_RESULTS_DIR, 'normalization.npz'),
        }
        ensure_dir(os.path.dirname(output_indices))
        array_payload = {
            'best_idx_seq': idx_np,
            'feedback_timeline': np.asarray(feedback_timeline, dtype=np.int16),
            'codebook_fingerprint': gdf.codebook_fingerprint,
            'codebook_norm': 'raw_gaussian',
            'codebook_seed': int(gdf.codebook_seed),
            'encode_time': np.asarray([[elapsed]], dtype=np.float64),
            'feedback_bits': np.asarray([[feedback_bits]], dtype=np.float64),
            'Nslot': np.asarray([[feedback_slots]], dtype=np.int32),
            'Ncall': np.asarray([[denoiser_calls_total]], dtype=np.int32),
            'encode_metadata_json': json.dumps(encode_meta, sort_keys=True),
        }
        sio.savemat(
            output_indices,
            array_payload,
        )
        write_json_manifest(os.path.splitext(output_indices)[0] + '_encode.json', encode_meta)
        logger.info(f'[encode] saved {output_indices} shape={idx_np.shape} dtype={idx_np.dtype} time={elapsed:.2f}s')

    def _decode_indices_core(self, checkpoint_path, idx, mean, std, prefix_lengths=None, metric_observer=None):
        cfg = self.cfg
        contract = codec_contract_for_cfg(cfg)
        T, n = idx.shape
        infer_batch = max(1, int(getattr(cfg, 'DIFF_INFER_BATCH', 256)))
        denoiser_micro = max(0, int(getattr(cfg, 'DIFF_DENOISER_MICRO_BATCH', 0)))
        use_amp = bool(getattr(cfg, 'DIFF_INFER_USE_AMP', True) and self.device.type == 'cuda')
        use_ch_last = bool(getattr(cfg, 'DIFF_CHANNELS_LAST', True) and self.device.type == 'cuda')
        net = self._build_net(checkpoint_path)
        model_meta = model_metadata_from_cfg(cfg, prefix='DIFF_MODEL', params=count_model_params(net))
        gdf = CodebookGaussianDiffusion(cfg, self.device)

        clip_flag = bool(cfg.CODEBOOK_CLIP)
        pred_mode = normalize_pred_mode(cfg.DIFF_PRED_MODE)
        macro_stride = max(1, int(getattr(cfg, 'DIFF_MACRO_STRIDE', 1)))
        macro_tail = max(1, int(getattr(cfg, 'DIFF_MACRO_TAIL_STEPS', 4)))
        macro_eta = float(getattr(cfg, 'DIFF_MACRO_ETA', 1.0))
        macro_multi = bool(getattr(cfg, 'DIFF_MACRO_MULTI_INDEX', True))
        use_macro = macro_stride > 1
        denoise_steps, transitions, feedback_slots = build_macro_sampling_plan(T, macro_stride, macro_tail)
        refresh_mode = normalize_macro_refresh_mode(getattr(cfg, 'DIFF_MACRO_REFRESH_MODE', 'none'))
        refresh_ratio = float(getattr(cfg, 'DIFF_MACRO_REFRESH_RATIO', 0.5))
        refresh_min_span = max(2, int(getattr(cfg, 'DIFF_MACRO_REFRESH_MIN_SPAN', 6)))
        refresh_count = max(1, int(getattr(cfg, 'DIFF_MACRO_REFRESH_COUNT', 1)))
        refresh_t_list = str(getattr(cfg, 'DIFF_MACRO_REFRESH_T_LIST', '')).strip()
        if not use_macro:
            refresh_mode = 'none'
        refresh_net = None
        if refresh_mode == 'teacher':
            refresh_net = net
        refresh_calls = count_macro_refresh_calls_with_explicit_points(
            transitions, refresh_mode, macro_multi, refresh_min_span, refresh_ratio, refresh_count, refresh_t_list
        )
        denoiser_calls_total = len(denoise_steps) + refresh_calls
        if not macro_multi:
            feedback_slots = len(transitions)
        feedback_timeline = build_feedback_timeline(T, macro_stride, macro_tail, macro_multi)
        max_prefix = len(feedback_timeline)
        prefix_set = set(int(x) for x in (prefix_lengths or []) if 0 < int(x) <= max_prefix)

        if int(getattr(cfg, 'DIFF_TIMESTEPS', T)) != int(T):
            logger.warning(f'[decode] idx timestep mismatch: idx_T={T}, cfg.DIFF_TIMESTEPS={cfg.DIFF_TIMESTEPS}')

        bits_per_index = float(np.log2(max(2, int(cfg.CODEBOOK_SIZE))))
        logger.info(
            f'[decode] start N={n} T={T} gen_mode={contract["gen_mode"]} codec={contract["codec"]} '
            f'codec_update={contract["codec_update"]} '
            f'codec_selector={contract.get("codec_selector", "legacy_dot")} '
            f'mode={pred_mode} batch={infer_batch} amp={use_amp} '
            f'denoiser_micro_batch={(denoiser_micro if denoiser_micro > 0 else infer_batch)} '
            f'macro={use_macro} stride={macro_stride} tail={macro_tail} eta={macro_eta:.3f} '
            f'multi_index={macro_multi} refresh={refresh_mode} refresh_ratio={refresh_ratio:.2f} '
            f'refresh_min_span={refresh_min_span} refresh_count={refresh_count} '
            f'refresh_t_list={(refresh_t_list if refresh_t_list else "<auto>")} '
            f'denoiser_calls={denoiser_calls_total} '
            f'base_calls={len(denoise_steps)} refresh_calls={refresh_calls} feedback_slots={feedback_slots} '
            f'feedback_bits/sample={feedback_slots * bits_per_index:.2f}'
        )
        t0 = time.time()

        rec_nchw = np.empty((n, cfg.IN_CHANNELS, cfg.MAT_SIZE, cfg.MAT_SIZE), dtype=np.float32)

        with torch.inference_mode():
            for st in range(0, n, infer_batch):
                ed = min(st + infer_batch, n)
                bsz = ed - st
                samples = gdf.init_noise.unsqueeze(0).repeat(bsz, 1, 1, 1).to(self.device)
                if use_ch_last:
                    samples = samples.contiguous(memory_format=torch.channels_last)
                prefix_count = 0

                def observe_prefix(u, h0_est):
                    nonlocal prefix_count
                    if int(u) <= 0:
                        return
                    prefix_count += 1
                    if prefix_count in prefix_set and prefix_count != max_prefix and metric_observer is not None:
                        metric_observer(prefix_count, st, ed, h0_est)

                if not use_macro:
                    for t in reversed(range(T)):
                        out, tt = self._run_denoiser(net, samples, t, use_amp)
                        best_idx = idx[t, st:ed].to(self.device)
                        samples, pred_h0 = gdf.p_sample_with_index(out, samples, tt, best_idx, clip_flag, pred_mode)
                        observe_prefix(t, pred_h0)
                else:
                    for (t, s_next) in transitions:
                        out, tt = self._run_denoiser(net, samples, t, use_amp)
                        h0_pred = predict_h0(gdf, samples, tt, out, pred_mode, clip_flag)
                        use_refresh = should_use_macro_refresh(t, s_next, refresh_mode, macro_multi, refresh_min_span)
                        if not use_refresh:
                            x_cur = samples
                            step_iter = range(t, s_next, -1) if macro_multi else [t]
                            for u in step_iter:
                                best_idx_u = idx[u, st:ed].to(self.device)
                                x_cur = gdf.p_sample_with_fixed_h0_and_index(h0_pred, x_cur, u, best_idx_u, eta=macro_eta)
                                observe_prefix(u, h0_pred)
                            samples = x_cur
                        else:
                            refresh_points = get_macro_refresh_points_for_transition(
                                t, s_next, refresh_ratio, refresh_count, refresh_t_list
                            )
                            if not refresh_points:
                                x_cur = samples
                                for u in range(t, s_next, -1):
                                    best_idx_u = idx[u, st:ed].to(self.device)
                                    x_cur = gdf.p_sample_with_fixed_h0_and_index(h0_pred, x_cur, u, best_idx_u, eta=macro_eta)
                                    observe_prefix(u, h0_pred)
                                samples = x_cur
                            else:
                                x_cur = samples
                                seg_start = t
                                seg_h0 = h0_pred
                                for boundary in refresh_points + [s_next]:
                                    for u in range(seg_start, boundary, -1):
                                        best_idx_u = idx[u, st:ed].to(self.device)
                                        x_cur = gdf.p_sample_with_fixed_h0_and_index(seg_h0, x_cur, u, best_idx_u, eta=macro_eta)
                                        observe_prefix(u, seg_h0)

                                    if boundary == s_next:
                                        break
                                    seg_h0 = self._refresh_segment_prediction(
                                        refresh_net, gdf, x_cur, boundary, use_amp, pred_mode, clip_flag
                                    )
                                    seg_start = boundary
                                samples = x_cur

                    out0, tt0 = self._run_denoiser(net, samples, 0, use_amp)
                    h0_last = predict_h0(gdf, samples, tt0, out0, pred_mode, clip_flag)
                    samples = h0_last

                if max_prefix in prefix_set and metric_observer is not None:
                    metric_observer(max_prefix, st, ed, samples)

                rec_nchw[st:ed] = inverse_normalize_csi_array(samples.detach().cpu().numpy(), mean, std)

        elapsed = time.time() - t0
        meta = {
            'timesteps': int(T),
            'num_samples': int(n),
            'denoiser_calls': int(denoiser_calls_total),
            'base_calls': int(len(denoise_steps)),
            'refresh_calls': int(refresh_calls),
            'feedback_slots': int(feedback_slots),
            'feedback_bits': float(feedback_slots * bits_per_index),
            'bits_per_index': float(bits_per_index),
            **model_meta,
            'encode_decode_mode': 'ddpm',
            'macro': bool(use_macro),
            'macro_stride': int(macro_stride),
            'macro_tail': int(macro_tail),
            'macro_multi_index': bool(macro_multi),
            'refresh_mode': refresh_mode,
            'refresh_t_list': refresh_t_list,
            'feedback_timeline': feedback_timeline,
            'codebook_status': gdf.codebook_status,
            'codebook_path': gdf.codebook_path,
            'codebook_seed': int(gdf.codebook_seed),
            'codebook_norm': 'raw_gaussian',
            'codebook_unit_norm': False,
            'codebook_fingerprint': gdf.codebook_fingerprint,
            'codebook_vector_norm_stats': gdf.codebook_vector_norm_stats,
            'decode_time': float(elapsed),
        }
        return rec_nchw, meta

    def decode_from_indices(self, checkpoint_path, index_path, output_reconstruction, gt_data_file=None, gt_raw_file=None, return_metrics=False):
        cfg = self.cfg
        codec_contract_for_cfg(cfg)
        set_seed(cfg.SEED, deterministic=True)
        mean, std = self._load_norm_stats()

        md = sio.loadmat(index_path)
        idx = torch.from_numpy(md['best_idx_seq'].astype(np.int64))
        encode_time = array_scalar(md, 'encode_time', None)
        T, n = idx.shape

        macro_stride = max(1, int(getattr(cfg, 'DIFF_MACRO_STRIDE', 1)))
        macro_tail = max(1, int(getattr(cfg, 'DIFF_MACRO_TAIL_STEPS', 4)))
        macro_multi = bool(getattr(cfg, 'DIFF_MACRO_MULTI_INDEX', True))
        max_prefix = len(build_feedback_timeline(T, macro_stride, macro_tail, macro_multi))
        prefix_lengths = parse_prefix_lengths(getattr(cfg, 'DIFF_PREFIX_LENGTHS', 'all'), max_prefix)

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

        has_rho = raw_gt_cpu is not None
        prefix_nmse_sum = {int(m): 0.0 for m in prefix_lengths}
        prefix_rho_sum = {int(m): 0.0 for m in prefix_lengths}
        prefix_cnt = {int(m): 0 for m in prefix_lengths}

        def metric_observer(prefix_len, st, ed, h0_norm):
            eval_ed = min(ed, eval_n)
            eval_bsz = max(0, eval_ed - st)
            if eval_bsz <= 0 or prefix_len not in prefix_nmse_sum:
                return
            sparse_pred = inverse_normalize_csi_array(h0_norm[:eval_bsz].detach().cpu().numpy(), mean, std)
            sparse_gt_chunk = sparse_gt[st:eval_ed]
            raw_gt_chunk = raw_gt_cpu[st:eval_ed] if has_rho else None
            nmse_b, rho_b = evaluate_crnet_metrics_np(
                sparse_pred, sparse_gt_chunk, raw_gt_chunk, self.device, batch_size=min(500, eval_bsz)
            )
            prefix_nmse_sum[prefix_len] += float(nmse_b) * eval_bsz
            if has_rho and rho_b is not None:
                prefix_rho_sum[prefix_len] += float(rho_b) * eval_bsz
            prefix_cnt[prefix_len] += eval_bsz

        rec_nchw, meta = self._decode_indices_core(
            checkpoint_path,
            idx,
            mean,
            std,
            prefix_lengths=prefix_lengths,
            metric_observer=metric_observer if prefix_lengths else None,
        )

        rec = rec_nchw.transpose(0, 2, 3, 1)
        ensure_dir(os.path.dirname(output_reconstruction))
        rec_key = 'ldm_ddpm_rec'
        sio.savemat(output_reconstruction, {rec_key: rec})
        logger.info(f'[decode] saved {output_reconstruction} time={meta["decode_time"]:.2f}s')

        if eval_n > 0:
            nmse, rho = evaluate_crnet_metrics_np(rec_nchw[:eval_n], sparse_gt[:eval_n], (raw_gt_cpu[:eval_n] if raw_gt_cpu is not None else None), self.device, batch_size=500)
        else:
            nmse, rho = float('nan'), None
        if rho is None:
            logger.info(f'[decode][CRNet-aligned] NMSE(dB)={nmse:.6f}')
        else:
            logger.info(f'[decode][CRNet-aligned] NMSE(dB)={nmse:.6f} rho={rho:.6f}')

        prefix_nmse = []
        prefix_rho = [] if has_rho else None
        for m in prefix_lengths:
            cnt = max(1, int(prefix_cnt.get(m, 0)))
            nmse_m = float(prefix_nmse_sum.get(m, float('nan')) / cnt) if prefix_cnt.get(m, 0) > 0 else float('nan')
            rho_m = (
                float(prefix_rho_sum.get(m, float('nan')) / cnt)
                if has_rho and prefix_cnt.get(m, 0) > 0
                else None
            )
            prefix_nmse.append(nmse_m)
            if has_rho:
                prefix_rho.append(float('nan') if rho_m is None else rho_m)
            if bool(cfg.SAMPLE_PRINT_EVERY_STEP):
                if rho_m is None:
                    logger.info(f'[decode][prefix] m={m} NMSE(dB)={nmse_m:.4f}')
                else:
                    logger.info(f'[decode][prefix] m={m} NMSE(dB)={nmse_m:.4f} rho={rho_m:.4f}')

        metrics_payload = {
            'nmse': float(nmse),
            'rho': (None if rho is None else float(rho)),
            'feedback_bits': float(meta['feedback_bits']),
            'Nslot': int(meta['feedback_slots']),
            'Ncall': int(meta['denoiser_calls']),
            'encode_time': encode_time,
            'decode_time': float(meta['decode_time']),
            'prefix_lengths': [int(x) for x in prefix_lengths],
            'per_prefix_nmse': prefix_nmse,
            'per_prefix_rho': prefix_rho,
            'step_nmse': prefix_nmse,
            'step_rho': prefix_rho,
            'rec_key': rec_key,
            'codebook_fingerprint': meta['codebook_fingerprint'],
            'codebook_norm': 'raw_gaussian',
            'norm_stats_path': str(getattr(cfg, 'DIFF_NORM_STATS_PATH', '')).strip()
            or os.path.join(cfg.DIFF_RESULTS_DIR, 'normalization.npz'),
        }
        metrics_payload.update(meta)
        write_json_manifest(os.path.splitext(output_reconstruction)[0] + '_metrics.json', metrics_payload)

        if return_metrics:
            return metrics_payload
