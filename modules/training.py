"""Training CsiCoGen and CsiCoGen-Lite denoisers."""

import heapq
import json
import os
import shutil

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from utils import logger
from utils.metrics import evaluate_crnet_metrics_np, init_step_buffers, aggregate_step_metrics
from utils.statics import AverageMeter
from modules.codebook import CodebookGaussianDiffusion, predict_h0, select_codebook_index
from modules.diffusion import (
    EMA,
    GaussianDiffusion,
    amp_autocast,
    build_unet_from_cfg,
    count_model_params,
    ensure_dir,
    load_ht_mat,
    load_raw_hf_all,
    load_torch,
    make_grad_scaler,
    model_metadata_from_cfg,
    normalize_csi_array,
    normalize_gen_mode,
    normalize_pred_mode,
    parse_diff_epoch_from_path,
    save_norm_stats,
    should_save_periodic_checkpoint,
    write_json_manifest,
    checkpoint_names,
)


class DiffusionTrainer:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device

    def _new_unet(self):
        return build_unet_from_cfg(self.cfg, self.device, prefix='DIFF_MODEL')


    def _make_eval_net(self, train_net, ema):
        eval_net = self._new_unet()
        eval_net.load_state_dict(train_net.state_dict())
        if bool(getattr(self.cfg, 'DIFF_EVAL_USE_EMA', True)) and (ema is not None):
            ema.copy_to(eval_net)
        eval_net.eval()
        return eval_net

    @torch.inference_mode()
    def _evaluate_during_train(self, net, mean, std, sparse_gt_np, raw_gt_cpu, max_samples):
        cfg = self.cfg
        n = min(int(max_samples), sparse_gt_np.shape[0])
        if n <= 0:
            return None, None

        sparse_gt_eval = sparse_gt_np[:n]
        raw_gt_eval = raw_gt_cpu[:n] if raw_gt_cpu is not None else None

        gdf = CodebookGaussianDiffusion(cfg, self.device)
        clip_flag = bool(cfg.CODEBOOK_CLIP)
        pred_mode = normalize_pred_mode(cfg.DIFF_PRED_MODE)
        T = int(cfg.DIFF_TIMESTEPS)
        print_steps = bool(getattr(cfg, 'DIFF_EVAL_PRINT_STEPS', False))
        eval_batch = max(1, int(getattr(cfg, 'DIFF_EVAL_BATCH', 500)))

        nmse_sum = 0.0
        rho_sum = 0.0
        cnt = 0
        has_rho = raw_gt_eval is not None
        if print_steps:
            step_nmse_sum, step_rho_sum, step_cnt = init_step_buffers(T, has_rho)

        use_amp = bool(getattr(cfg, 'DIFF_INFER_USE_AMP', True) and self.device.type == 'cuda')

        for st in range(0, n, eval_batch):
            ed = min(st + eval_batch, n)
            bsz = ed - st
            sparse_gt_chunk = sparse_gt_eval[st:ed]
            raw_gt_chunk = raw_gt_eval[st:ed] if has_rho else None

            h0 = (sparse_gt_chunk - mean) / (std + 1e-7)
            h0 = torch.from_numpy(h0).float().to(self.device, non_blocking=True)
            samples = gdf.init_noise.unsqueeze(0).repeat(bsz, 1, 1, 1).to(self.device)

            for t in reversed(range(T)):
                tt = torch.full((bsz,), t, device=self.device, dtype=torch.long)
                with amp_autocast(enabled=use_amp):
                    out = net(samples, tt)
                out = out.float()

                h0_pred = predict_h0(gdf, samples, tt, out, pred_mode, clip_flag)

                if print_steps:
                    sparse_pred_step = (h0_pred * (std + 1e-7) + mean).detach().cpu().numpy()
                    nmse_step, rho_step = evaluate_crnet_metrics_np(
                        sparse_pred_step, sparse_gt_chunk, raw_gt_chunk, self.device, batch_size=min(500, bsz)
                    )
                    step_nmse_sum[t] += float(nmse_step) * bsz
                    if has_rho and rho_step is not None:
                        step_rho_sum[t] += float(rho_step) * bsz
                    step_cnt[t] += bsz

                resid = h0 - h0_pred
                best_idx = select_codebook_index(gdf, t, resid)
                samples, _ = gdf.p_sample_with_index(out, samples, tt, best_idx, clip_flag, pred_mode)

            rec_nchw = (samples * (std + 1e-7) + mean).detach().cpu().numpy()
            nmse_b, rho_b = evaluate_crnet_metrics_np(rec_nchw, sparse_gt_chunk, raw_gt_chunk, self.device, batch_size=min(500, bsz))
            nmse_sum += float(nmse_b) * bsz
            if has_rho and rho_b is not None:
                rho_sum += float(rho_b) * bsz
            cnt += bsz

        if print_steps:
            for t in reversed(range(T)):
                nmse_t, rho_t = aggregate_step_metrics(step_nmse_sum, step_rho_sum, step_cnt, t, has_rho)
                if rho_t is None:
                    logger.info(f'[train_diffusion][eval][step] t={t} NMSE(dB)={nmse_t:.6f}')
                else:
                    logger.info(f'[train_diffusion][eval][step] t={t} NMSE(dB)={nmse_t:.6f} rho={rho_t:.6f}')

        nmse = nmse_sum / max(1, cnt)
        rho = (rho_sum / max(1, cnt)) if has_rho else None
        return nmse, rho

    def train(self):
        cfg = self.cfg
        gen_mode = normalize_gen_mode(getattr(cfg, 'GEN_MODE', 'ddpm'))
        train_tag = 'train_diffusion'
        ckpt_name, ema_ckpt_name = checkpoint_names(cfg)
        logger.info(f'[{train_tag}] start gen_mode={gen_mode}')

        ensure_dir(cfg.DIFF_CKPT_DIR)
        ensure_dir(cfg.DIFF_RESULTS_DIR)

        if self.device.type == 'cuda':
            if bool(getattr(cfg, 'DIFF_FAST_BENCHMARK', True)):
                torch.backends.cudnn.benchmark = True
            if bool(getattr(cfg, 'DIFF_ALLOW_TF32', True)):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision(str(getattr(cfg, 'DIFF_MATMUL_PRECISION', 'high')))
            except Exception:
                pass

        net = self._new_unet()
        model_params = count_model_params(net)
        model_meta = model_metadata_from_cfg(cfg, prefix='DIFF_MODEL', params=model_params)
        logger.info(f'[{train_tag}] model_meta={model_meta}')

        x = load_ht_mat(cfg.TRAIN_FILE, cfg.MAT_SIZE, cfg.IN_CHANNELS)
        train_samples = int(getattr(cfg, 'DIFF_TRAIN_SAMPLES', 0))
        if train_samples > 0:
            train_samples = min(train_samples, int(x.shape[0]))
            x = x[:train_samples]
        mean = float(np.mean(x))
        std = float(np.std(x))
        logger.info(
            f'[{train_tag}] csi tensor_contract=NCHW shape={tuple(x.shape)} '
            f'train_samples={int(x.shape[0])} '
            f'normalized_tensor=H0 mean={mean:.6f} std={std:.6f} '
            f'normalization="H0=(HT-mean)/(std+1e-7)" inverse="HT_hat=H_hat*(std+1e-7)+mean"'
        )
        norm_stats_path = os.path.join(cfg.DIFF_RESULTS_DIR, 'normalization.npz')
        save_norm_stats(norm_stats_path, mean, std, source_file=cfg.TRAIN_FILE, tensor_shape=x.shape, gen_mode=gen_mode)
        x = normalize_csi_array(x, mean, std)
        write_json_manifest(
            os.path.join(cfg.DIFF_RESULTS_DIR, 'run_manifest.json'),
            {'gen_mode': gen_mode, 'model': model_meta, 'train_file': cfg.TRAIN_FILE, 'test_file': cfg.TEST_FILE, 'norm_stats_path': norm_stats_path, 'train_samples': int(x.shape[0]), 'normalization': {'mean': mean, 'std': std, 'source_file': cfg.TRAIN_FILE, 'tensor_shape': list(x.shape), 'tensor_layout': 'NCHW', 'normalized_tensor': 'H0', 'normalization_formula': 'H0=(HT-mean)/(std+1e-7)', 'inverse_formula': 'HT_hat=H_hat*(std+1e-7)+mean'}},
        )

        num_workers = int(getattr(cfg, 'DIFF_TRAIN_WORKERS', 8))
        prefetch = int(getattr(cfg, 'DIFF_TRAIN_PREFETCH', 4))
        loader_kwargs = dict(
            batch_size=cfg.DIFF_BATCH,
            shuffle=True,
            num_workers=max(0, num_workers),
            pin_memory=True,
            drop_last=bool(getattr(cfg, 'DIFF_TRAIN_DROP_LAST', True)),
        )
        if num_workers > 0:
            loader_kwargs['persistent_workers'] = True
            loader_kwargs['prefetch_factor'] = max(2, prefetch)
        loader = DataLoader(TensorDataset(torch.from_numpy(x)), **loader_kwargs)

        gdf = GaussianDiffusion(
            cfg.DIFF_BETA_START,
            cfg.DIFF_BETA_END,
            int(cfg.DIFF_TIMESTEPS),
            self.device,
            schedule=getattr(cfg, 'DIFF_NOISE_SCHED_TRAIN', 'linear'),
        )
        start_epoch = 1

        if cfg.DIFF_RESUME_PATH and os.path.isfile(cfg.DIFF_RESUME_PATH):
            net.load_state_dict(load_torch(cfg.DIFF_RESUME_PATH, map_location=self.device))
            start_epoch = parse_diff_epoch_from_path(cfg.DIFF_RESUME_PATH) + 1
            logger.info(f'[{train_tag}] resume from {cfg.DIFF_RESUME_PATH} (start_epoch={start_epoch})')

        opt = optim.Adam(net.parameters(), lr=float(cfg.DIFF_LR), eps=1e-7)
        loss_fn = nn.MSELoss()

        if cfg.DIFF_LR_SCHED == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.DIFF_EPOCHS)
        elif cfg.DIFF_LR_SCHED == 'multistep':
            scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[1000, 2000, 2600], gamma=0.1)
        else:
            scheduler = None

        use_amp = bool(cfg.DIFF_USE_AMP and self.device.type == 'cuda')
        use_channels_last = bool(getattr(cfg, 'DIFF_CHANNELS_LAST', True)) and self.device.type == 'cuda'
        scaler = make_grad_scaler(enabled=use_amp)
        ema = EMA(net, cfg.DIFF_EMA_DECAY) if cfg.DIFF_USE_EMA else None

        eval_every = int(getattr(cfg, 'DIFF_EVAL_EVERY', 0))
        eval_samples = int(getattr(cfg, 'DIFF_EVAL_SAMPLES', 2000))
        sparse_gt_eval, raw_gt_eval = None, None
        if eval_every > 0:
            sparse_gt_eval = load_ht_mat(cfg.TEST_FILE, cfg.MAT_SIZE, cfg.IN_CHANNELS)
            if os.path.isfile(cfg.GT_RAW_FILE):
                raw_gt_eval = load_raw_hf_all(cfg.GT_RAW_FILE)
            logger.info(
                f'[{train_tag}] periodic eval enabled: every={eval_every}, samples={eval_samples}, '
                f'batch={int(getattr(cfg, "DIFF_EVAL_BATCH", 500))} '
                f'codec_selector={"ddpm"}'
            )

        loss_hist = []
        heap = []
        eval_hist = []
        best_eval_nmse = float('inf')
        best_eval_epoch = 0

        for ep in range(start_epoch, cfg.DIFF_EPOCHS + 1):
            net.train()
            epoch_loss = AverageMeter('diff_loss')

            for (h0,) in loader:
                h0 = h0.to(self.device, non_blocking=True)
                if use_channels_last:
                    h0 = h0.contiguous(memory_format=torch.channels_last)

                b = h0.shape[0]
                if str(getattr(cfg, 'DIFF_T_SAMPLER', 'uniform')).lower() == 'low_bias':
                    p = max(1e-6, float(getattr(cfg, 'DIFF_T_BIAS_POWER', 2.0)))
                    u = torch.rand((b,), device=self.device)
                    t = torch.clamp((u.pow(p) * cfg.DIFF_TIMESTEPS).long(), max=cfg.DIFF_TIMESTEPS - 1)
                else:
                    t = torch.randint(0, cfg.DIFF_TIMESTEPS, (b,), device=self.device).long()

                noise = torch.randn_like(h0)
                xt = gdf.q_sample(h0, t, noise)

                opt.zero_grad(set_to_none=True)
                with amp_autocast(enabled=scaler.is_enabled()):
                    out = net(xt, t)
                    target = h0 if (normalize_pred_mode(cfg.DIFF_PRED_MODE) == 'h0') else noise
                    if str(getattr(cfg, 'DIFF_LOSS_WEIGHT', 'none')).lower() == 'low_t':
                        per_sample = ((out - target) ** 2).flatten(1).mean(dim=1)
                        t_norm = t.float() / max(1, cfg.DIFF_TIMESTEPS - 1)
                        w = (1.0 - t_norm).pow(float(getattr(cfg, 'DIFF_LOSS_T_POWER', 1.0)))
                        w = w / (w.mean() + 1e-8)
                        loss = (per_sample * w).mean()
                    loss = loss_fn(target, out)

                scaler.scale(loss).backward()
                if cfg.DIFF_GRAD_CLIP and cfg.DIFF_GRAD_CLIP > 0:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(net.parameters(), cfg.DIFF_GRAD_CLIP)
                scaler.step(opt)
                scaler.update()

                if ema is not None:
                    ema.update(net)
                epoch_loss.update(loss.item(), h0.size(0))

            loss_hist.append(epoch_loss.avg)
            logger.info(f'[{train_tag}] ep {ep:05d}/{cfg.DIFF_EPOCHS} loss={epoch_loss.avg:.6f} lr={opt.param_groups[0]["lr"]:.3e}')

            improved_eval = False
            nmse_eval, rho_eval = None, None
            if eval_every > 0 and (ep % eval_every == 0):
                eval_net = self._make_eval_net(net, ema)
                nmse_eval, rho_eval = self._evaluate_during_train(eval_net, mean, std, sparse_gt_eval, raw_gt_eval, eval_samples)
                eval_row = {'epoch': int(ep), 'loss': float(epoch_loss.avg), 'nmse': float(nmse_eval), 'rho': None if rho_eval is None else float(rho_eval), 'eval_samples': int(min(eval_samples, sparse_gt_eval.shape[0])), 'ckpt_name': ckpt_name, 'ema_ckpt_name': ema_ckpt_name, 'gen_mode': gen_mode, 'model_arch': model_meta['model_arch'], 'model_preset': model_meta['model_preset'], 'model_params': int(model_params)}
                eval_hist.append(eval_row)
                with open(os.path.join(cfg.DIFF_RESULTS_DIR, 'eval_history.jsonl'), 'a', encoding='utf-8') as f:
                    f.write(json.dumps(eval_row, sort_keys=True) + '\n')
                if rho_eval is None:
                    logger.info(f'[{train_tag}][eval] ep={ep:05d} NMSE(dB)={nmse_eval:.6f}')
                else:
                    logger.info(f'[{train_tag}][eval] ep={ep:05d} NMSE(dB)={nmse_eval:.6f} rho={rho_eval:.6f}')
                improved_eval = float(nmse_eval) < best_eval_nmse
                if improved_eval:
                    best_eval_nmse = float(nmse_eval)
                    best_eval_epoch = int(ep)
                    logger.info(
                        f'[{train_tag}][best] ep={ep:05d} NMSE(dB)={best_eval_nmse:.6f} '
                        f'rho={(float(rho_eval) if rho_eval is not None else "nan")}'
                    )

            if scheduler is not None:
                scheduler.step()

            save_periodic = should_save_periodic_checkpoint(ep, cfg.DIFF_SAVE_EVERY)
            if save_periodic:
                dname = f'ep{ep:05d}_loss{epoch_loss.avg:.6f}'
                save_dir = ensure_dir(os.path.join(cfg.DIFF_CKPT_DIR, dname))
                torch.save(net.state_dict(), os.path.join(save_dir, ckpt_name))
                if ema is not None:
                    torch.save(ema.state_dict(), os.path.join(save_dir, ema_ckpt_name))

            latest_dir = ensure_dir(os.path.join(cfg.DIFF_CKPT_DIR, 'latest'))
            torch.save(net.state_dict(), os.path.join(latest_dir, ckpt_name))
            if ema is not None:
                torch.save(ema.state_dict(), os.path.join(latest_dir, ema_ckpt_name))
            write_json_manifest(
                os.path.join(latest_dir, 'checkpoint_meta.json'),
                {'epoch': int(ep), 'loss': float(epoch_loss.avg), 'gen_mode': gen_mode, 'model_arch': model_meta['model_arch'], 'model_preset': model_meta['model_preset'], 'model_params': int(model_params), 'ckpt_name': ckpt_name, 'ema_ckpt_name': ema_ckpt_name, 'eval_nmse': None if nmse_eval is None else float(nmse_eval), 'eval_rho': None if rho_eval is None else float(rho_eval)},
            )

            if improved_eval:
                best_dir = ensure_dir(os.path.join(cfg.DIFF_CKPT_DIR, 'best'))
                torch.save(net.state_dict(), os.path.join(best_dir, ckpt_name))
                if ema is not None:
                    torch.save(ema.state_dict(), os.path.join(best_dir, ema_ckpt_name))
                write_json_manifest(
                    os.path.join(best_dir, 'checkpoint_meta.json'),
                    {'epoch': int(ep), 'loss': float(epoch_loss.avg), 'gen_mode': gen_mode, 'model_arch': model_meta['model_arch'], 'model_preset': model_meta['model_preset'], 'model_params': int(model_params), 'best_metric': 'eval_nmse_db_min', 'eval_nmse': float(nmse_eval), 'eval_rho': None if rho_eval is None else float(rho_eval), 'eval_samples': int(min(eval_samples, sparse_gt_eval.shape[0])), 'ckpt_name': ckpt_name, 'ema_ckpt_name': ema_ckpt_name},
                )

            if save_periodic:
                heapq.heappush(heap, (-epoch_loss.avg, ep, epoch_loss.avg))
                if len(heap) > cfg.DIFF_MAX_KEEP:
                    _, worst_ep, worst_loss = heapq.heappop(heap)
                    shutil.rmtree(os.path.join(cfg.DIFF_CKPT_DIR, f'ep{worst_ep:05d}_loss{worst_loss:.6f}'), ignore_errors=True)

        sio.savemat(os.path.join(cfg.DIFF_RESULTS_DIR, 'loss.mat'), {'loss': np.array(loss_hist)})
        if eval_hist:
            write_json_manifest(os.path.join(cfg.DIFF_RESULTS_DIR, 'eval_summary.json'), {'eval_history': eval_hist})
        logger.info(
            f'[{train_tag}] done ckpt_name={ckpt_name} ema_ckpt_name={ema_ckpt_name} '
            f'best_eval_epoch={best_eval_epoch} best_eval_nmse={best_eval_nmse}'
        )
