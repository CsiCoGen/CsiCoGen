import numpy as np
import torch


@torch.no_grad()
def evaluator_crnet(sparse_pred, sparse_gt, raw_gt):
    """Return CRNet-aligned channel correlation and NMSE."""
    nt = 32
    nc = 32
    nc_expand = 257

    sparse_gt = sparse_gt - 0.5
    sparse_pred = sparse_pred - 0.5

    power_gt = sparse_gt[:, 0, :, :] ** 2 + sparse_gt[:, 1, :, :] ** 2
    difference = sparse_gt - sparse_pred
    mse = difference[:, 0, :, :] ** 2 + difference[:, 1, :, :] ** 2
    nmse = 10 * torch.log10((mse.sum(dim=[1, 2]) / power_gt.sum(dim=[1, 2])).mean())

    n = sparse_pred.size(0)
    sparse_pred = sparse_pred.permute(0, 2, 3, 1)
    zeros = sparse_pred.new_zeros((n, nt, nc_expand - nc, 2))
    sparse_pred = torch.cat((sparse_pred, zeros), dim=2)

    sparse_pred_c = torch.view_as_complex(sparse_pred.contiguous())
    raw_pred_c = torch.fft.fft(sparse_pred_c, dim=2)[:, :, :125]
    raw_gt_c = torch.view_as_complex(raw_gt.contiguous())

    norm_pred = torch.sqrt((raw_pred_c.abs() ** 2).sum(dim=1))
    norm_gt = torch.sqrt((raw_gt_c.abs() ** 2).sum(dim=1))

    cross = (raw_pred_c * torch.conj(raw_gt_c)).sum(dim=1)
    norm_cross = torch.abs(cross)

    corr = norm_cross / (norm_pred * norm_gt + 1e-12)
    rho = corr.mean().real
    return rho, nmse.real


@torch.no_grad()
def nmse_crnet_only(sparse_pred, sparse_gt):
    sparse_gt = sparse_gt - 0.5
    sparse_pred = sparse_pred - 0.5
    power_gt = sparse_gt[:, 0, :, :] ** 2 + sparse_gt[:, 1, :, :] ** 2
    difference = sparse_gt - sparse_pred
    mse = difference[:, 0, :, :] ** 2 + difference[:, 1, :, :] ** 2
    nmse = 10 * torch.log10((mse.sum(dim=[1, 2]) / power_gt.sum(dim=[1, 2])).mean())
    return nmse.real


@torch.no_grad()
def evaluate_crnet_metrics_np(
    sparse_pred_np,
    sparse_gt_np,
    raw_gt_cpu,
    device,
    batch_size=500,
):
    """Evaluate CRNet-aligned NMSE and channel correlation."""
    n = sparse_pred_np.shape[0]
    nmse_sum = 0.0
    rho_sum = 0.0
    cnt = 0
    has_rho = raw_gt_cpu is not None

    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        p = torch.from_numpy(sparse_pred_np[s:e]).float().to(device)
        g = torch.from_numpy(sparse_gt_np[s:e]).float().to(device)
        bsz = e - s

        if has_rho:
            r = raw_gt_cpu[s:e].float().to(device)
            rho_b, nmse_b = evaluator_crnet(p, g, r)
            rho_sum += float(rho_b.item()) * bsz
        else:
            nmse_b = nmse_crnet_only(p, g)

        nmse_sum += float(nmse_b.item()) * bsz
        cnt += bsz

    nmse = nmse_sum / max(1, cnt)
    rho = (rho_sum / max(1, cnt)) if has_rho else None
    return nmse, rho


def evaluate_nmse_crnet_np(sparse_pred_np, sparse_gt_np, batch_size=500):
    n = sparse_pred_np.shape[0]
    nmse_sum = 0.0
    cnt = 0

    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        sparse_pred = sparse_pred_np[s:e] - 0.5
        sparse_gt = sparse_gt_np[s:e] - 0.5
        power_gt = sparse_gt[:, 0, :, :] ** 2 + sparse_gt[:, 1, :, :] ** 2
        difference = sparse_gt - sparse_pred
        mse = difference[:, 0, :, :] ** 2 + difference[:, 1, :, :] ** 2
        ratio = mse.sum(axis=(1, 2)) / power_gt.sum(axis=(1, 2))
        nmse_b = 10 * np.log10(ratio.mean())
        bsz = e - s
        nmse_sum += float(nmse_b) * bsz
        cnt += bsz

    return nmse_sum / max(1, cnt)


def aggregate_step_metrics(step_nmse_sum, step_rho_sum, step_cnt, t, has_rho):
    c = max(1, int(step_cnt[t]))
    nmse_t = float(step_nmse_sum[t] / c)
    if has_rho:
        rho_t = float(step_rho_sum[t] / c)
        return nmse_t, rho_t
    return nmse_t, None


def init_step_buffers(T, has_rho):
    step_nmse_sum = np.zeros((T,), dtype=np.float64)
    step_rho_sum = np.zeros((T,), dtype=np.float64) if has_rho else None
    step_cnt = np.zeros((T,), dtype=np.int64)
    return step_nmse_sum, step_rho_sum, step_cnt
