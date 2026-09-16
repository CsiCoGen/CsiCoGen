import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def kernel_init(scale):
    def _initializer(module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            real_scale = max(scale, 1e-10)
            nn.init.xavier_uniform_(module.weight, gain=real_scale)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    return _initializer


def get_timestep_embedding(timesteps, embedding_dim):
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


class TimeEmbedding(nn.Module):
    def __init__(self, temb_dim):
        super().__init__()
        self.temb_dim = temb_dim
        self.dense_temb = nn.Sequential(
            nn.Linear(temb_dim, temb_dim),
            nn.SiLU(),
            nn.Linear(temb_dim, temb_dim),
        )
        self.apply(kernel_init(1.0))

    def forward(self, t):
        t_emb = get_timestep_embedding(t, self.temb_dim)
        return self.dense_temb(t_emb)


class LiteAttention2d(nn.Module):
    def __init__(self, channels, heads=4, downsample=2, max_hw=8):
        super().__init__()
        assert channels % heads == 0
        self.heads = heads
        self.head_dim = channels // heads
        self.scale = self.head_dim ** -0.5
        self.downsample = max(1, int(downsample))
        self.max_hw = max(2, int(max_hw))

        self.norm = nn.GroupNorm(8, channels, eps=1e-5)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.apply(kernel_init(1.0))
        self.proj.apply(kernel_init(0.0))

    def forward(self, x):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)

        auto_ds_h = max(1, math.ceil(h / self.max_hw))
        auto_ds_w = max(1, math.ceil(w / self.max_hw))
        ds = max(self.downsample, auto_ds_h, auto_ds_w)
        xs = F.avg_pool2d(x, kernel_size=ds, stride=ds) if ds > 1 else x
        hs, ws = xs.shape[-2:]

        qkv = self.qkv(xs)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        def reshape_heads(t):
            t = t.view(b, self.heads, self.head_dim, hs * ws)
            return t.permute(0, 1, 3, 2)

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).contiguous().view(b, c, hs, ws)
        out = self.proj(out)

        if ds > 1:
            out = F.interpolate(out, size=(h, w), mode='nearest')
        return x_in + out


class ResAttnBlock2d(nn.Module):
    def __init__(self, channels, temb_dim, use_attn=False, attn_heads=4, attn_downsample=2, attn_max_hw=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels, eps=1e-5)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels, eps=1e-5)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.temb_proj = nn.Linear(temb_dim, channels * 2)
        self.attn = LiteAttention2d(channels, attn_heads, attn_downsample, attn_max_hw) if use_attn else None

        self.apply(kernel_init(1.0))
        self.conv2.apply(kernel_init(0.0))

    def forward(self, x, temb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        style = self.temb_proj(F.silu(temb))[:, :, None, None]
        scale, shift = style.chunk(2, dim=1)
        h = h * (1 + scale) + shift

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        x = x + h
        if self.attn is not None:
            x = self.attn(x)
        return x


class CSI_ResAttnNet(nn.Module):
    def __init__(
        self,
        mat_size=32,
        img_channels=2,
        out_channels=None,
        dim=128,
        num_blocks=8,
        attn_every=2,
        attn_heads=4,
        attn_downsample=2,
        attn_max_hw=8,
    ):
        super().__init__()
        out_channels = img_channels if out_channels is None else int(out_channels)
        self.time_embedding = TimeEmbedding(dim)
        self.head = nn.Conv2d(img_channels, dim, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            use_attn = (attn_every > 0) and ((i + 1) % attn_every == 0)
            self.blocks.append(
                ResAttnBlock2d(dim, dim, use_attn, attn_heads, attn_downsample, attn_max_hw)
            )
        self.tail_norm = nn.GroupNorm(8, dim, eps=1e-5)
        self.tail = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1)

        self.apply(kernel_init(1.0))
        self.tail.apply(kernel_init(0.0))

    def forward(self, x, t):
        temb = self.time_embedding(t)
        h = self.head(x)
        for block in self.blocks:
            h = block(h, temb)
        h = self.tail_norm(h)
        h = F.silu(h)
        return self.tail(h)
