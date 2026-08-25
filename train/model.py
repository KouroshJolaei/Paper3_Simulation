#!/usr/bin/env python3
"""
model.py — the contact-completion network and its masked loss.

INPUT  (1, 96, 96)   the initial grasp alone on the pinned pad-frame canvas
OUTPUT (1, 96, 96)   the extended contact map

A U-Net, because the job is image-to-image at the same size: the encoder sees
enough context to know what kind of contact this is, the decoder puts detail
back, and the skip connections stop the 22x37 mm imprint being blurred away
by the bottleneck.

Deliberately small — 4 levels, base 32 channels, ~2 M parameters. The dataset
is tens of pairs, not tens of thousands, and a bigger network would memorise
them. This is a starting point to be grown when the data justifies it, not a
tuned final architecture.

THE MASKED LOSS IS THE POINT OF THIS FILE.
Roughly 60% of the canvas was never visited by any pad. Those cells are not
"no contact" — they are "nobody looked", and they are stored as zero exactly
like a measured zero. Scoring them teaches the network the OUTLINE OF THE
SWEEP, which is a fact about the grid, not about the object, and it would do
so while the loss curve fell and SSIM looked respectable. So every loss and
metric here multiplies by target_mask and divides by its sum.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, base=32, depth=4, in_ch=1, out_ch=1):
        super().__init__()
        self.depth = depth
        chs = [base * (2 ** i) for i in range(depth)]
        self.down = nn.ModuleList()
        c = in_ch
        for ch in chs:
            self.down.append(_block(c, ch)); c = ch
        self.bott = _block(c, c * 2); c = c * 2
        self.up_t = nn.ModuleList()
        self.up_b = nn.ModuleList()
        for ch in reversed(chs):
            self.up_t.append(nn.ConvTranspose2d(c, ch, 2, stride=2))
            self.up_b.append(_block(ch * 2, ch)); c = ch
        self.head = nn.Conv2d(c, out_ch, 1)

    def forward(self, x):
        skips = []
        for b in self.down:
            x = b(x); skips.append(x); x = F.max_pool2d(x, 2)
        x = self.bott(x)
        for t, b, s in zip(self.up_t, self.up_b, reversed(skips)):
            x = t(x)
            if x.shape[-2:] != s.shape[-2:]:      # 96 -> 48 -> 24 -> 12 -> 6
                x = F.interpolate(x, size=s.shape[-2:], mode="nearest")
            x = b(torch.cat([x, s], 1))
        # ReLU: contact pressure cannot be negative, and letting the network
        # produce negatives lets it cancel error in one cell against another.
        return F.relu(self.head(x))


def masked_l1(pred, target, mask):
    """Mean absolute error over MEASURED cells only.

    L1 rather than L2 because the targets are dominated by near-zero cells
    with a small bright ridge; squared error would let the network do well by
    predicting the background everywhere and giving up on the ridge, which is
    the only part anyone cares about."""
    m = mask.float()
    denom = m.sum().clamp(min=1.0)
    return ((pred - target).abs() * m).sum() / denom


def masked_mse(pred, target, mask):
    m = mask.float()
    return (((pred - target) ** 2) * m).sum() / m.sum().clamp(min=1.0)


def masked_ssim(pred, target, mask, C1=1e-4, C2=9e-4):
    """SSIM computed over measured cells only.

    Paper 1 and Paper 2 both report SSIM, so it is here for continuity. This
    is the GLOBAL (not windowed) form, evaluated on the masked region: a
    windowed SSIM would slide its window across unvisited cells and mix
    "nobody looked" into every statistic near the boundary.
    """
    out = []
    for p, t, m in zip(pred, target, mask):
        mm = m.bool()
        if mm.sum() < 8:
            continue
        a, b = p[mm], t[mm]
        mu_a, mu_b = a.mean(), b.mean()
        va, vb = a.var(unbiased=False), b.var(unbiased=False)
        cov = ((a - mu_a) * (b - mu_b)).mean()
        s = (((2 * mu_a * mu_b + C1) * (2 * cov + C2)) /
             ((mu_a ** 2 + mu_b ** 2 + C1) * (va + vb + C2)))
        out.append(s)
    if not out:
        return torch.tensor(float("nan"))
    return torch.stack(out).mean()


def tactile_centroid_error_mm(pred, target, mask, thresh_frac=0.4375,
                              mm_per_cell=1.0):
    """Paper-1/2 Tactile Centroid deviation, in mm, on the masked region.

    Binarise each map at a fraction of its own range (the same 0.4375 the
    blob-axis work uses), take the centroid of each, and report the distance
    between them. Returns NaN for a map with no contact above threshold —
    reported rather than counted as zero error."""
    out = []
    for p, t, m in zip(pred, target, mask):
        mm = m.bool()
        if mm.sum() < 8:
            continue
        res = []
        for img in (p, t):
            v = img.clone()
            v[~mm] = 0
            lo, hi = v[mm].min(), v[mm].max()
            if (hi - lo) <= 0:
                res.append(None); continue
            b = (v >= lo + thresh_frac * (hi - lo)) & mm
            if b.sum() == 0:
                res.append(None); continue
            idx = b.nonzero(as_tuple=False).float()
            res.append(idx.mean(0))
        if res[0] is None or res[1] is None:
            out.append(torch.tensor(float("nan"), device=p.device))
        else:
            out.append(torch.linalg.norm(res[0] - res[1]) * mm_per_cell)
    if not out:
        return torch.tensor(float("nan"))
    return torch.stack(out)
