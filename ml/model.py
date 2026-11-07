"""Small U-Net that outputs a heat map (logits) of mid-level mesocyclone centers."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class MesoNet(nn.Module):
    def __init__(self, cin: int = 7, widths=(16, 32, 64, 96, 128)):
        super().__init__()
        self.enc = nn.ModuleList()
        c = cin
        for w in widths:
            self.enc.append(block(c, w))
            c = w
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for w in reversed(widths[:-1]):
            self.up.append(nn.Conv2d(c, w, 1))
            self.dec.append(block(2 * w, w))
            c = w
        self.head = nn.Conv2d(c, 1, 1)
        nn.init.constant_(self.head.bias, -4.6)  # start near p = 0.01

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x)
            if i < len(self.enc) - 1:
                skips.append(x)
                x = F.max_pool2d(x, 2)
        for up, dec in zip(self.up, self.dec):
            s = skips.pop()
            x = F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False)
            x = dec(torch.cat([up(x), s], 1))
        return self.head(x)[:, 0]


def focal_loss(logits, target, mask, alpha: float = 2.0, beta: float = 4.0):
    """CenterNet penalty-reduced focal loss; target = Gaussian heat map with 1 at centers; mask = valid pixels."""
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = (target >= 0.999).float() * mask
    neg = (1 - pos) * mask
    pos_loss = -torch.log(p) * (1 - p) ** alpha * pos
    neg_loss = -torch.log(1 - p) * p ** alpha * (1 - target) ** beta * neg
    npos = pos.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / npos


def predict_full(model, x: np.ndarray, tile: int = 1024, pad: int = 64, threads: int | None = None) -> np.ndarray:
    """Probability map for a full-domain input (C, H, W), tiled with overlap."""
    if threads:
        torch.set_num_threads(threads)
    C, H, W = x.shape
    out = np.zeros((H, W), np.float32)
    model.eval()
    with torch.no_grad():
        for j in range(0, H, tile):
            for i in range(0, W, tile):
                j0, i0 = max(0, j - pad), max(0, i - pad)
                j1, i1 = min(H, j + tile + pad), min(W, i + tile + pad)
                sub = x[:, j0:j1, i0:i1]
                if not np.any(sub[2] > 0.25):   # no echo >= 15 dBZ: nothing to find
                    continue
                h, w = sub.shape[1:]
                ph, pw = (-h) % 16, (-w) % 16
                t = torch.from_numpy(np.pad(sub, ((0, 0), (0, ph), (0, pw)))).unsqueeze(0)
                p = torch.sigmoid(model(t))[0, :h, :w].numpy()
                out[j:min(H, j + tile), i:min(W, i + tile)] = p[j - j0:j - j0 + min(tile, H - j), i - i0:i - i0 + min(tile, W - i)]
    return out


def peaks(prob: np.ndarray, thr: float, nms_px: int = 6):
    """Local maxima >= thr (non-maximum suppression over a (2*nms_px+1) window): array of (row, col, p)."""
    t = torch.from_numpy(prob)[None, None]
    m = F.max_pool2d(t, 2 * nms_px + 1, stride=1, padding=nms_px)[0, 0].numpy()
    rr, cc = np.where((prob >= thr) & (prob == m))
    return np.column_stack([rr, cc, prob[rr, cc]])
