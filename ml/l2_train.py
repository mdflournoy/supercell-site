#!/usr/bin/env python3
"""Single-site stage 4: train MesoNet on Level II patches (ml/l2_patches.py) on CPU.

Same network, loss and schedule as the MRMS model (ml/train.py), with 28 input channels
(l2_volume.CHANNELS). Rotation augmentation only (no flips, which would turn cyclonic shear
anticyclonic). The detection threshold and tracker settings are tuned later on the held-out days.

  python ml/l2_train.py --patches patches/ --out model/ --max-minutes 300
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from frames import heatmap  # noqa: E402
from l2_patches import P, dequantize  # noqa: E402
from model import MesoNet, focal_loss  # noqa: E402

NCH = 28


def load(folder: Path):
    parts = [np.load(f) for f in sorted(folder.glob("l2patches_*.npz"))]
    parts = [p for p in parts if len(p["x"])]
    x = np.concatenate([p["x"] for p in parts]); m = np.concatenate([p["m"] for p in parts])
    kind = np.concatenate([p["kind"] for p in parts])
    by = [[] for _ in range(len(x))]
    off = 0
    for p in parts:
        for i, r, c in p["pts"]:
            by[off + int(i)].append((r, c))
        off += len(p["x"])
    return x, m, kind, by, [str(p["day"]) for p in parts]


def make_batch(idx, x, m, by, rng):
    X, Y, M = [], [], []
    for i in idx:
        xi = dequantize(x[i])
        yi = heatmap(np.array(by[i], np.float32).reshape(-1, 2), (P, P))
        mi = np.unpackbits(m[i])[:P * P].reshape(P, P).astype(np.float32)
        k = int(rng.integers(4))
        X.append(np.ascontiguousarray(np.rot90(xi, k, (1, 2))))
        Y.append(np.ascontiguousarray(np.rot90(yi, k))); M.append(np.ascontiguousarray(np.rot90(mi, k)))
    return torch.from_numpy(np.stack(X)), torch.from_numpy(np.stack(Y)), torch.from_numpy(np.stack(M))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches", required=True)
    ap.add_argument("--out", default="model")
    ap.add_argument("--max-minutes", type=float, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t_start = time.time()
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train_log.txt", "a")

    def log(s):
        line = f"[{(time.time() - t_start) / 60:6.1f} min] {s}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    x, m, kind, by, days = load(Path(a.patches))
    n = len(x)
    log(f"{n} patches from {len(days)} days: pos {int((kind == 1).sum())}, hard neg {int((kind == 2).sum())}, "
        f"random {int((kind == 3).sum())}; {sum(len(b) for b in by)} meso centers")
    model = MesoNet(cin=NCH)
    log(f"MesoNet: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters, {NCH} channels")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    budget = a.max_minutes * 60
    WARMUP = 100
    step, epoch, hist = 0, 0, []
    t0 = time.time() - t_start
    done = False
    while not done:
        perm = rng.permutation(n)
        model.train()
        ep_loss, k = 0.0, 0
        for b in range(math.ceil(n / a.batch)):
            X, Y, M = make_batch(perm[b * a.batch:(b + 1) * a.batch], x, m, by, rng)
            loss = focal_loss(model(X), Y, M)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += float(loss); k += 1; step += 1
            el = time.time() - t_start
            frac = min(1.0, (el - t0) / max(1.0, budget - t0))
            lr = a.lr * min(1.0, step / WARMUP) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * frac)))
            for g in opt.param_groups:
                g["lr"] = lr
            if step == 20:
                log(f"~{(el - t0) / 20:.2f} s/step")
            if step % 200 == 0:
                log(f"epoch {epoch} step {step}: loss {ep_loss / k:.3f} lr {lr:.2e}")
                torch.save(model.state_dict(), out / "mesonet_l2.pt")
            if el > budget:
                done = True
                break
        epoch += 1
        hist.append({"epoch": epoch, "steps": step, "loss": ep_loss / max(1, k), "minutes": (time.time() - t_start) / 60})
        log(f"epoch {epoch} done: mean loss {ep_loss / max(1, k):.3f}")
        torch.save(model.state_dict(), out / "mesonet_l2.pt")
    (out / "train_meta.json").write_text(json.dumps(
        {"patches": n, "train_days": days, "epochs": epoch, "steps": step, "history": hist, "channels": NCH,
         "minutes": round((time.time() - t_start) / 60, 1)}, indent=1))


if __name__ == "__main__":
    main()
