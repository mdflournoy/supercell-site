#!/usr/bin/env python3
"""Stage 2b: train MesoNet on CPU from patch files, then tune the detection threshold on held-out days.

  python ml/train.py --patches patches/ --val-tars tars/20220322.tar ... --out model/ --max-minutes 240

Training patches come from every patches_*.npz given (the workflow passes only the 2022 training days).
Validation runs the model over full frames of the held-out days (every 3rd frame) and scores detected
centers against tracked mesos (match within 5 km, inside the Southeast truth region).
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
from frames import KM_PER_PX, Day, extract, heatmap, stack_raw, static_fields, to_input, valid_mask  # noqa: E402
from model import MesoNet, focal_loss, peaks, predict_full  # noqa: E402
from evaluate import match_points  # noqa: E402


def load_patches(folder: Path):
    parts = [np.load(f) for f in sorted(folder.glob("patches_*.npz"))]
    az = np.concatenate([p["az"] for p in parts]); ref = np.concatenate([p["ref"] for p in parts])
    beam = np.concatenate([p["beam"] for p in parts]); orig = np.concatenate([p["orig"] for p in parts])
    kind = np.concatenate([p["kind"] for p in parts])
    pts, off = [], 0
    for p in parts:
        q = p["pts"].copy()
        q[:, 0] += off
        pts.append(q)
        off += len(p["az"])
    pts = np.concatenate(pts) if pts else np.zeros((0, 4), np.float32)
    by = [[] for _ in range(len(az))]
    for i, r, c, _ in pts:
        by[int(i)].append((r, c))
    files = [str(p["day"]) for p in parts]
    return az, ref, beam, orig, kind, by, files


def make_batch(idx, az, ref, beam, orig, by, se, rng, augment=True):
    X, Y, M = [], [], []
    P = az.shape[-1]
    for i in idx:
        x = to_input(az[i], ref[i], beam[i])
        rc = np.array(by[i], np.float32).reshape(-1, 2)
        y = heatmap(rc, (P, P))
        r0, c0 = orig[i]
        m = valid_mask(rc, se[r0:r0 + P, c0:c0 + P].copy())
        if augment:
            k = int(rng.integers(4))  # rotations keep cyclonic rotation cyclonic (no flips!)
            x, y, m = np.rot90(x, k, (1, 2)), np.rot90(y, k), np.rot90(m, k)
        X.append(np.ascontiguousarray(x)); Y.append(np.ascontiguousarray(y)); M.append(np.ascontiguousarray(m))
    return torch.from_numpy(np.stack(X)), torch.from_numpy(np.stack(Y)), torch.from_numpy(np.stack(M).astype(np.float32))


def validate(model, days, beam, se, thresholds, every=3, log=print):
    """Detection scores on full frames; returns per-threshold hits/misses/false alarms."""
    res = {t: {"hits": 0, "misses": 0, "false": 0} for t in thresholds}
    nf = 0
    for day in days:
        for fi in range(0, len(day), every):
            az, ref = stack_raw(day, fi)
            prob = predict_full(model, to_input(az, ref, beam))
            lab = day.labels_at(fi)
            truth = lab[["row", "col"]].to_numpy(float) if len(lab) else np.zeros((0, 2))
            for t in thresholds:
                det = peaks(prob, t)
                h, m, f = match_points(det[:, :2], truth, se, max_km=5.0)
                res[t]["hits"] += h; res[t]["misses"] += m; res[t]["false"] += f
            nf += 1
        log(f"  validated {day.file}: {nf} frames so far")
    for t, r in res.items():
        h, m, f = r["hits"], r["misses"], r["false"]
        r.update(pod=h / max(1, h + m), far=f / max(1, h + f), csi=h / max(1, h + m + f))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches", required=True)
    ap.add_argument("--val-tars", nargs="*", default=[])
    ap.add_argument("--out", default="model")
    ap.add_argument("--max-minutes", type=float, default=240)
    ap.add_argument("--val-minutes", type=float, default=60, help="time reserved for validation")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t_start = time.time()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train_log.txt", "a")

    def log(s):
        line = f"[{(time.time() - t_start) / 60:6.1f} min] {s}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    beam, se = static_fields()
    az, ref, bm, orig, kind, by, files = load_patches(Path(a.patches))
    n = len(az)
    log(f"{n} patches from {len(files)} days: pos {int((kind == 1).sum())}, hard neg {int((kind == 2).sum())}, "
        f"random {int((kind == 3).sum())}; {sum(len(b) for b in by)} meso centers")

    model = MesoNet()
    log(f"MesoNet: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    budget = (a.max_minutes - a.val_minutes) * 60
    steps_per_epoch = math.ceil(n / a.batch)
    # estimate total steps from the first 20 steps' speed, for the cosine schedule
    step, epoch, total_steps = 0, 0, None
    hist = []
    while True:
        perm = rng.permutation(n)
        model.train()
        ep_loss, k = 0.0, 0
        for b in range(steps_per_epoch):
            idx = perm[b * a.batch:(b + 1) * a.batch]
            X, Y, M = make_batch(idx, az, ref, bm, orig, by, se, rng)
            loss = focal_loss(model(X), Y, M)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += float(loss); k += 1; step += 1
            el = time.time() - t_start
            if step == 20:
                total_steps = int(budget / (el / 20) * 0.95)
                log(f"~{el / 20:.2f} s/step → planning {total_steps} steps (~{total_steps / steps_per_epoch:.1f} epochs)")
            if total_steps:
                lr = a.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, step / total_steps)))
                for g in opt.param_groups:
                    g["lr"] = lr
            if step % 200 == 0:
                log(f"epoch {epoch} step {step}: loss {ep_loss / k:.3f}")
            if el > budget:
                break
        epoch += 1
        hist.append({"epoch": epoch, "steps": step, "loss": ep_loss / max(1, k), "minutes": (time.time() - t_start) / 60})
        log(f"epoch {epoch} done: mean loss {ep_loss / max(1, k):.3f}")
        torch.save(model.state_dict(), out / "mesonet.pt")
        if time.time() - t_start > budget:
            break

    # validation on held-out days
    thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]
    val = {}
    if a.val_tars and a.val_minutes > 0:
        days = [Day(extract(Path(t), Path("work-val"))) for t in a.val_tars]
        val = validate(model, days, beam, se, thresholds, log=log)
        best = max(val, key=lambda t: val[t]["csi"])
        log("threshold  POD   FAR   CSI   hits misses false")
        for t, r in val.items():
            log(f"  {t:4.2f}   {r['pod']:.2f}  {r['far']:.2f}  {r['csi']:.2f}  {r['hits']:5d} {r['misses']:5d} {r['false']:5d}")
        log(f"best threshold {best}")
    meta = {"patches": n, "train_days": files, "epochs": epoch, "steps": step, "history": hist,
            "val_days": [Path(t).stem for t in a.val_tars],
            "val": {str(k): v for k, v in val.items()},
            "best_threshold": max(val, key=lambda t: val[t]["csi"]) if val else None,
            "minutes": round((time.time() - t_start) / 60, 1)}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
