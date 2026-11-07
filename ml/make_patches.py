#!/usr/bin/env python3
"""Stage 2a: cut training patches from one event day's frames.

Per frame:
  * one patch per tracked supercell active in the frame (centered near its mesos, random offset),
  * up to HARD hard-negative patches centered on strong 3-6 km AzShear (>= 5) inside the Southeast
    states and > 25 km from every tracked meso,
  * one random patch centered on echo >= 30 dBZ inside the Southeast states.
Each patch stores the raw int8/uint8 channels (see frames.py), its top-left corner on the full grid,
and the tracked meso centers inside it (patch row/col). Masks and heat maps are made at train time.

Output: OUT/patches_YYYYMMDD.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from frames import KM_PER_PX, Day, extract, stack_raw, static_fields  # noqa: E402

P = 192
HARD = 2
JITTER = 60  # px


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", required=True)
    ap.add_argument("--out", default="patches")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--work", default="work")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed + int(Path(a.tar).stem))
    day = Day(extract(Path(a.tar), Path(a.work)))
    beam, se = static_fields()
    H, W = se.shape
    far_px = int(25 / KM_PER_PX)

    AZ, REF, BEAM, ORIG, KIND, FRAME, PTS = [], [], [], [], [], [], []

    def add(az, ref, r0, c0, kind, fi, lab_rc):
        r0 = int(np.clip(r0, 0, H - P)); c0 = int(np.clip(c0, 0, W - P))
        AZ.append(az[:, r0:r0 + P, c0:c0 + P]); REF.append(ref[:, r0:r0 + P, c0:c0 + P])
        BEAM.append(beam[r0:r0 + P, c0:c0 + P]); ORIG.append((r0, c0)); KIND.append(kind); FRAME.append(fi)
        inside = [(r - r0, c - c0, k) for k, (r, c) in enumerate(lab_rc) if 0 <= r - r0 < P and 0 <= c - c0 < P]
        PTS.append(inside)

    for fi in range(len(day)):
        lab = day.labels_at(fi)
        az, ref = stack_raw(day, fi)
        lab_rc = lab[["row", "col"]].to_numpy(float) if len(lab) else np.zeros((0, 2))
        # positives: one per supercell
        if len(lab):
            for _, g in lab.groupby(["file", "supercell"]):
                rc = g[["row", "col"]].to_numpy(float).mean(0)
                off = rng.integers(-JITTER, JITTER + 1, 2)
                add(az, ref, rc[0] - P / 2 + off[0], rc[1] - P / 2 + off[1], 1, fi, lab_rc)
        # hard negatives: strong mid-level AzShear far from all tracked mesos, inside the Southeast
        a36 = az[0].astype(np.int16)
        cand = (a36 >= 5) & se
        if cand.any():
            rr, cc = np.where(cand)
            if len(lab_rc):
                # distance to nearest label, coarse check via subsampling for speed
                keep = np.ones(len(rr), bool)
                for r, c in lab_rc:
                    keep &= (rr - r) ** 2 + (cc - c) ** 2 > far_px ** 2
                rr, cc = rr[keep], cc[keep]
            order = np.argsort(-a36[rr, cc])
            picked = []
            for k in order:
                if len(picked) >= HARD:
                    break
                if all((rr[k] - pr) ** 2 + (cc[k] - pc) ** 2 > (P // 2) ** 2 for pr, pc in picked):
                    picked.append((rr[k], cc[k]))
            for r, c in picked:
                off = rng.integers(-JITTER, JITTER + 1, 2)
                add(az, ref, r - P / 2 + off[0], c - P / 2 + off[1], 2, fi, lab_rc)
        # random echo patch inside the Southeast
        dbz30 = (ref[0] >= (30 + 32) * 2 + 1) & se
        if dbz30.any():
            rr, cc = np.where(dbz30)
            k = rng.integers(len(rr))
            add(az, ref, rr[k] - P / 2, cc[k] - P / 2, 3, fi, lab_rc)

    n = len(AZ)
    npts = np.array([len(p) for p in PTS])
    pts = np.array([(i, r, c, k) for i, p in enumerate(PTS) for r, c, k in p], dtype=np.float32).reshape(-1, 4)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"patches_{day.file}.npz", az=np.stack(AZ), ref=np.stack(REF), beam=np.stack(BEAM),
                        orig=np.array(ORIG, np.int32), kind=np.array(KIND, np.int8), frame=np.array(FRAME, np.int32),
                        npts=npts, pts=pts, day=day.file)
    print(f"{day.file}: {len(day)} frames -> {n} patches (pos {sum(k == 1 for k in KIND)}, hard {sum(k == 2 for k in KIND)}, "
          f"random {sum(k == 3 for k in KIND)}); {int(npts.sum())} meso centers")


if __name__ == "__main__":
    main()
