#!/usr/bin/env python3
"""Stage 1 diagnostic: time offset between MRMS AzShear frames and the Level II track times.

For tracked mesos in a frame at nominal MRMS time T, take the peak AzShear within 2 km of where the
track puts the meso at T + tau, for tau = -12 … +12 min. The tau with the highest average AzShear is
the offset between the two time bases. Writes ml/diagnostics/lag.json and lag.png.

  python ml/diag_lag.py DAYDIR [DAYDIR ...]     (extracted frame folders from the ml-frames-v1 release)
  python ml/diag_lag.py --json OUT.json DAYDIR  (one day, JSON only; used by ml-lag.yml)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import DX_AZ, ML_DIR, TrackIndex, latlon_to_rc, read_tracks


def main(dirs: list[str], json_out: str | None = None) -> None:
    tr = read_tracks()
    idx = TrackIndex(tr)
    taus = np.arange(-12, 12.5, 0.5) * 60
    rad = int(2 / (DX_AZ * 111.2)) + 1
    acc = {k: {float(t): [] for t in taus} for k in ("az36", "az02")}
    for d in dirs:
        d = Path(d)
        fr = pd.read_csv(d / "frames.csv").sort_values("t")
        for name, t in zip(fr.name.iloc[::2], fr.t.iloc[::2]):
            base = idx.at(t)
            if not len(base):
                continue
            with np.load(d / "frames" / f"{name}.npz") as z:
                fields = {k: z[k].astype(np.float32) for k in acc}
            for f in fields.values():
                f[f == -128] = np.nan
            keys = set(zip(base.file, base.label))
            for tau in taus:
                p = idx.at(t + tau)
                p = p[[k in keys for k in zip(p.file, p.label)]]
                if not len(p):
                    continue
                rr, cc = latlon_to_rc(p.lat.to_numpy(), p.lon.to_numpy())
                for r, c in zip(np.round(rr).astype(int), np.round(cc).astype(int)):
                    for k, f in fields.items():
                        w = f[max(0, r - rad):r + rad + 1, max(0, c - rad):c + rad + 1]
                        acc[k][float(tau)].append(np.nanmax(w) if np.isfinite(w).any() else np.nan)
    out = {"days": [Path(d).name for d in dirs], "tau_min": [t / 60 for t in taus]}
    for k in acc:
        m = [float(np.nanmean(acc[k][float(t)])) for t in taus]
        out[f"{k}_mean"] = m
        i = int(np.argmax(m))
        out[f"{k}_best_tau_min"] = taus[i] / 60
    out["n_per_tau"] = [len(acc["az36"][float(t)]) for t in taus]
    if json_out:
        Path(json_out).write_text(json.dumps(out, indent=1))
        print({k: v for k, v in out.items() if k.endswith("best_tau_min")})
        return
    (ML_DIR / "diagnostics" / "lag.json").write_text(json.dumps(out, indent=1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(out["tau_min"], out["az36_mean"], "o-", ms=3, label="3–6 km AzShear", color="#e34948")
    ax.plot(out["tau_min"], out["az02_mean"], "o-", ms=3, label="0–2 km AzShear", color="#2a78d6")
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_xlabel("Track time offset τ (min): track position at MRMS time + τ")
    ax.set_ylabel("Mean peak AzShear within 2 km (10⁻³ s⁻¹)")
    ax.set_title(f"MRMS vs. track timing ({', '.join(out['days'])})")
    ax.legend()
    fig.savefig(ML_DIR / "diagnostics" / "lag.png", dpi=130)
    print({k: v for k, v in out.items() if k.endswith("best_tau_min")})


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["--json"]:
        main(args[2:], args[1])
    else:
        main(args)
