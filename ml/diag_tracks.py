#!/usr/bin/env python3
"""Stage 1 diagnostic: how well do the hand-tracked (Level II) mesocyclones show up in MRMS AzShear?

Uses tracks-AzShear/ (peak MergedAzShear within 2 km of each tracked position at every MRMS time)
and the distance / lowest-beam height to the nearest WSR-88D. Writes to ml/diagnostics/:
  azshear_vs_range.csv   one row per meso per MRMS time, with radar distance and beam height
  azshear_by_range.csv   binned summary (fraction missing, median, fraction >= thresholds)
  meso_peaks.csv         per meso: lifetime-max AzShear, median radar distance
  azshear_vs_range.png   figure
  summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import REPO, ML_DIR, nearest_radar, in_southeast, read_tracks

OUT = ML_DIR / "diagnostics"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rows = []
    for p in sorted((REPO / "tracks-AzShear").glob("2*.csv")):
        d = pd.read_csv(p, dtype={"supercell_id": str, "meso_id": str})
        d["file"] = p.stem
        rows.append(d)
    az = pd.concat(rows, ignore_index=True)
    az["label"] = az["supercell_id"].str.strip() + az["meso_id"].str.strip()
    dist, beam, site = nearest_radar(az["lat"], az["lon"])
    az["radar_km"], az["beam_m"], az["radar"] = dist.round(1), beam.round(0), site
    az["in_se"] = in_southeast(az["lat"].to_numpy(), az["lon"].to_numpy())
    tr = read_tracks()
    tor = tr.groupby(["file", "label"])["tornadic"].first()
    az["tornadic"] = [bool(tor.get((f, l), False)) for f, l in zip(az["file"], az["label"])]
    az["year"] = az["file"].str[:4]
    az.to_csv(OUT / "azshear_vs_range.csv", index=False)

    bins = [0, 50, 100, 150, 200, 300]
    az["range_bin"] = pd.cut(az["radar_km"], bins, right=False, labels=[f"{a}-{b} km" for a, b in zip(bins, bins[1:])])
    def summ(g):
        v = g["azshear_3_6km"]
        v2 = g["azshear_0_2km"]
        return pd.Series({
            "n": len(g), "beam_height_m_median": g["beam_m"].median(),
            "mid_missing_frac": v.isna().mean(), "mid_median": v.median(),
            "mid_ge3_frac": (v >= 3).mean(), "mid_ge5_frac": (v >= 5).mean(), "mid_ge8_frac": (v >= 8).mean(),
            "low_missing_frac": v2.isna().mean(), "low_median": v2.median(), "low_ge5_frac": (v2 >= 5).mean(),
        })
    by = az.groupby("range_bin", observed=True).apply(summ, include_groups=False).round(3)
    by.to_csv(OUT / "azshear_by_range.csv")

    mp = az.groupby(["file", "label"]).agg(year=("year", "first"), tornadic=("tornadic", "first"), n=("label", "size"),
                                         mid_max=("azshear_3_6km", "max"), low_max=("azshear_0_2km", "max"),
                                         mid_median=("azshear_3_6km", "median"), radar_km=("radar_km", "median")).reset_index()
    mp.to_csv(OUT / "meso_peaks.csv", index=False)

    s = {
        "meso_times": int(len(az)), "mesos": int(len(mp)),
        "mid_missing_frac": round(float(az["azshear_3_6km"].isna().mean()), 3),
        "mid_ge3_frac": round(float((az["azshear_3_6km"] >= 3).mean()), 3),
        "mid_ge5_frac": round(float((az["azshear_3_6km"] >= 5).mean()), 3),
        "mesos_mid_max_lt3": int((mp["mid_max"].fillna(0) < 3).sum()),
        "mesos_mid_max_lt5": int((mp["mid_max"].fillna(0) < 5).sum()),
        "median_mid_by_tornadic": az.groupby("tornadic")["azshear_3_6km"].median().to_dict(),
        "by_range": json.loads(by.to_json(orient="index")),
    }
    (OUT / "summary.json").write_text(json.dumps(s, indent=1, default=str))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    a = ax[0]
    x = az["radar_km"]; y = az["azshear_3_6km"]
    a.scatter(x, y.fillna(-1) + np.random.default_rng(0).uniform(-0.35, 0.35, len(y)), s=2, alpha=0.15, color="0.3")
    med = az.groupby("range_bin", observed=True)["azshear_3_6km"].median()
    mids = [(a_ + b_) / 2 for a_, b_ in zip(bins, bins[1:])][:len(med)]
    a.plot(mids, med.values, "o-", color="#e34948", lw=2, label="median")
    a.axhline(0, color="0.7", lw=0.8)
    a.set_xlabel("Distance to nearest WSR-88D (km)"); a.set_ylabel("Peak 3–6 km AzShear within 2 km (10⁻³ s⁻¹)")
    a.set_title("Mid-level AzShear at tracked meso positions\n(−1 = no MRMS value)"); a.legend()
    a = ax[1]
    cats = by.index.astype(str)
    a.bar(cats, by["mid_missing_frac"], color="0.6", label="no value")
    a.bar(cats, by["mid_ge5_frac"], bottom=by["mid_missing_frac"], color="#2a78d6", alpha=0.8, label="≥ 5")
    a.set_ylim(0, 1); a.set_ylabel("Fraction of meso times"); a.set_title("Missing vs. ≥ 5 ×10⁻³ s⁻¹ by range"); a.legend()
    a.tick_params(axis="x", rotation=20)
    for i, n in enumerate(by["n"]):
        a.text(i, 0.97, f"n={int(n)}", ha="center", va="top", fontsize=8)
    a = ax[2]
    for torn, c in ((False, "#2a78d6"), (True, "#e34948")):
        v = mp.loc[mp["tornadic"] == torn, "mid_max"].fillna(0)
        a.hist(v, bins=np.arange(0, 41, 2), alpha=0.6, color=c, label=f"{'tornadic' if torn else 'nontornadic'} (n={len(v)})")
    a.set_xlabel("Lifetime max 3–6 km AzShear (10⁻³ s⁻¹)"); a.set_ylabel("Mesocyclones"); a.set_title("Per-meso peak"); a.legend()
    fig.savefig(OUT / "azshear_vs_range.png", dpi=130)
    print(json.dumps({k: v for k, v in s.items() if k != "by_range"}, indent=1, default=str))
    print(by.to_string())


if __name__ == "__main__":
    main()
