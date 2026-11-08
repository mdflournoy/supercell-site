#!/usr/bin/env python3
"""Stage 5: assemble the comparison report page from one or more report.json files.

  python ml/build_report.py ml/results/v2 ml/results/v1 --out ml/results/report.html

The first folder's baseline is used for the AzShear-threshold baseline. Each folder's model becomes
one selectable "run" on the page.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import ML_DIR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    reps = [(Path(f).name, json.loads((Path(f) / "report.json").read_text())) for f in a.folders]
    runs, days = [], {}
    for i, (key, r) in enumerate(reps):
        methods = [("model", key, r["label"] or key)]
        if i == 0:
            methods.append(("baseline", "baseline", "AzShear threshold"))
        for m, rk, label in methods:
            runs.append({"key": rk, "label": label, "settings": r["settings"][m],
                         "pooled": {s: r["pooled"][s][m] for s in r["pooled"]},
                         "days": {e["day"]: {"scores": e["scores"][m], "pred": e["pred"][m]} for e in r["days"]}})
        for e in r["days"]:
            d = days.setdefault(e["day"], {"day": e["day"], "set": e["set"], "truth": {}})
            for t in e["truth"]:
                tt = d["truth"].setdefault(t["id"], {k: t[k] for k in ("id", "sc", "tornadic", "scored", "pts")} | {"found": {}})
                tt["found"][key] = t["found"]["model"]
                if i == 0:
                    tt["found"]["baseline"] = t["found"]["baseline"]
    data = {"lag_s": reps[0][1]["lag_s"], "states": reps[0][1]["states"], "runs": runs,
            "days": [dict(d, truth=list(d["truth"].values())) for _, d in sorted(days.items())]}
    tpl = (ML_DIR / "report_template.html").read_text()
    Path(a.out).write_text(tpl.replace("/*DATA*/null", json.dumps(data, separators=(",", ":"))))
    print(a.out, round(Path(a.out).stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
