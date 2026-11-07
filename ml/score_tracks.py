"""Stage 4: compare predicted meso/supercell tracks with the hand-made tracks for one event day.

Frame by frame, predicted meso positions are matched one-to-one with tracked meso positions within
MATCH_KM. From those frame matches:

meso level
  truth meso found   >= 50% of its frames matched (by any predicted meso)
  predicted meso ok  >= 50% of its frames matched; otherwise a false alarm (counted only if most of
                     its points are inside the Southeast truth region)
  fragmentation      number of predicted mesos that matched a found truth meso
  start/end error    of the predicted meso that matched it most (minutes; + = later)
supercell level
  truth supercell found   any of its mesos found
  predicted supercell ok  any of its mesos ok
  grouping                for truth supercells with >= 2 found mesos: share of those mesos whose main
                          predicted meso is in the supercell's most common predicted supercell
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from common import in_southeast

MATCH_KM = 5.0


def _km(a_lat, a_lon, b_lat, b_lon):
    return np.hypot((a_lat[:, None] - b_lat[None, :]) * 111.2,
                    (a_lon[:, None] - b_lon[None, :]) * 111.2 * np.cos(np.radians(b_lat[None, :])))


def frame_matches(pred: pd.DataFrame, truth: pd.DataFrame, max_km: float = MATCH_KM) -> pd.DataFrame:
    """pred: t, lat, lon, pkey; truth: t, lat, lon, tkey. Returns rows (t, pkey, tkey, km)."""
    out = []
    pg = {t: g for t, g in pred.groupby("t")}
    for t, tg in truth.groupby("t"):
        g = pg.get(t)
        if g is None or not len(g):
            continue
        d = _km(g.lat.to_numpy(), g.lon.to_numpy(), tg.lat.to_numpy(), tg.lon.to_numpy())
        used_p, used_t = set(), set()
        for k in np.argsort(d, axis=None):
            i, j = np.unravel_index(k, d.shape)
            if d[i, j] > max_km:
                break
            if i in used_p or j in used_t:
                continue
            used_p.add(i); used_t.add(j)
            out.append((t, g.pkey.iloc[i], tg.tkey.iloc[j], float(d[i, j])))
    return pd.DataFrame(out, columns=["t", "pkey", "tkey", "km"])


def score_day(pred: pd.DataFrame, truth: pd.DataFrame) -> dict:
    """pred: supercell_id, meso_id, t, lat, lon (predicted, on frame times).
    truth: labels at the same frames: file, supercell, label, t, lat, lon."""
    pred = pred.copy()
    truth = truth.copy()
    pred["pkey"] = pred.supercell_id + pred.meso_id.astype(str)
    truth["tkey"] = truth.file.astype(str) + ":" + truth.label
    truth["tsc"] = truth.file.astype(str) + ":" + truth.supercell
    m = frame_matches(pred, truth) if len(pred) else pd.DataFrame(columns=["t", "pkey", "tkey", "km"])

    t_frames = truth.groupby("tkey").t.nunique()
    t_hit = m.groupby("tkey").t.nunique().reindex(t_frames.index, fill_value=0)
    found = (t_hit / t_frames) >= 0.5

    p_frames = pred.groupby("pkey").t.nunique() if len(pred) else pd.Series(dtype=float)
    p_hit = m.groupby("pkey").t.nunique().reindex(p_frames.index, fill_value=0)
    p_ok = (p_hit / p_frames) >= 0.5
    if len(pred):
        pred["in_se"] = in_southeast(pred.lat.to_numpy(), pred.lon.to_numpy())
        p_se = pred.groupby("pkey").in_se.mean() > 0.5
    else:
        p_se = pd.Series(dtype=bool)
    p_false = (~p_ok) & p_se.reindex(p_ok.index, fill_value=False)

    # main predicted meso per truth meso; timing and fragmentation
    frag, start_err, end_err, main_of = [], [], [], {}
    pt = pred.groupby("pkey").t.agg(["min", "max"]) if len(pred) else None
    tt = truth.groupby("tkey").t.agg(["min", "max"])
    for tk, g in m.groupby("tkey"):
        if not found.get(tk, False):
            continue
        c = Counter(g.pkey)
        main = c.most_common(1)[0][0]
        main_of[tk] = main
        frag.append(len(c))
        start_err.append((pt.loc[main, "min"] - tt.loc[tk, "min"]) / 60)
        end_err.append((pt.loc[main, "max"] - tt.loc[tk, "max"]) / 60)

    # supercells
    t_sc_of = truth.groupby("tkey").tsc.first()
    t_sc_found = found.groupby(t_sc_of.reindex(found.index)).any()
    p_sc_of = pred.groupby("pkey").supercell_id.first() if len(pred) else pd.Series(dtype=str)
    p_sc_ok = p_ok.groupby(p_sc_of.reindex(p_ok.index)).any() if len(pred) else pd.Series(dtype=bool)
    p_sc_se = p_se.groupby(p_sc_of.reindex(p_se.index)).any() if len(pred) else pd.Series(dtype=bool)
    p_sc_false = (~p_sc_ok) & p_sc_se.reindex(p_sc_ok.index, fill_value=False)
    grouped, group_tot = 0, 0
    by_tsc = defaultdict(list)
    for tk, pk in main_of.items():
        by_tsc[t_sc_of[tk]].append(p_sc_of[pk])
    for tsc, scs in by_tsc.items():
        if len(scs) >= 2:
            grouped += Counter(scs).most_common(1)[0][1]
            group_tot += len(scs)

    n_t, n_f = int(len(found)), int(found.sum())
    n_p_counted = int((p_ok | p_false).sum())
    res = {
        "truth_mesos": n_t, "found_mesos": n_f, "missed_mesos": n_t - n_f,
        "pred_mesos": int(len(p_frames)), "pred_mesos_ok": int(p_ok.sum()), "false_mesos": int(p_false.sum()),
        "meso_pod": n_f / max(1, n_t), "meso_far": int(p_false.sum()) / max(1, n_p_counted),
        "truth_supercells": int(len(t_sc_found)), "found_supercells": int(t_sc_found.sum()),
        "pred_supercells": int(p_sc_of.nunique()) if len(pred) else 0, "false_supercells": int(p_sc_false.sum()),
        "sc_pod": float(t_sc_found.mean()) if len(t_sc_found) else 0.0,
        "sc_far": int(p_sc_false.sum()) / max(1, int((p_sc_ok | p_sc_false).sum())),
        "frag_mean": float(np.mean(frag)) if frag else np.nan,
        "start_err_med_min": float(np.median(start_err)) if start_err else np.nan,
        "end_err_med_min": float(np.median(end_err)) if end_err else np.nan,
        "pos_err_med_km": float(m.km.median()) if len(m) else np.nan,
        "frame_pod": len(m) / max(1, len(truth)),
        "grouping": grouped / group_tot if group_tot else np.nan, "grouping_n": group_tot,
        "truth_mesos_per_sc": n_t / max(1, len(t_sc_found)),
        "pred_mesos_per_sc": int(p_ok.sum()) / max(1, int(p_sc_ok.sum())) if len(pred) else np.nan,
    }
    res["meso_csi"] = n_f / max(1, n_t + int(p_false.sum()))
    res["sc_csi"] = res["found_supercells"] / max(1, res["truth_supercells"] + res["false_supercells"])
    return res
