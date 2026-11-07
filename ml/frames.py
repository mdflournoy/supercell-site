"""Reading stage-1 frames and turning them into model inputs and targets."""
from __future__ import annotations

import tarfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from common import DX_AZ, box_shape, radar_fields, se_mask

NCH = 7  # az36, az02, ref, az36_prev, az02_prev, ref_prev, beam
PREV_MAX_DT = 330  # s; older previous frames are replaced by the current frame
SIGMA_PX = 3.0  # target Gaussian width (~1.7 km)
KM_PER_PX = DX_AZ * 111.2


class Day:
    """One event day extracted from its release tar (a folder with frames/, frames.csv, labels.csv)."""

    def __init__(self, folder: Path | str):
        self.dir = Path(folder)
        self.file = self.dir.name
        self.frames = pd.read_csv(self.dir / "frames.csv").sort_values("t").reset_index(drop=True)
        lab = pd.read_csv(self.dir / "labels.csv", dtype={"file": str, "supercell": str, "meso": str})
        self.labels = lab
        self.by_frame = {k: g for k, g in lab.groupby("frame")}

    def __len__(self):
        return len(self.frames)

    def raw(self, i: int):
        name = self.frames.name.iloc[i]
        with np.load(self.dir / "frames" / f"{name}.npz") as d:
            return d["az36"], d["az02"], d["ref"]

    def prev_index(self, i: int) -> int:
        if i == 0:
            return i
        return i - 1 if self.frames.t.iloc[i] - self.frames.t.iloc[i - 1] <= PREV_MAX_DT else i

    def labels_at(self, i: int) -> pd.DataFrame:
        return self.by_frame.get(self.frames.name.iloc[i], self.labels.iloc[:0])


def extract(tar_path: Path, dest: Path) -> Path:
    with tarfile.open(tar_path) as tf:
        tf.extractall(dest, filter="data")
    return dest / Path(tar_path).stem


@lru_cache(maxsize=1)
def static_fields():
    """(beam height as uint8 in 50 m steps, Southeast mask) on the AzShear grid."""
    _, beam = radar_fields(DX_AZ)
    beam_u8 = np.clip(np.nan_to_num(beam / 50.0, posinf=255), 0, 255).astype(np.uint8)
    return beam_u8, se_mask(DX_AZ)


def up2(a: np.ndarray) -> np.ndarray:
    """0.01° reflectivity → 0.005° grid (nearest)."""
    return np.repeat(np.repeat(a, 2, axis=0), 2, axis=1)


def stack_raw(day: Day, i: int):
    """Raw (int8 AzShear x4, uint8 reflectivity x2) full-domain stacks for frame i and its previous frame."""
    a36, a02, r = day.raw(i)
    p = day.prev_index(i)
    b36, b02, rp = day.raw(p) if p != i else (a36, a02, r)
    az = np.stack([a36, a02, b36, b02])          # int8
    ref = np.stack([up2(r), up2(rp)])            # uint8 on the AzShear grid
    return az, ref


def to_input(az: np.ndarray, ref: np.ndarray, beam: np.ndarray) -> np.ndarray:
    """Normalize to float32 channels (C, H, W). az: (4,...) int8; ref: (2,...) uint8; beam: uint8."""
    a = az.astype(np.float32)
    a[az == -128] = 0.0
    a = np.clip(a / 10.0, -3, 5)
    dbz = (ref.astype(np.float32) - 1) / 2 - 32
    dbz[ref == 0] = 0.0
    dbz = np.clip(dbz, 0, 75) / 60.0
    b = np.clip(beam.astype(np.float32) * 50.0 / 5000.0, 0, 2)
    return np.concatenate([a[[0]], a[[1]], dbz[[0]], a[[2]], a[[3]], dbz[[1]], b[None]], axis=0)


def heatmap(points_rc: np.ndarray, shape, sigma: float = SIGMA_PX) -> np.ndarray:
    """Max of Gaussians at (row, col) points; 1 at each center."""
    h = np.zeros(shape, np.float32)
    rad = int(3 * sigma) + 1
    for r, c in points_rc:
        r0, c0 = int(round(r)), int(round(c))
        j0, j1 = max(0, r0 - rad), min(shape[0], r0 + rad + 1)
        i0, i1 = max(0, c0 - rad), min(shape[1], c0 + rad + 1)
        if j0 >= j1 or i0 >= i1:
            continue
        jj, ii = np.mgrid[j0:j1, i0:i1]
        g = np.exp(-((jj - r) ** 2 + (ii - c) ** 2) / (2 * sigma ** 2))
        g[(jj == r0) & (ii == c0)] = 1.0
        np.maximum(h[j0:j1, i0:i1], g, out=h[j0:j1, i0:i1])
    return h


def valid_mask(points_rc: np.ndarray, se: np.ndarray, near_km: float = 25.0) -> np.ndarray:
    """Pixels where the truth is known: inside the Southeast states, or within near_km of a tracked meso."""
    m = se.copy()
    rad = int(near_km / KM_PER_PX)
    H, W = m.shape
    for r, c in points_rc:
        r0, c0 = int(round(r)), int(round(c))
        j0, j1 = max(0, r0 - rad), min(H, r0 + rad + 1)
        i0, i1 = max(0, c0 - rad), min(W, c0 + rad + 1)
        if j0 >= j1 or i0 >= i1:
            continue
        jj, ii = np.mgrid[j0:j1, i0:i1]
        m[j0:j1, i0:i1] |= (jj - r0) ** 2 + (ii - c0) ** 2 <= rad ** 2
    return m


__all__ = ["Day", "extract", "static_fields", "stack_raw", "to_input", "heatmap", "valid_mask", "NCH", "box_shape"]
