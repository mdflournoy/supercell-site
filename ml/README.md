# Automatic mesocyclone tracking from MRMS (experiment)

Goal: a model that reads MRMS composite reflectivity and 0–2 km / 3–6 km AzShear and outputs
**mid-level mesocyclone tracks, grouped into supercells**, like the hand-made tracks in `tracks/`.
It is trained on the 2022 events and tested on the 2023 events, which it never sees in training.

## Plan

1. **Data** *(this stage)*: MRMS frames every ~4 min for each event day, cropped to a Southeast box
   (28–40.5° N, 97.5–77.5° W). The MRMS grids are 0.005° (~0.5 km) for AzShear and 0.01° for
   reflectivity. Each frame comes with labels: every tracked meso, interpolated to the frame time.
   Also includes diagnostics of how tracked mesos appear in MRMS.
2. **Model**: a small fully convolutional network, trained on CPU in GitHub Actions. It outputs a
   heat map of mid-level meso centers from the current and previous frames plus a radar-coverage
   layer (lowest-beam height). Training uses 22 of the 2022 days; the other 3 are held out for tuning.
3. **Tracking**: link detected centers frame to frame into meso tracks. Group meso tracks into
   supercells (shared reflectivity core / new meso forming next to the previous one), with written
   rules for meso numbering.
4. **Test on 2023**: score against the 2023 tracks, and against a simple AzShear-threshold baseline.
   Measures: mesos and supercells found, missed and falsely detected; start/end timing; tracks that
   break apart or swap identities; mesos per supercell.
5. **Report**: predicted vs. actual tracks for each 2023 day.

## Truth region

Every supercell whose track entered **AR, LA, MS, AL, GA, TN or KY** (`static/se_states.geojson`) was
tracked. So inside those states, anything without a track is "no mesocyclone". Outside them, the
truth is known only near tracks. Detections outside the region are ignored unless they match a
track.

Each track file's analysis window runs from 1 h before its first tracked scan to 1 h after its
last.

## Stage 1 findings (`diagnostics/`)

- **MRMS sees almost all tracked mesos.** 3–6 km AzShear has a value at 99% of tracked positions,
  and coverage barely changes out to 200 km from the nearest WSR-88D. Signal strength drops with
  range: 43% of positions within 50 km reach ≥ 5 × 10⁻³ s⁻¹, vs. about 20% beyond 50 km
  (`azshear_vs_range.png`).
- **Time offset of about 6 minutes.** An MRMS AzShear frame stamped T shows each meso where the
  Level II track puts it at about T − 6 min. Mean peak AzShear along the tracks doubles when this
  offset is applied, in both 0–2 and 3–6 km and on both a 2022 and a 2023 day (`lag.png`,
  `lag.json`). The model therefore labels each frame with track positions 350 s earlier, and
  reports predicted tracks at frame time − 350 s (`common.MRMS_LAG_S`). Without the correction, the
  AzShear maximum sits a median ~7 km behind the tracked position (`offsets.png`).
- Built 5,950 frames over 30 days (4.0 GB) with no failed downloads.

## Files

| file | what |
|---|---|
| `common.py` | grid, Southeast mask, radar distance / beam height, track reading and interpolation |
| `diag_tracks.py` | MRMS AzShear at tracked positions vs. distance to the nearest WSR-88D → `diagnostics/` |
| `diag_lag.py` | MRMS-vs-track time offset → `diagnostics/lag.*` |
| `stage1_report.py` | per-day frame/label counts and AzShear-max offsets → `diagnostics/` |
| `frames.py`, `make_patches.py`, `model.py`, `train.py` | stage 2: patches, MesoNet (small U-Net), CPU training (`.github/workflows/ml-train.yml`) |
| `detect.py`, `track.py`, `score_tracks.py`, `run_tracking.py` | stages 3–4: detection (`.github/workflows/ml-detect.yml`), tracking, tuning, scoring |
| `build_frames.py` | one event day → `frames/*.npz`, `frames.csv`, `labels.csv`, `offsets.csv` (run by `.github/workflows/ml-frames.yml`) |
| `static/nexrad_sites.csv` | CONUS WSR-88D locations (from Py-ART) |
| `static/se_states.geojson` | the seven Southeast states (coarse outlines) |

### Frame encoding

Frames are stored as release assets of the
[`ml-frames-v1`](../../releases/tag/ml-frames-v1) pre-release, one `.tar` per event day.

- `az02`, `az36`: int8 AzShear in 10⁻³ s⁻¹ (MRMS stores whole numbers); −128 = no data.
- `ref`: uint8 composite reflectivity, (dBZ + 32) × 2 + 1; 0 = no data.
