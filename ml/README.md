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

## Files

| file | what |
|---|---|
| `common.py` | grid, Southeast mask, radar distance / beam height, track reading and interpolation |
| `diag_tracks.py` | MRMS AzShear at tracked positions vs. distance to the nearest WSR-88D → `diagnostics/` |
| `build_frames.py` | one event day → `frames/*.npz`, `frames.csv`, `labels.csv`, `offsets.csv` (run by `.github/workflows/ml-frames.yml`) |
| `static/nexrad_sites.csv` | CONUS WSR-88D locations (from Py-ART) |
| `static/se_states.geojson` | the seven Southeast states (coarse outlines) |

### Frame encoding

Frames are stored as release assets of the
[`ml-frames-v1`](../../releases/tag/ml-frames-v1) pre-release, one `.tar` per event day.

- `az02`, `az36`: int8 AzShear in 10⁻³ s⁻¹ (MRMS stores whole numbers); −128 = no data.
- `ref`: uint8 composite reflectivity, (dBZ + 32) × 2 + 1; 0 = no data.
