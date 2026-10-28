# Observational Database of Supercells

A single-page dashboard of supercell mesocyclone tracks and their tornadoes, rebuilt
every night at midnight US Central by GitHub Actions and served on GitHub Pages.

**What runs each night**

1. `scripts/fetch_drive.py` downloads every `.csv` in the Google Drive folder (subfolders included).
2. `scripts/build_data.py` parses each event file and resolves each linked NCEI event ID
   to the **whole tornado** using the 1950–2024 tornado database CSV in the same folder
   (any CSV with `oneTorID` and `stormEventsReportID` columns): every segment sharing the
   linked segment's `oneTorID` is plotted, and totals (length, max EF, casualties, start/end)
   are computed across segments. Event IDs not in the database (e.g. 2025+) fall back to a
   single segment from NCEI Storm Events. Writes `site/data.json`.
3. The `site/` folder is deployed to GitHub Pages.

## One-time setup (≈10 minutes)

### 1. Give GitHub read access to the Drive folder

Pick **one** option.

**Option A: service account (recommended; the folder stays private)**

1. Open <https://console.cloud.google.com/>. Create a project (any name), or pick an existing one.
2. Go to **APIs & Services → Library**, search for **Google Drive API**, and click **Enable**.
3. Go to **IAM & Admin → Service Accounts → Create service account**. Name it, for example,
   `supercell-site`. Skip the optional role steps.
4. Open the new account. Go to **Keys → Add key → Create new key → JSON**. A `.json` file downloads.
5. In Google Drive, **share the folder** with the service account's email
   (`supercell-site@<project>.iam.gserviceaccount.com`) as a **Viewer**.
6. In GitHub, open the repo and go to **Settings → Secrets and variables → Actions → New repository secret**.
   - Name: `GDRIVE_SERVICE_ACCOUNT_JSON`
   - Value: paste the **entire contents** of the JSON key file.

**Option B: API key (the folder must be shared "Anyone with the link")**

1. Do steps 1–2 above. Then go to **APIs & Services → Credentials → Create credentials → API key**.
   Restricting the key to the Drive API is a good idea.
2. Set the Drive folder's sharing to **Anyone with the link → Viewer**.
3. Add the repository secret `GDRIVE_API_KEY` with the key as its value.

### 2. Tell the workflow which folder to read

Go to **Settings → Secrets and variables → Actions → Variables tab → New repository variable**.
- Name: -------
- Value: -------

### 3. Turn on GitHub Pages

Go to **Settings → Pages → Build and deployment → Source**, and choose **GitHub Actions**.

### 4. Run it once

Go to **Actions → Nightly rebuild → Run workflow**. When it finishes, the site is at
`https://mdflournoy.github.io/supercell-site/`.

After that it rebuilds at midnight Central every night. Any push to `main` that changes
the site or scripts also rebuilds it. GitHub sometimes starts scheduled runs a few
minutes late.

## CSV format expected

One file per day (the date is read from the file name, e.g. `20230331.csv`):

```
supercell_id,meso_id,time_utc,lat,lon,dissipation,primary_sup
A,1,2023-03-31T17:45:42+00:00,33.92533,-94.29710,dissipate,
...

Tornadoes
A1,1089126        <- supercell A, meso 1, NCEI Storm Events EVENT_ID
```

- Links are counted as **whole tornadoes**: two links that are segments of the same tornado
  (same `oneTorID`) count once. A tornado linked to two mesos makes both tornadic.
- A **mesocyclone** is tornadic if at least one tornado is linked to it.
  A **supercell** is tornadic if any of its mesocyclones is.
- **Time to first tornado** is the earliest linked tornado's start (first segment's
  `beginUnix`) minus the supercell's first tracked scan.
- NCEI times are in local standard time and get converted to UTC using each
  record's `CZ_TIMEZONE`.
- NCEI usually publishes events about 2–3 months after they happen. Until then, a linked
  tornado still counts toward the tornadic totals, but it has no map path or timing. The
  page shows a notice listing how many tornadoes are in this state.
- Any problem in a file (an unreadable row, or a tornado link that doesn't match a meso)
  is printed in the Action log and shown in the browser console.

## Run it locally

```bash
pip install -r requirements.txt
# put some CSVs in data/raw/ (or use fetch_drive.py with the env vars above)
python scripts/build_data.py --raw data/raw --out site/data.json
cd site && python -m http.server 8000      # open http://localhost:8000
```

## Dashboard features

- Pan/zoom map (Leaflet + OpenFreeMap vector basemap with state/county lines; sharp on high-DPI screens). Nontornadic mesos are blue, tornadic mesos red,
  and NCEI tornado paths are drawn in black with width scaled by EF. Hover a track to see
  its whole supercell; click any track or path for details and a link to the NCEI event.
- Time window: a dual-handle slider, exact UTC start/end inputs, "Jump to a day", a year
  filter, and "All days". The window is saved in the URL, so you can share a link to a
  specific window.
- KPIs: supercells, mesocyclones, linked tornadoes (and how many are EF2+), and the median
  time to first tornado.
- Donut charts: tornadic vs. nontornadic supercells, mesocyclones, and mesocyclones
  within tornadic supercells.
- Histograms: mesocyclones per supercell, time from track start to first tornado, and
  mesocyclone lifetime.
- Bar charts of supercell dissipation mode (by tornadic status) and of tornado max-EF ratings.
- Summary boxes also include total mesocyclone track length and total mesocyclone duration.
- Tornado paths are off by default (toggle above the map). Hovering any track or tornado
  widens every track and tornado belonging to that supercell.
- A sortable table of days; click a row to jump to that day.
- Light and dark themes (follows the system setting; the button toggles).
- **AzShear** button (upper right of the map): colors each mesocyclone track by MRMS peak AzShear
  from `tracks-AzShear/` — choose Low-level (0–2 km AGL) or Mid-level (3–6 km AGL). Track outlines
  stay red/blue (tornadic/nontornadic); the fill uses the plasma colormap on a fixed 0–15
  (×10⁻³ s⁻¹) scale, gray where MRMS has no data. Off by default.

- **Environments** checkbox (above the map, off by default): draws each HRRR inflow slice from
  `env-params/` — a 40 km, 90° wedge centered on the inflow direction at the meso's position at each top
  of the hour, outlined and lightly filled in its track's color. Hovering a slice shows its inflow
  sounding as a Skew-T and a 0–10 km hodograph (observed and Bunkers right-mover motions marked), with
  surface T/Td, MLCAPE/MLCIN, MLLCL and effective SRH. The page loads these from `site/env.json`
  only when the box is first checked. A dot marks each slice's apex on its parent track; hovering a slice
  or dot also highlights the parent supercell.
- **Meso IDs on the map** (e.g. `A1`, in the track's color, at the start of each track) whenever the
  window spans about a day or less — e.g. after picking a day.
- **Supercells** menu (left of "Jump to a day"): appears once a day is selected (dropdown or Days table)
  and lists that day's supercells. Click a letter to show only that supercell on the map; tick or untick
  boxes to add or remove others; "Show all", another day, or "All days" brings everything back. It
  disappears if the window moves off that day. This filters the map only.

Everything except the map shows **whole supercells active at any point in the window**.
The map shows only the parts of each track inside the window.

## Renumbered track files (`tracks/`)

The original CSVs in Google Drive are never modified. Each build writes a renumbered copy of
every day file to [`tracks/`](tracks/) and commits it to this repo; **the website uses these
copies**:

- supercells are relettered **A, B, C, … Z, AA, AB, …** in order of their start time (earliest scan
  of any of their mesocyclones);
- mesocyclones within each supercell are renumbered **1, 2, 3, …** in order of their own start time;
- the `Tornadoes` link keys and the `primary_sup` merge column are remapped to the new IDs.

[`tracks/id_map.csv`](tracks/id_map.csv) lists the old and new IDs of every mesocyclone.

## MRMS AzShear series (`tracks-AzShear/`)

For each day, [`tracks-AzShear/YYYYMMDD.csv`](tracks-AzShear/) has one row for every mesocyclone
at every MRMS time (~2 min) within its tracked lifetime (first to last scan), using the
renumbered IDs from `tracks/` (which is not modified):

| column | meaning |
|---|---|
| `supercell_id`, `meso_id` | renumbered IDs (same as `tracks/`) |
| `time_utc` | MRMS valid time of the 0–2 km AzShear file |
| `lat`, `lon` | meso position linearly interpolated along its track to that time |
| `azshear_0_2km` | peak MRMS MergedAzShear 0–2 km AGL within 2 km of that position, in **10⁻³ s⁻¹** as stored by MRMS (e.g. `12` = 0.012 s⁻¹) |
| `azshear_3_6km` | same for 3–6 km AGL, from the 3–6 km file nearest in time (within 90 s), position interpolated to that file's time |

Blank values mean MRMS had no valid data in the circle (or the file is missing). MRMS encodes missing/no coverage as 0, so zeros are treated as missing. A mesocyclone
tracked on only one scan gets the single MRMS time nearest to it (within 5 min).

Source: `s3://noaa-mrms-pds/CONUS/MergedAzShear_0-2kmAGL_00.50/` and `..._3-6kmAGL_00.50/`.
Each MRMS file is processed once; results are cached in `azshear-cache/cache.csv`, so nightly
runs only fetch what new or edited tracks need. A run spends at most ~4 hours on MRMS and
resumes where it left off on the next run; `azshear-cache/status.json` summarizes the last run.

## HRRR near-storm environments (`env-profiles/`, `env-params/`)

Following the HRRR-Inflow method of Coffer, Parker, Coniglio & Homeyer (2025, *Wea. Forecasting*),
for every mesocyclone in `tracks/` at every top of the hour within its tracked lifetime (or the hour
before it formed, if it never spans one):

- position interpolated to HH:00; observed motion from track positions ±15 min around HH:00;
- HRRR 00-h analysis valid at HH:00 on **native hybrid levels** (`s3://noaa-hrrr-bdp-pds`,
  `wrfnatf00`), winds rotated from grid- to earth-relative;
- a 90° slice (±45°) centered on the 0–1 km storm-relative inflow direction (observed motion; Bunkers
  right-mover when no observed motion exists), within 40 km of the grid point nearest the storm;
  θ, qv, p, u, v and height AGL are averaged over the slice into one inflow sounding.

**`env-profiles/YYYYMMDD_<meso>_<HHMM>_profile.csv`** (event date in the name, even after 00 UTC):
line 1 = event date, valid time, meso ID, lat/lon; line 2 = header; then
`height_agl_m, pressure_hpa, temperature_c, dewpoint_c, u_ms, v_ms` from the lowest model level to 100 hPa.

**`env-params/YYYYMMDD.csv`**: one row per meso per hour (sorted A1, A2, …, then time) with the
approved parameter list — SB/ML/MU parcels (ML = lowest 100 hPa, MU searched in lowest 300 hPa),
0–3 km MLCAPE, ECAPE (ML parcel; Peters et al. 2023), effective inflow layer (Thompson et al. 2007),
DCAPE, lapse rates, moisture, ground-relative winds, bulk shear, and storm-relative quantities twice
(`_obs` = observed motion, `_bunk` = Bunkers right-mover): SRH, storm-relative flow, 0–500 m
streamwise/crosswise vorticity, critical angle, ECAPE, STP (fixed, effective, 500 m) and SCP, plus
yes/no flags for the paper's filters (MUCAPE = 0, elevated effective inflow base, EBWD < 12.5 m s⁻¹).
`date`/`time_utc` are the actual valid date/time. Results are cached in `env-cache/`, so each HRRR
hour is processed once.
