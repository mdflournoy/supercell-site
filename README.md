# Observed Database of Supercells

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

Everything except the map shows **whole supercells active at any point in the window**.
The map can clip tracks to the window ("Clip to window").
