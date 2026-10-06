#!/usr/bin/env python3
"""Download every CSV in a Google Drive folder (recursively) into a local directory.

Two auth modes, picked from environment variables:

  GDRIVE_SERVICE_ACCOUNT_JSON  Full JSON key of a Google service account. Share the
                               Drive folder (Viewer) with the service account's email.
                               Works for private folders — recommended.
  GDRIVE_API_KEY               A Google Cloud API key with the Drive API enabled.
                               Only works if the folder is shared "Anyone with the link".

Usage:
  python scripts/fetch_drive.py --folder 1JaWWC-JEr4ArHjQmqZ4gYcud9CmYn6GC --out data/raw
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import requests

API = "https://www.googleapis.com/drive/v3/files"
FOLDER_MIME = "application/vnd.google-apps.folder"
SHEET_MIME = "application/vnd.google-apps.spreadsheet"


class Drive:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.params: dict[str, str] = {"supportsAllDrives": "true"}
        sa = os.environ.get("GDRIVE_SERVICE_ACCOUNT_JSON", "").strip()
        key = os.environ.get("GDRIVE_API_KEY", "").strip()
        if sa:
            from google.oauth2 import service_account
            from google.auth.transport.requests import Request

            creds = service_account.Credentials.from_service_account_info(
                json.loads(sa), scopes=["https://www.googleapis.com/auth/drive.readonly"]
            )
            creds.refresh(Request())
            self.session.headers["Authorization"] = f"Bearer {creds.token}"
            print(f"Auth: service account {creds.service_account_email}")
        elif key:
            self.params["key"] = key
            print("Auth: API key (folder must be shared 'Anyone with the link')")
        else:
            sys.exit("Set GDRIVE_SERVICE_ACCOUNT_JSON or GDRIVE_API_KEY.")

    def list_children(self, folder_id: str) -> list[dict]:
        items, token = [], None
        while True:
            params = {
                **self.params,
                "q": f"'{folder_id}' in parents and trashed = false",
                "fields": "nextPageToken, files(id, name, mimeType, modifiedTime, size)",
                "pageSize": "1000",
                "includeItemsFromAllDrives": "true",
            }
            if token:
                params["pageToken"] = token
            r = self.session.get(API, params=params, timeout=60)
            if r.status_code != 200:
                sys.exit(f"Drive list failed ({r.status_code}): {r.text[:500]}")
            data = r.json()
            items.extend(data.get("files", []))
            token = data.get("nextPageToken")
            if not token:
                return items

    def download(self, f: dict, dest: Path) -> None:
        if f["mimeType"] == SHEET_MIME:  # native Google Sheet -> export first tab as CSV
            url = f"{API}/{f['id']}/export"
            params = {**self.params, "mimeType": "text/csv"}
        else:
            url = f"{API}/{f['id']}"
            params = {**self.params, "alt": "media"}
        r = self.session.get(url, params=params, timeout=300)
        if r.status_code != 200:
            print(f"  ! failed {f['name']} ({r.status_code}): {r.text[:200]}")
            return
        dest.write_bytes(r.content)


def walk(drive: Drive, folder_id: str, out: Path, prefix: str = "") -> list[dict]:
    got = []
    for f in drive.list_children(folder_id):
        name = f["name"]
        if f["mimeType"] == FOLDER_MIME:
            got += walk(drive, f["id"], out, prefix + name + "__")
            continue
        is_csv = name.lower().endswith(".csv") or f["mimeType"] in ("text/csv", SHEET_MIME)
        if not is_csv:
            continue
        local = name if name.lower().endswith(".csv") else name + ".csv"
        dest = out / (prefix + local)
        drive.download(f, dest)
        print(f"  ✓ {prefix}{name}")
        got.append({"name": prefix + local, "id": f["id"], "modified": f.get("modifiedTime")})
    return got


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=os.environ.get("GDRIVE_FOLDER_ID"))
    ap.add_argument("--out", default="data/raw")
    a = ap.parse_args()
    if not a.folder:
        sys.exit("No folder id (pass --folder or set GDRIVE_FOLDER_ID).")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    drive = Drive()
    files = walk(drive, a.folder, out)
    (out / "_manifest.json").write_text(json.dumps(files, indent=1))
    print(f"Downloaded {len(files)} CSV file(s) to {out}/")
    if not files:
        sys.exit("No CSVs found — check the folder id and sharing.")


if __name__ == "__main__":
    main()
