# app.py — Clustering + Batch Navigation + Skip/Mark + Clickable Image Popups
# with Map-Click Start & Address Search (no GPS/IP needed)
# ----------------------------------------------------------------------------
# Added: JSON persistence for visited/skipped + photo counts (no photos saved to disk!)
# - Progress JSON: ./progress/<dataset_id>.json
# - Remembers visited/skipped and how many photos uploaded per Issue ID
# - Actual photos remain session-only (in memory); use the ZIP download to keep them

import math
import io
import json
import hashlib
from pathlib import Path
from datetime import datetime
import tempfile
import requests
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.cluster import DBSCAN
import folium
from folium.plugins import MarkerCluster
from openpyxl import load_workbook
from streamlit_folium import st_folium  # << for map click and map display
from html import escape
from urllib.parse import quote

# ----------------- JSON Progress Persistence -----------------
PROGRESS_ROOT = Path("./progress")
PROGRESS_ROOT.mkdir(parents=True, exist_ok=True)

def dataset_fingerprint(file_bytes: bytes) -> str:
    """Stable ID for the uploaded CSV so we can restore progress on re-uploads."""
    return hashlib.sha1(file_bytes).hexdigest()

def progress_path(dataset_id: str) -> Path:
    return PROGRESS_ROOT / f"{dataset_id}.json"

def load_progress(dataset_id: str):
    """Return (visited_set, skipped_set, photo_counts_dict)."""
    p = progress_path(dataset_id)
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        visited = set(map(str, data.get("visited_ticket_ids", [])))
        skipped = set(map(str, data.get("skipped_ticket_ids", [])))
        photo_counts = {str(k): int(v) for k, v in data.get("uploaded_after_photos", {}).items()}
        return visited, skipped, photo_counts
    return set(), set(), {}

def save_progress(dataset_id: str, visited_ids: set, skipped_ids: set, photo_counts: dict):
    """Persist visited/skipped and photo counts (no image bytes/paths)."""
    if not dataset_id:
        return
    p = progress_path(dataset_id)
    payload = {
        "visited_ticket_ids": sorted(list(map(str, visited_ids))),
        "skipped_ticket_ids": sorted(list(map(str, skipped_ids))),
        "uploaded_after_photos": {str(k): int(v) for k, v in photo_counts.items()},
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

# ----------------- Constants -----------------
EARTH_RADIUS_M = 6_371_000.0
REQUIRED_COLS = {
    'ISSUE ID', 'CITY', 'ZONE', 'WARD', 'SUBCATEGORY', 'CREATED AT',
    'STATUS', 'LATITUDE', 'LONGITUDE', 'BEFORE PHOTO', 'AFTER PHOTO', 'ADDRESS'
}

# ----------------- Helpers -----------------
def normalize_subcategory(series: pd.Series):
    return series.astype(str).str.strip().str.lower()

def hav

