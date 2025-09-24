# app.py — Clustering + Batch Navigation + Skip/Mark + Camera/Gallery Upload
# ----------------------------------------------------------------------------
# Auto-save progress to JSON, session-only photo storage, batch navigation

import math
import io
import json
import hashlib
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import DBSCAN

# ----------------- JSON Progress Persistence -----------------
PROGRESS_ROOT = Path("./progress")
PROGRESS_ROOT.mkdir(parents=True, exist_ok=True)

def dataset_fingerprint(file_bytes: bytes) -> str:
    return hashlib.sha1(file_bytes).hexdigest()

def progress_path(dataset_id: str) -> Path:
    return PROGRESS_ROOT / f"{dataset_id}.json"

def load_progress(dataset_id: str):
    p = progress_path(dataset_id)
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        visited = set(map(str, data.get("visited_ticket_ids", [])))
        skipped = set(map(str, data.get("skipped_ticket_ids", [])))
        photo_counts = data.get("uploaded_after_photos", {})  # {issue_id: count}
        return visited, skipped, photo_counts
    return set(), set(), {}

def save_progress(dataset_id: str, visited_ids: set, skipped_ids: set, photo_counts: dict):
    if not dataset_id:
        return
    p = progress_path(dataset_id)
    payload = {
        "visited_ticket_ids": sorted(list(visited_ids)),
        "skipped_ticket_ids": sorted(list(skipped_ids)),
        "uploaded_after_photos": photo_counts,
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

def haversine_m(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(float(lat1)), math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlmb = math.radians(float(lon2) - float(lon1))
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2)**2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))

def build_nearest_neighbor_sequence(df_like: pd.DataFrame, start_lat: float, start_lon: float, limit: int):
    seq = []
    pool = df_like.copy()
    if pool.empty:
        return seq
    cur_lat, cur_lon = float(start_lat), float(start_lon)
    for _ in range(min(limit, len(pool))):
        pool['__dist'] = pool.apply(lambda r: haversine_m(cur_lat, cur_lon, r['LATITUDE'], r['LONGITUDE']), axis=1)
        nxt = pool.sort_values('__dist').iloc[0]
        seq.append(nxt)
        cur_lat, cur_lon = float(nxt['LATITUDE']), float(nxt['LONGITUDE'])
        pool = pool[pool['ISSUE ID'] != nxt['ISSUE ID']]
    return seq

def is_url(u):
    return isinstance(u, str) and u.startswith(("http://", "https://"))

# ----------------- App Setup -----------------
st.set_page_config(layout="wide", page_title="Clustering + Batch Navigation")
st.title("🗺️ Clustering + Batch Navigation")

# Session state
for k in ["visited_ticket_ids", "skipped_ticket_ids"]:
    if k not in st.session_state:
        st.session_state[k] = set()
if "batch_cursor" not in st.session_state:
    st.session_state.batch_cursor = 0
if "uploaded_after_photos" not in st.session_state:
    st.session_state.uploaded_after_photos = {}
if "restored_photo_counts" not in st.session_state:
    st.session_state.restored_photo_counts = {}
if "dataset_id" not in st.session_state:
    st.session_state.dataset_id = None

# ----------------- Inputs -----------------
csv_file = st.file_uploader("Upload CSV", type=["csv"], key="csv_uploader_main")
subcategory_option = st.selectbox(
    "Issue Subcategory",
    [
        "Pothole","Garbage dumped on public land","Unpaved Road","Broken Footpath / Divider",
        "Sand piled on roadsides + Mud/slit on roadside","Malba, bricks, bori, etc dumped on public land",
        "Construction/ demolition activity without safeguards","Encroachment-Building Materials Dumped on Road",
        "Burning of garbage, plastic, leaves, branches etc.","Overflowing Dustbins",
        "Barren land to be greened","Greening of Central Verges","Unsurfaced Parking Lots"
    ],
    key="sel_subcategory"
)
radius_m = st.number_input("Clustering radius (m)", 1, 1000, 15, key="num_radius")
min_samples = st.number_input("Minimum per cluster", 1, 100, 2, key="num_min_samples")

if not csv_file:
    st.info("Upload the required CSV to proceed.")
    st.stop()

# Load CSV
csv_bytes = csv_file.getvalue()
dataset_id = hashlib.sha1(csv_bytes).hexdigest()
st.session_state.dataset_id = dataset_id
st.caption(f"Dataset ID: `{dataset_id[:8]}…`")

df = pd.read_csv(io.BytesIO(csv_bytes))
missing = list(REQUIRED_COLS - set(df.columns))
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# Restore prior progress
rest_visited, rest_skipped, rest_photo_counts = load_progress(dataset_id)
st.session_state.visited_ticket_ids |= rest_visited
st.session_state.skipped_ticket_ids |= rest_skipped
st.session_state.restored_photo_counts = rest_photo_counts

df['SUBCATEGORY_NORM'] = normalize_subcategory(df['SUBCATEGORY'])
df = df[df['SUBCATEGORY_NORM'] == subcategory_option.lower()].copy()
df['LATITUDE']  = pd.to_numeric(df['LATITUDE'], errors='coerce')
df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
df = df[df['LATITUDE'].between(-90, 90) & df['LONGITUDE'].between(-180, 180)]
df.dropna(subset=['LATITUDE','LONGITUDE'], inplace=True)

if df.empty:
    st.warning("No valid rows for the selected subcategory after cleaning LAT/LON.")
    st.stop()

# Clustering
coords_deg = df[['LATITUDE','LONGITUDE']].to_numpy(dtype=float)
coords_rad = np.radians(coords_deg)
eps_rad = float(radius_m) / EARTH_RADIUS_M
db = DBSCAN(eps=eps_rad, min_samples=int(min_samples), metric='haversine', algorithm='ball_tree')
labels = db.fit_predict(coords_rad)
df['CLUSTER NUMBER'] = labels
df['IS_CLUSTERED'] = df['CLUSTER NUMBER'] != -1

# ----------------- Batch Navigation -----------------
st.subheader("Step 4: Set Start Point (Optional)")

origin_lat = st.number_input("Start Latitude (optional)", value=float(df['LATITUDE'].mean()))
origin_lon = st.number_input("Start Longitude (optional)", value=float(df['LONGITUDE'].mean()))

# Filter out visited/skipped
pool = df[~df['ISSUE ID'].astype(str).isin(st.session_state.visited_ticket_ids)]
pool = pool[~pool['ISSUE ID'].astype(str).isin(st.session_state.skipped_ticket_ids)]
st.caption(f"Remaining tickets: {len(pool)} | Visited: {len(st.session_state.visited_ticket_ids)} | Skipped: {len(st.session_state.skipped_ticket_ids)}")

if pool.empty:
    st.info("No tickets remaining after filters/visited/skipped.")
    st.stop()

# Build nearest neighbor sequence
long_seq_rows = build_nearest_neighbor_sequence(pool, origin_lat, origin_lon, limit=min(200, len(pool)))
seq_df = pd.DataFrame(long_seq_rows)

if st.session_state.batch_cursor >= len(seq_df):
    st.session_state.batch_cursor = 0
start = st.session_state.batch_cursor
end = min(start + 10, len(seq_df))
batch_df = seq_df.iloc[start:end].copy()

st.subheader("Step 5: Current Batch — View & Upload After Photos")

def _save_photo_bytes(img_bytes: bytes, issue_id: str, ward: str, status: str, original_name: str, source: str):
    ts_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    saved_name = f"{issue_id}_after_{ts_str}.jpg"
    st.session_state.uploaded_after_photos.setdefault(str(issue_id), []).append({
        "bytes": img_bytes,
        "saved_name": saved_name,
        "original_name": original_name,
        "source": source,
        "ward": str(ward),
        "status": str(status),
        "ts_str": ts_str,
    })
    counts = {iid: len(items) for iid, items in st.session_state.uploaded_after_photos.items()}
    save_progress(st.session_state.dataset_id,
                  st.session_state.visited_ticket_ids,
                  st.session_state.skipped_ticket_ids,
                  counts)

for _, row in batch_df.iterrows():
    issue_id = str(row['ISSUE ID'])
    ward     = row.get('WARD', '')
    status   = row.get('STATUS', '')

    st.markdown(f"**Issue {issue_id}** — Ward {ward}, Status: {status}")
    cam = st.camera_input(f"Take photo ({issue_id})", key=f"cam_{issue_id}")
    if cam is not None:
        _save_photo_bytes(cam.getvalue(), issue_id, ward, status, original_name="camera.jpg", source="camera")
        st.success("Captured ✅")
    up = st.file_uploader(f"Upload photo ({issue_id})", type=["jpg","jpeg","png"], key=f"upl_{issue_id}")
    if up is not None:
        _save_photo_bytes(up.read(), issue_id, ward, status, original_name=up.name, source="upload")
        st.success("Uploaded ✅")

    # Status line
    if issue_id in st.session_state.uploaded_after_photos:
        cnt = len(st.session_state.uploaded_after_photos[issue_id])
        st.info(f"✅ {cnt} photo(s) saved in this session")
    elif issue_id in st.session_state.restored_photo_counts:
        cnt = st.session_state.restored_photo_counts[issue_id]
        st.info(f"✅ {cnt} photo(s) previously saved (no files in session)")
    else:
        st.warning("⚠️ No After Photo saved yet")
    st.divider()

# ---------------- Batch Actions ----------------
c1, c2, c3, c4 = st.columns(4)
first_id = str(batch_df.iloc[0]['ISSUE ID']) if not batch_df.empty else None

def _persist_counts_now():
    counts = {iid: len(items) for iid, items in st.session_state.uploaded_after_photos.items()}
    save_progress(st.session_state.dataset_id,
                  st.session_state.visited_ticket_ids,
                  st.session_state.skipped_ticket_ids,
                  counts)

if c1.button("Mark All Visited"):
    for iid in batch_df['ISSUE ID'].astype(str):
        st.session_state.visited_ticket_ids.add(iid)
    _persist_counts_now()
    st.session_state.batch_cursor += 10
    st.experimental_rerun()

if c2.button("Skip All"):
    for iid in batch_df['ISSUE ID'].astype(str):
        st.session_state.skipped_ticket_ids.add(iid)
    _persist_counts_now()
    st.session_state.batch_cursor += 10
    st.experimental_rerun()

if c3.button("Next Batch"):
    st.session_state.batch_cursor += 10
    st.experimental_rerun()

if c4.button("Reset Batch Cursor"):
    st.session_state.batch_cursor = 0
    st.experimental_rerun()

st.caption("All progress is auto-saved locally and restored on reload.")
