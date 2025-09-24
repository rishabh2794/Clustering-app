import io, json, hashlib
from pathlib import Path
from datetime import datetime
import math

import pandas as pd
import numpy as np
import streamlit as st
from html import escape
import requests
import zipfile

# ----------------- JSON Persistence -----------------
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
        photo_counts = data.get("uploaded_after_photos", {})
        return visited, skipped, photo_counts
    return set(), set(), {}

def save_progress(dataset_id: str, visited_ids: set, skipped_ids: set, photo_counts: dict):
    if not dataset_id:
        return
    payload = {
        "visited_ticket_ids": sorted(list(visited_ids)),
        "skipped_ticket_ids": sorted(list(skipped_ids)),
        "uploaded_after_photos": photo_counts,
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with progress_path(dataset_id).open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

# ----------------- Helpers -----------------
EARTH_RADIUS_M = 6_371_000.0

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
    return isinstance(u, str) and u.startswith(("http://","https://"))

def google_maps_url(origin_lat, origin_lon, dest_lat, dest_lon, waypoints=None):
    base = "https://www.google.com/maps/dir/?api=1"
    parts = []
    if origin_lat is not None and origin_lon is not None:
        parts.append(f"origin={origin_lat},{origin_lon}")
    parts.append(f"destination={dest_lat},{dest_lon}")
    parts.append("travelmode=driving")
    if waypoints:
        from urllib.parse import quote
        wp = "|".join([f"{lat},{lon}" for (lat,lon) in waypoints])
        parts.append(f"waypoints={quote(wp, safe='|,')}")
    return base + "&" + "&".join(parts)

# ----------------- App Setup -----------------
st.set_page_config(layout="wide", page_title="Batch Navigation App")
st.title("🗺️ Nearest-Neighbor Batch Navigation")

REQUIRED_COLS = ['ISSUE ID','WARD','STATUS','LATITUDE','LONGITUDE','BEFORE PHOTO']

# Session state
for k in ["visited_ticket_ids", "skipped_ticket_ids"]:
    if k not in st.session_state: st.session_state[k] = set()
if "batch_cursor" not in st.session_state: st.session_state.batch_cursor = 0
if "origin_lat" not in st.session_state: st.session_state.origin_lat = None
if "origin_lon" not in st.session_state: st.session_state.origin_lon = None
if "uploaded_after_photos" not in st.session_state: st.session_state.uploaded_after_photos = {}
if "restored_photo_counts" not in st.session_state: st.session_state.restored_photo_counts = {}
if "dataset_id" not in st.session_state: st.session_state.dataset_id = None

# ----------------- CSV Upload -----------------
csv_file = st.file_uploader("Upload CSV", type=["csv"])
if not csv_file:
    st.stop()

csv_bytes = csv_file.getvalue()
dataset_id = dataset_fingerprint(csv_bytes)
st.session_state.dataset_id = dataset_id
st.caption(f"Dataset ID: `{dataset_id[:8]}…`")

df = pd.read_csv(io.BytesIO(csv_bytes))
missing = list(set(REQUIRED_COLS)-set(df.columns))
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# Restore prior progress
rest_visited, rest_skipped, rest_photo_counts = load_progress(dataset_id)
st.session_state.visited_ticket_ids |= rest_visited
st.session_state.skipped_ticket_ids |= rest_skipped
st.session_state.restored_photo_counts = rest_photo_counts

df['LATITUDE']  = pd.to_numeric(df['LATITUDE'], errors='coerce')
df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
df = df[df['LATITUDE'].between(-90,90)]
df = df[df['LONGITUDE'].between(-180,180)]
df.dropna(subset=['LATITUDE','LONGITUDE'], inplace=True)

if df.empty:
    st.warning("No valid rows after cleaning LAT/LON.")
    st.stop()

# ----------------- Start Point -----------------
st.subheader("Step 1: Set Start Point (No GPS Needed)")

colA, colB = st.columns(2)
with colA:
    q = st.text_input("Search address/landmark (optional)")
    if st.button("Set from search"):
        if q.strip():
            try:
                r = requests.get("https://nominatim.openstreetmap.org/search", 
                                 params={"q":q,"format":"json","limit":1}, headers={"User-Agent":"batch-app/1.0"}, timeout=6)
                data = r.json()
                if data:
                    st.session_state.origin_lat = float(data[0]["lat"])
                    st.session_state.origin_lon = float(data[0]["lon"])
                    st.success(f"Start set: {st.session_state.origin_lat:.6f},{st.session_state.origin_lon:.6f}")
                    st.rerun()
            except: st.error("Failed to geocode.")
with colB:
    man_lat = st.text_input("Manual latitude")
    man_lon = st.text_input("Manual longitude")
    if st.button("Set manual"):
        try:
            st.session_state.origin_lat = float(man_lat)
            st.session_state.origin_lon = float(man_lon)
            st.success(f"Start set manually.")
            st.rerun()
        except: st.error("Invalid coordinates.")

origin_lat = st.session_state.origin_lat
origin_lon = st.session_state.origin_lon

if origin_lat is None or origin_lon is None:
    seed_lat, seed_lon = float(df['LATITUDE'].mean()), float(df['LONGITUDE'].mean())
else:
    seed_lat, seed_lon = origin_lat, origin_lon

# ----------------- Batch Navigation -----------------
st.subheader("Step 2: Nearest-Neighbor Batch (10 tickets)")

pool = df[~df['ISSUE ID'].astype(str).isin(st.session_state.visited_ticket_ids)]
pool = pool[~pool['ISSUE ID'].astype(str).isin(st.session_state.skipped_ticket_ids)]

st.caption(f"Remaining: {len(pool)}, Visited: {len(st.session_state.visited_ticket_ids)}, Skipped: {len(st.session_state.skipped_ticket_ids)}")

if pool.empty:
    st.info("No tickets remaining.")
    st.stop()

seq_rows = build_nearest_neighbor_sequence(pool, seed_lat, seed_lon, limit=len(pool))
seq_df = pd.DataFrame(seq_rows)

start = st.session_state.batch_cursor
end = min(start+10, len(seq_df))
batch_df = seq_df.iloc[start:end].copy()
if batch_df.empty: batch_df = seq_df.iloc[0:10].copy(); st.session_state.batch_cursor=0

batch_df_display = batch_df[['ISSUE ID','WARD','STATUS','LATITUDE','LONGITUDE','BEFORE PHOTO']].reset_index(drop=True)
batch_df_display.index += 1

def as_link_or_none(x): return x if is_url(str(x)) else None
batch_df_display['BEFORE PHOTO'] = batch_df_display['BEFORE PHOTO'].apply(as_link_or_none)

st.dataframe(batch_df_display, use_container_width=True)

# ----------------- Photo Uploads -----------------
st.markdown("### After Photo Uploads")
def save_photo_bytes(img_bytes, issue_id):
    ts_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    saved_name = f"{issue_id}_after_{ts_str}.jpg"
    st.session_state.uploaded_after_photos.setdefault(str(issue_id), []).append({"bytes":img_bytes,"saved_name":saved_name})
    counts = {iid: len(items) for iid, items in st.session_state.uploaded_after_photos.items()}
    save_progress(st.session_state.dataset_id, st.session_state.visited_ticket_ids, st.session_state.skipped_ticket_ids, counts)

for _, row in batch_df.iterrows():
    iid = str(row['ISSUE ID'])
    st.markdown(f"**Issue {iid}** — Ward {row['WARD']}, Status {row['STATUS']}")
    cam = st.camera_input(f"Camera ({iid})", key=f"cam_{iid}")
    if cam: save_photo_bytes(cam.getvalue(), iid)
    up = st.file_uploader(f"Upload ({iid})", type=["jpg","jpeg","png"], key=f"upl_{iid}")
    if up: save_photo_bytes(up.read(), iid)

# ----------------- Photo Download -----------------
st.subheader("Download After Photos")
if st.session_state.uploaded_after_photos:
    # ZIP per ticket
    for iid, photos in st.session_state.uploaded_after_photos.items():
        if photos:
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as zf:
                for p in photos: zf.writestr(p["saved_name"], p["bytes"])
            st.download_button(f"Download ZIP for Ticket {iid}", buffer.getvalue(), file_name=f"ticket_{iid}.zip")
    # ZIP full batch
    buffer_full = io.BytesIO()
    with zipfile.ZipFile(buffer_full, "w") as zf:
        for iid, photos in st.session_state.uploaded_after_photos.items():
            for p in photos: zf.writestr(p["saved_name"], p["bytes"])
    st.download_button("Download ZIP for Entire Batch", buffer_full.getvalue(), file_name="entire_batch.zip")

# ----------------- Google Maps Directions -----------------
st.subheader("Google Maps Directions")

# Batch only
waypoints_batch = [(row['LATITUDE'], row['LONGITUDE']) for idx,row in batch_df.iterrows()]
if waypoints_batch:
    map_url_batch = google_maps_url(seed_lat, seed_lon, waypoints_batch[-1][0], waypoints_batch[-1][1], waypoints=waypoints_batch[:-1])
    st.markdown(f"[Open batch route in Google Maps]({map_url_batch})")
    st.components.v1.iframe(map_url_batch, width=800, height=400)

# Full remaining route
waypoints_full = [(row['LATITUDE'], row['LONGITUDE']) for row in seq_rows]
if waypoints_full:
    map_url_full = google_maps_url(seed_lat, seed_lon, waypoints_full[-1][0], waypoints_full[-1][1], waypoints=waypoints_full[:-1])
    st.markdown(f"[Open full remaining route in Google Maps]({map_url_full})")
    st.components.v1.iframe(map_url_full, width=800, height=400)

# ----------------- Navigation -----------------
st.subheader("Batch Navigation Controls")
col1, col2 = st.columns(2)
with col1:
    if st.button("Mark all in batch as Visited"):
        for iid in batch_df['ISSUE ID'].astype(str): st.session_state.visited_ticket_ids.add(iid)
        save_progress(st.session_state.dataset_id, st.session_state.visited_ticket_ids, st.session_state.skipped_ticket_ids,
                      {iid: len(v) for iid,v in st.session_state.uploaded_after_photos.items()})
        st.session_state.batch_cursor += 10
        st.experimental_rerun()
with col2:
    if st.button("Skip this batch"):
        for iid in batch_df['ISSUE ID'].astype(str): st.session_state.skipped_ticket_ids.add(iid)
        save_progress(st.session_state.dataset_id, st.session_state.visited_ticket_ids, st.session_state.skipped_ticket_ids,
                      {iid: len(v) for iid,v in st.session_state.uploaded_after_photos.items()})
        st.session_state.batch_cursor += 10
        st.experimental_rerun()
