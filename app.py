# app.py — Batch Navigation + Skip/Mark + Map + After Photo uploads
# ---------------------------------------------------------------

import math
import io
import json
import hashlib
from pathlib import Path
from datetime import datetime
import requests
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.cluster import DBSCAN
import folium
from folium.plugins import MarkerCluster
from openpyxl import load_workbook
from streamlit_folium import st_folium
from html import escape

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
        photo_counts = data.get("uploaded_after_photos", {})
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
    'ISSUE ID','CITY','ZONE','WARD','SUBCATEGORY','CREATED AT',
    'STATUS','LATITUDE','LONGITUDE','BEFORE PHOTO','AFTER PHOTO','ADDRESS'
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

def is_url(u): return isinstance(u, str) and u.startswith(("http://","https://"))

# ----------------- App Setup -----------------
st.set_page_config(layout="wide", page_title="Batch Navigation App")
st.title("🗺️ Batch Navigation (Skip/Mark • Map • Downloads)")

# Session state
for k in ["visited_ticket_ids","skipped_ticket_ids"]:
    if k not in st.session_state: st.session_state[k] = set()
if "batch_cursor" not in st.session_state: st.session_state.batch_cursor = 0
if "origin_lat" not in st.session_state: st.session_state.origin_lat = None
if "origin_lon" not in st.session_state: st.session_state.origin_lon = None
if "uploaded_after_photos" not in st.session_state: st.session_state.uploaded_after_photos = {}
if "restored_photo_counts" not in st.session_state: st.session_state.restored_photo_counts = {}
if "dataset_id" not in st.session_state: st.session_state.dataset_id = None

# ----------------- Inputs -----------------
csv_file = st.file_uploader("Upload CSV", type=["csv"], key="csv_uploader_main")

subcategory_option = st.selectbox(
    "Issue Subcategory",
    ["Pothole","Garbage dumped on public land","Unpaved Road","Broken Footpath / Divider",
     "Overflowing Dustbins","Barren land to be greened","Greening of Central Verges","Unsurfaced Parking Lots"],
    key="sel_subcategory"
)

radius_m = st.number_input("Clustering radius (m)", 1, 1000, 15, key="num_radius")
min_samples = st.number_input("Minimum per cluster", 1, 100, 2, key="num_min_samples")

# ----------------- Main -----------------
if not csv_file:
    st.info("Upload the required CSV to proceed.")
    st.stop()

csv_bytes = csv_file.getvalue()
dataset_id = dataset_fingerprint(csv_bytes)
st.session_state.dataset_id = dataset_id

df = pd.read_csv(io.BytesIO(csv_bytes))
missing = list(REQUIRED_COLS - set(df.columns))
if missing: st.error(f"Missing required columns: {missing}"); st.stop()

rest_visited, rest_skipped, rest_photo_counts = load_progress(dataset_id)
st.session_state.visited_ticket_ids |= rest_visited
st.session_state.skipped_ticket_ids |= rest_skipped
st.session_state.restored_photo_counts = rest_photo_counts

df['SUBCATEGORY_NORM'] = normalize_subcategory(df['SUBCATEGORY'])
df = df[df['SUBCATEGORY_NORM'] == subcategory_option.lower()].copy()
df['LATITUDE']  = pd.to_numeric(df['LATITUDE'], errors='coerce')
df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
df.dropna(subset=['LATITUDE','LONGITUDE'], inplace=True)

if df.empty: st.warning("No valid rows after filtering."); st.stop()

# Clustering
coords_rad = np.radians(df[['LATITUDE','LONGITUDE']])
eps_rad = float(radius_m) / EARTH_RADIUS_M
db = DBSCAN(eps=eps_rad, min_samples=int(min_samples), metric='haversine', algorithm='ball_tree')
df['CLUSTER NUMBER'] = db.fit_predict(coords_rad)
df['IS_CLUSTERED'] = df['CLUSTER NUMBER'] != -1
gdf_all = df.copy()

# ---------------- Start Point ----------------
st.subheader("Step 1: Set Start Point (Map Click or Manual)")
center_lat, center_lon = float(df['LATITUDE'].mean()), float(df['LONGITUDE'].mean())
mini = folium.Map(location=[center_lat, center_lon], zoom_start=12)
click_state = st_folium(mini, height=320, width=None, key="mini_click", returned_objects=["last_clicked"])
if isinstance(click_state, dict) and click_state.get("last_clicked"):
    st.session_state.origin_lat = float(click_state["last_clicked"]["lat"])
    st.session_state.origin_lon = float(click_state["last_clicked"]["lng"])
origin_lat, origin_lon = st.session_state.origin_lat, st.session_state.origin_lon

# ---------------- Batch ----------------
st.subheader("Step 2: Batches of 10 — View • Skip • Mark")
pool = gdf_all[~gdf_all['ISSUE ID'].astype(str).isin(st.session_state.visited_ticket_ids | st.session_state.skipped_ticket_ids)]
if pool.empty:
    st.info("No tickets remaining."); st.stop()
seed_lat, seed_lon = (origin_lat, origin_lon) if origin_lat else (pool['LATITUDE'].mean(), pool['LONGITUDE'].mean())
seq_df = pd.DataFrame(build_nearest_neighbor_sequence(pool, seed_lat, seed_lon, limit=200))
start, end = st.session_state.batch_cursor, st.session_state.batch_cursor+10
batch_df = seq_df.iloc[start:end]

st.dataframe(batch_df[['ISSUE ID','WARD','STATUS','LATITUDE','LONGITUDE','BEFORE PHOTO']], use_container_width=True)

# ---------------- Map ----------------
st.subheader("Step 3: Map")
m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
mc = MarkerCluster(name="Tickets").add_to(m)
for _, row in batch_df.iterrows():
    folium.Marker([row['LATITUDE'],row['LONGITUDE']],popup=f"Issue {row['ISSUE ID']}").add_to(mc)
st_folium(m, height=500, width=None, key="main_map")
