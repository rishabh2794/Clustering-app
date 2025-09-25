# app.py — Clustering + Batch Navigation + Skip/Mark + Clickable Image Popups
# with Map-Click Start & Address Search (no GPS/IP needed)
# ----------------------------------------------------------------------------
# Added: JSON persistence for visited/skipped + photo counts (no photos saved to disk!)

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

# Optional heavy deps
try:
    import geopandas as gpd
    HAVE_GPD = True
except Exception:
    HAVE_GPD = False

from sklearn.cluster import DBSCAN
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

def google_maps_url(origin_lat, origin_lon, dest_lat, dest_lon, mode="driving", waypoints=None):
    base = "https://www.google.com/maps/dir/?api=1"
    parts = []
    if origin_lat is not None and origin_lon is not None:
        parts.append(f"origin={origin_lat},{origin_lon}")
    parts.append(f"destination={dest_lat},{dest_lon}")
    mode = mode if mode in {"driving","walking","bicycling","transit"} else "driving"
    parts.append(f"travelmode={mode}")
    if waypoints:
        from urllib.parse import quote
        wp = "|".join([f"{lat},{lon}" for (lat,lon) in waypoints])
        parts.append(f"waypoints={quote(wp, safe='|,')}")
    return base + "&" + "&".join(parts)

def is_url(u):
    return isinstance(u, str) and u.startswith(("http://", "https://"))

# ----------------- App Setup -----------------
st.set_page_config(layout="wide", page_title="Clustering + Batch Navigation")
st.title("🗺️ Clustering + Batch Navigation (Map-Click Start • Skip/Mark • Downloads)")

# Session state
for k in ["visited_ticket_ids", "skipped_ticket_ids"]:
    if k not in st.session_state:
        st.session_state[k] = set()
if "batch_cursor" not in st.session_state:
    st.session_state.batch_cursor = 0
if "origin_lat" not in st.session_state:
    st.session_state.origin_lat = None
if "origin_lon" not in st.session_state:
    st.session_state.origin_lon = None
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

# ----------------- Main -----------------
if not csv_file:
    st.info("Upload the required CSV to proceed.")
    st.stop()

csv_bytes = csv_file.getvalue()
dataset_id = dataset_fingerprint(csv_bytes)
st.session_state.dataset_id = dataset_id
st.caption(f"Dataset ID: `{dataset_id[:8]}…`")

df = pd.read_csv(io.BytesIO(csv_bytes))
missing = list(REQUIRED_COLS - set(df.columns))
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

rest_visited, rest_skipped, rest_photo_counts = load_progress(dataset_id)
st.session_state.visited_ticket_ids |= rest_visited
st.session_state.skipped_ticket_ids |= rest_skipped
st.session_state.restored_photo_counts = rest_photo_counts

df['SUBCATEGORY_NORM'] = normalize_subcategory(df['SUBCATEGORY'])
df = df[df['SUBCATEGORY_NORM'] == subcategory_option.lower()].copy()

df['LATITUDE']  = pd.to_numeric(df['LATITUDE'], errors='coerce')
df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
df.dropna(subset=['LATITUDE','LONGITUDE'], inplace=True)

if df.empty:
    st.warning("No valid rows for the selected subcategory after cleaning LAT/LON.")
    st.stop()

coords_deg = df[['LATITUDE','LONGITUDE']].to_numpy(dtype=float)
coords_rad = np.radians(coords_deg)
eps_rad = float(radius_m) / EARTH_RADIUS_M
db = DBSCAN(eps=eps_rad, min_samples=int(min_samples), metric='haversine', algorithm='ball_tree')
labels = db.fit_predict(coords_rad)
df['CLUSTER NUMBER'] = labels
df['IS_CLUSTERED'] = df['CLUSTER NUMBER'] != -1

gdf_all = df.copy()

# ---------------- Start Point ----------------
st.subheader("Step 4: Set Start Point")

center_lat = float(df['LATITUDE'].mean())
center_lon = float(df['LONGITUDE'].mean())
mini = folium.Map(location=[center_lat, center_lon], zoom_start=12)

if st.session_state.origin_lat is not None and st.session_state.origin_lon is not None:
    folium.Marker(
        [st.session_state.origin_lat, st.session_state.origin_lon],
        popup="Start here",
        icon=folium.Icon(color="green", icon="flag")
    ).add_to(mini)

click_state = st_folium(mini, height=320, width=None, key="mini_click", returned_objects=["last_clicked"])
if isinstance(click_state, dict):
    clicked = click_state.get("last_clicked")
    if clicked and "lat" in clicked and "lng" in clicked:
        st.session_state.origin_lat = float(clicked["lat"])
        st.session_state.origin_lon = float(clicked["lng"])
        st.success(f"Start set from map click: {st.session_state.origin_lat:.6f}, {st.session_state.origin_lon:.6f}")
        st.rerun()

origin_lat = st.session_state.origin_lat
origin_lon = st.session_state.origin_lon
if origin_lat is not None and origin_lon is not None:
    st.info(f"Using start: {origin_lat:.6f}, {origin_lon:.6f}")
else:
    st.warning("No start set. Google Maps will start from your phone’s GPS.")

# ---------------- Batch + Skip/Mark ----------------
st.subheader("Step 5: Batches of 10 — View • Skip • Mark")

pool = gdf_all.copy()
pool = pool[~pool['ISSUE ID'].astype(str).isin(st.session_state.visited_ticket_ids)]
pool = pool[~pool['ISSUE ID'].astype(str).isin(st.session_state.skipped_ticket_ids)]

st.caption(f"Remaining: {len(pool)} | Visited: {len(st.session_state.visited_ticket_ids)} | Skipped: {len(st.session_state.skipped_ticket_ids)}")

if pool.empty:
    st.info("No tickets remaining.")
    batch_df = pd.DataFrame(columns=['ISSUE ID','WARD','STATUS','LATITUDE','LONGITUDE','BEFORE PHOTO'])
else:
    batch_df = pool.head(10)

    st.dataframe(
        batch_df[['ISSUE ID','WARD','STATUS','LATITUDE','LONGITUDE','BEFORE PHOTO']].reset_index(drop=True),
        use_container_width=True
    )

    # Google Maps link
    waypoints = [(float(r['LATITUDE']), float(r['LONGITUDE'])) for _, r in batch_df.iterrows()]
    if waypoints:
        last = waypoints[-1]
        mids = waypoints[:-1] if len(waypoints) > 1 else []
        nav_url = google_maps_url(origin_lat, origin_lon, last[0], last[1], waypoints=mids)
        st.markdown(f"[🧭 Open route in Google Maps for this batch]({nav_url})")

    c1, c2, c3, c4 = st.columns(4)
    first_id = str(batch_df.iloc[0]['ISSUE ID']) if not batch_df.empty else None

    with c1:
        if st.button("✅ Mark first as Visited"):
            if first_id:
                st.session_state.visited_ticket_ids.add(first_id)
                st.rerun()
    with c2:
        if st.button("⏭️ Skip first"):
            if first_id:
                st.session_state.skipped_ticket_ids.add(first_id)
                st.rerun()
    with c3:
        if st.button("✅ Mark entire batch as Visited"):
            for _id in batch_df['ISSUE ID'].astype(str).tolist():
                st.session_state.visited_ticket_ids.add(_id)
            st.rerun()
    with c4:
        if st.button("➡️ Next batch"):
            st.session_state.batch_cursor = 0
            st.rerun()
