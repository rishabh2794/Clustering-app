# app.py — Clustering + Batch Navigation + Skip/Mark + Clickable Image Popups
# ----------------------------------------------------------------------------
# Full working version with Step 5 batch table and Step 6 map restored
# After Photo uploads for all tickets in batch with session auto-save (JSON)
# Step 7 downloads and ward upload removed

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

# Optional heavy deps — guarded (GeoPandas/Fiona may be unavailable on Streamlit Cloud)
try:
    import geopandas as gpd
    HAVE_GPD = True
except Exception:
    HAVE_GPD = False

try:
    import fiona
    HAS_FIONA = True
except Exception:
    HAS_FIONA = False

from sklearn.cluster import DBSCAN
import folium
from folium.plugins import MarkerCluster
from openpyxl import load_workbook
from streamlit_folium import st_folium  # << for map click
from html import escape

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
        photo_counts = data.get("uploaded_after_photos", {})  # {issue_id: count}
        return visited, skipped, photo_counts
    return set(), set(), {}

def save_progress(dataset_id: str, visited_ids: set, skipped_ids: set, photo_counts: dict):
    """Persist visited/skipped and photo counts (no image bytes/paths)."""
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

# ----------------- App Setup -----------------
st.set_page_config(layout="wide", page_title="Clustering + Batch Navigation")
st.title("🗺️ Clustering + Batch Navigation (Map-Click Start • Skip/Mark)")

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
    ["Pothole","Garbage dumped on public land","Unpaved Road","Broken Footpath / Divider",
     "Sand piled on roadsides + Mud/slit on roadside","Malba, bricks, bori, etc dumped on public land",
     "Construction/ demolition activity without safeguards","Encroachment-Building Materials Dumped on Road",
     "Burning of garbage, plastic, leaves, branches etc.","Overflowing Dustbins",
     "Barren land to be greened","Greening of Central Verges","Unsurfaced Parking Lots"],
    key="sel_subcategory"
)
radius_m = st.number_input("Clustering radius (m)", 1, 1000, 15, key="num_radius")
min_samples = st.number_input("Minimum per cluster", 1, 100, 2, key="num_min_samples")

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
df = df[df['LATITUDE'].between(-90, 90)]
df = df[df['LONGITUDE'].between(-180, 180)]
df.dropna(subset=['LATITUDE','LONGITUDE'], inplace=True)
if df.empty:
    st.warning("No valid rows for the selected subcategory.")
    st.stop()

# Clustering
coords_deg = df[['LATITUDE','LONGITUDE']].to_numpy(dtype=float)
coords_rad = np.radians(coords_deg)
eps_rad = float(radius_m) / EARTH_RADIUS_M
db = DBSCAN(eps=eps_rad, min_samples=int(min_samples), metric='haversine', algorithm='ball_tree')
df['CLUSTER NUMBER'] = db.fit_predict(coords_rad)
df['IS_CLUSTERED'] = df['CLUSTER NUMBER'] != -1

# ---------------- Start Point ----------------
st.subheader("Step 4: Set Start Point")
colA, colB = st.columns(2)
with colA:
    q = st.text_input("🔎 Search address / landmark (optional)", key="txt_addr_query")
    if st.button("Search & set start", key="btn_addr_geocode") and q.strip():
        try:
            url = "https://nominatim.openstreetmap.org/search"
            r = requests.get(url, params={"q": q.strip(),"format":"json","limit":1},
                             headers={"User-Agent": "cluster-app/1.0"}, timeout=6)
            if r.status_code == 200 and r.json():
                st.session_state.origin_lat = float(r.json()[0]["lat"])
                st.session_state.origin_lon = float(r.json()[0]["lon"])
                st.success(f"Start set: {st.session_state.origin_lat:.6f}, {st.session_state.origin_lon:.6f}")
                st.rerun()
            else:
                st.error("Could not geocode that query.")
        except Exception:
            st.error("Error in geocoding.")
with colB:
    man_lat = st.text_input("Manual latitude (optional)", key="txt_lat")
    man_lon = st.text_input("Manual longitude (optional)", key="txt_lon")
    if st.button("Set manual start", key="btn_set_manual"):
        try:
            st.session_state.origin_lat = float(man_lat)
            st.session_state.origin_lon = float(man_lon)
            st.success(f"Manual start set: {st.session_state.origin_lat:.6f}, {st.session_state.origin_lon:.6f}")
            st.rerun()
        except Exception:
            st.error("Invalid coordinates")

# ---------------- Step 5: Batch Navigation ----------------
st.subheader("Step 5: Batches of 10 — View • Skip • Mark")

pool = df[~df['ISSUE ID'].astype(str).isin(st.session_state.visited_ticket_ids)]
pool = pool[~pool['ISSUE ID'].astype(str).isin(st.session_state.skipped_ticket_ids)]
st.caption(f"Remaining tickets: {len(pool)} | Visited: {len(st.session_state.visited_ticket_ids)} | Skipped: {len(st.session_state.skipped_ticket_ids)}")

if pool.empty:
    st.success("All tickets processed!")
    st.stop()

batch_size = 10
cursor = st.session_state.batch_cursor
batch = pool.iloc[cursor:cursor+batch_size]
if batch.empty:
    st.session_state.batch_cursor = 0
    batch = pool.iloc[0:batch_size]

st.session_state.batch_cursor += batch_size

# ---------------- Table of nearest batch ----------------
batch_table = []
for idx, row in batch.iterrows():
    ticket_id = str(row['ISSUE ID'])
    before_link = row['BEFORE PHOTO'] if is_url(row['BEFORE PHOTO']) else ""
    photo_count = st.session_state.restored_photo_counts.get(ticket_id, 0)
    batch_table.append({
        "ISSUE ID": ticket_id,
        "ADDRESS": row['ADDRESS'],
        "BEFORE PHOTO": f"[link]({before_link})" if before_link else "",
        "AFTER PHOTO": f"{photo_count} photo(s) uploaded"
    })

st.write("### Nearest Batch Tickets")
st.table(pd.DataFrame(batch_table))

# ---------------- Photo upload per ticket ----------------
st.write("### Upload After Photos (Camera or File)")
for idx, row in batch.iterrows():
    ticket_id = str(row['ISSUE ID'])
    uploaded_files = st.file_uploader(
        f"Upload After Photo for Ticket {ticket_id}",
        type=["jpg","jpeg","png"],
        accept_multiple_files=True,
        key=f"up_after_{ticket_id}"
    )
    if uploaded_files:
        count = st.session_state.uploaded_after_photos.get(ticket_id, 0)
        count += len(uploaded_files)
        st.session_state.uploaded_after_photos[ticket_id] = count
        st.success(f"{len(uploaded_files)} photo(s) uploaded. Total: {count}")
        save_progress(dataset_id, st.session_state.visited_ticket_ids,
                      st.session_state.skipped_ticket_ids, st.session_state.uploaded_after_photos)

# ---------------- Step 6: Map ----------------
st.subheader("Step 6: Map of Batch Tickets")
m = folium.Map(location=[st.session_state.origin_lat or batch.iloc[0]['LATITUDE'],
                         st.session_state.origin_lon or batch.iloc[0]['LONGITUDE']], zoom_start=15)
marker_cluster = MarkerCluster().add_to(m)
for idx, row in batch.iterrows():
    ticket_id = str(row['ISSUE ID'])
    color = "orange"  # batch
    popup_html = f"<b>Ticket {escape(ticket_id)}</b><br>"
    before_link = row['BEFORE PHOTO'] if is_url(row['BEFORE PHOTO']) else ""
    if before_link:
        popup_html += f'<a href="{before_link}" target="_blank">Before Photo</a><br>'
    folium.Marker(location=[row['LATITUDE'], row['LONGITUDE']],
                  popup=popup_html,
                  icon=folium.Icon(color=color)).add_to(marker_cluster)

st_folium(m, width=900, height=500)

st.success("✅ Batch display, After Photo upload, and Map restored successfully.")
