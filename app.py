# app.py — Clustering + Batch Navigation + Skip/Mark + Clickable Image Popups
# Local auto-save version: JSON persistence ensures progress survives refreshes or interruptions
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

def hyperlinkify_excel(excel_path):
    try:
        wb = load_workbook(excel_path)
        ws = wb["Clustering Application Summary"]
        for row in range(2, ws.max_row + 1):
            for col in (11, 12):  # BEFORE/AFTER PHOTO columns
                link = ws.cell(row, col).value
                if link and isinstance(link, str) and link.startswith(("http://", "https://")):
                    ws.cell(row, col).hyperlink = link
                    ws.cell(row, col).style = "Hyperlink"
        wb.save(excel_path)
    except Exception:
        pass

def load_wards_uploaded(file):
    if not HAVE_GPD:
        st.info("GeoPandas not available; ward overlay/join skipped.")
        return None
    try:
        suffix = file.name.split(".")[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix="."+suffix) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        if suffix in ("geojson", "json"):
            gdf = gpd.read_file(tmp_path)
        elif suffix == "kml":
            if not HAS_FIONA:
                st.warning("KML requires Fiona/GDAL. Upload GeoJSON/JSON instead.")
                return None
            layers = fiona.listlayers(tmp_path)
            gdf = None
            for layer in layers:
                gdf_try = gpd.read_file(tmp_path, driver="KML", layer=layer)
                if not gdf_try.empty:
                    gdf = gdf_try
                    break
            if gdf is None:
                st.warning("No non-empty KML layers found.")
                return None
        else:
            st.warning("Unsupported ward file type. Use GeoJSON/JSON/KML.")
            return None

        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True)
        elif gdf.crs.to_string() != "EPSG:4326":
            gdf = gdf.to_crs(epsg=4326)
        return gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    except Exception:
        return None

def geocode_query(q: str, timeout=6):
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": q, "format": "json", "limit": 1}
        headers = {"User-Agent": "cluster-app/1.0"}
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            if data:
                return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception:
        pass
    return None, None

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
ward_file = st.file_uploader("Upload Wards file (optional)", type=["geojson","json","kml"], key="ward_uploader_main")

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

# Read CSV
csv_bytes = csv_file.getvalue()
dataset_id = dataset_fingerprint(csv_bytes)
st.session_state.dataset_id = dataset_id
st.caption(f"Dataset ID: `{dataset_id[:8]}…`")

df = pd.read_csv(io.BytesIO(csv_bytes))
missing = list(REQUIRED_COLS - set(df.columns))
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# Restore progress
rest_visited, rest_skipped, rest_photo_counts = load_progress(dataset_id)
st.session_state.visited_ticket_ids |= rest_visited
st.session_state.skipped_ticket_ids |= rest_skipped
st.session_state.restored_photo_counts = rest_photo_counts

# ----------------- Subcategory Filtering -----------------
df['SUBCATEGORY_NORM'] = normalize_subcategory(df['SUBCATEGORY'])
df_sel = df[df['SUBCATEGORY_NORM'] == normalize_subcategory(pd.Series([subcategory_option])).iloc[0]]
if df_sel.empty:
    st.warning("No tickets found for this subcategory.")
    st.stop()

# Remove already visited/skipped
df_sel = df_sel[~df_sel['ISSUE ID'].astype(str).isin(st.session_state.visited_ticket_ids | st.session_state.skipped_ticket_ids)]
df_sel.reset_index(drop=True, inplace=True)

if df_sel.empty:
    st.info("All tickets for this subcategory are already visited or skipped.")
    st.stop()

# ----------------- Clustering -----------------
coords = df_sel[['LATITUDE','LONGITUDE']].to_numpy()
db = DBSCAN(eps=radius_m/1000/EARTH_RADIUS_M, min_samples=min_samples, algorithm='ball_tree', metric='haversine')
df_sel['CLUSTER'] = db.fit_predict(np.radians(coords))

# ----------------- Map Start -----------------
st.subheader("🌍 Click map to select start location / search address")

m = folium.Map(location=[df_sel['LATITUDE'].mean(), df_sel['LONGITUDE'].mean()], zoom_start=13)
if ward_file and HAVE_GPD:
    gdf_wards = load_wards_uploaded(ward_file)
    if gdf_wards is not None:
        folium.GeoJson(gdf_wards.geometry).add_to(m)

clicked_coords = st_folium(m, height=400, key="map_start_click")

origin_lat, origin_lon = st.session_state.origin_lat, st.session_state.origin_lon
if clicked_coords and clicked_coords.get("last_clicked"):
    origin_lat = clicked_coords["last_clicked"]["lat"]
    origin_lon = clicked_coords["last_clicked"]["lng"]
    st.session_state.origin_lat = origin_lat
    st.session_state.origin_lon = origin_lon

addr_input = st.text_input("Or search address")
if addr_input:
    qlat, qlon = geocode_query(addr_input)
    if qlat:
        origin_lat, origin_lon = qlat, qlon
        st.session_state.origin_lat, st.session_state.origin_lon = qlat, qlon
        st.success(f"Address geocoded: {qlat},{qlon}")
    else:
        st.warning("Unable to geocode address.")

if origin_lat is None or origin_lon is None:
    st.info("Select start location on map or search address.")
    st.stop()

# ----------------- Nearest Neighbor Sequence -----------------
batch_limit = 10
nn_sequence = build_nearest_neighbor_sequence(df_sel, origin_lat, origin_lon, batch_limit)
if not nn_sequence:
    st.info("No tickets to visit in this batch.")
    st.stop()

# ----------------- Batch Navigation -----------------
st.subheader(f"📝 Current Batch ({len(nn_sequence)} tickets)")

for idx, ticket in enumerate(nn_sequence):
    tid = str(ticket['ISSUE ID'])
    st.markdown(f"**{idx+1}. {ticket['ADDRESS']} (ID: {tid})**")
    cols = st.columns([2,2,1,1,1])
    with cols[0]:
        before_link = ticket['BEFORE PHOTO']
        if is_url(before_link):
            st.markdown(f"[Before]({before_link})", unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f"[After]({ticket['AFTER PHOTO']})" if is_url(ticket['AFTER PHOTO']) else "No after photo")
    with cols[2]:
        if st.button(f"✅ Mark Visited {tid}", key=f"btn_visited_{tid}"):
            st.session_state.visited_ticket_ids.add(tid)
            save_progress(dataset_id, st.session_state.visited_ticket_ids, st.session_state.skipped_ticket_ids,
                          st.session_state.uploaded_after_photos)
            st.experimental_rerun()
    with cols[3]:
        if st.button(f"⏭️ Skip {tid}", key=f"btn_skip_{tid}"):
            st.session_state.skipped_ticket_ids.add(tid)
            save_progress(dataset_id, st.session_state.visited_ticket_ids, st.session_state.skipped_ticket_ids,
                          st.session_state.uploaded_after_photos)
            st.experimental_rerun()
    with cols[4]:
        uploaded_file = st.file_uploader(f"Upload After Photo {tid}", key=f"after_photo_{tid}", type=["jpg","png","jpeg"])
        if uploaded_file:
            if tid not in st.session_state.uploaded_after_photos:
                st.session_state.uploaded_after_photos[tid] = []
            st.session_state.uploaded_after_photos[tid].append(uploaded_file.name)
            save_progress(dataset_id, st.session_state.visited_ticket_ids, st.session_state.skipped_ticket_ids,
                          st.session_state.uploaded_after_photos)
            st.success(f"Uploaded {uploaded_file.name}. Total uploaded: {len(st.session_state.uploaded_after_photos[tid])}")

# ----------------- Summary -----------------
st.subheader("📊 Batch Summary")
st.write(f"Visited this session: {len(st.session_state.visited_ticket_ids)}")
st.write(f"Skipped this session: {len(st.session_state.skipped_ticket_ids)}")
st.write(f"After photos uploaded: {sum(len(v) for v in st.session_state.uploaded_after_photos.values())}")

