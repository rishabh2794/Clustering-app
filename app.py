# app.py — Clustering + Batch Navigation + Skip/Mark + Clickable Image Popups + Local Auto-Save
# ----------------------------------------------------------------------------
# Added: JSON persistence + localStorage auto-save for visited/skipped + photo counts
# - Session-only photos still stored in memory; ZIP download remains session-only

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
from html import escape

# Optional heavy deps — guarded (GeoPandas/Fiona may be unavailable)
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
from streamlit_folium import st_folium
from streamlit_js_eval import streamlit_js_eval  # <- for localStorage

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

def hyperlinkify_excel(excel_path):
    try:
        wb = load_workbook(excel_path)
        ws = wb["Clustering Application Summary"]
        for row in range(2, ws.max_row + 1):
            for col in (11, 12):
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

# ----------------- LocalStorage Auto-Save -----------------
def autosave_local():
    save_data = {
        "visited_ticket_ids": list(st.session_state.visited_ticket_ids),
        "skipped_ticket_ids": list(st.session_state.skipped_ticket_ids),
        "uploaded_after_photos_counts": {iid: len(items)
                                         for iid, items in st.session_state.uploaded_after_photos.items()}
    }
    js_code = f"localStorage.setItem('clusterAppState', '{json.dumps(save_data)}');"
    streamlit_js_eval(js_code=js_code, key=f"autosave_js_{datetime.now().timestamp()}")

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

# Restore from localStorage first
saved_json = streamlit_js_eval(js_code="localStorage.getItem('clusterAppState')", key="load_localstorage")
if saved_json:
    try:
        data = json.loads(saved_json)
        st.session_state.visited_ticket_ids |= set(data.get("visited_ticket_ids", []))
        st.session_state.skipped_ticket_ids |= set(data.get("skipped_ticket_ids", []))
        st.session_state.restored_photo_counts |= data.get("uploaded_after_photos_counts", {})
    except Exception:
        pass

# ----------------- Inputs -----------------
csv_file = st.file_uploader("Upload CSV", type=["csv"], key="csv_uploader_main")
ward_file = st.file_uploader("Upload Wards file (optional)", type=["geojson","json","kml"], key="ward_uploader_main")
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

# ----------------- Main -----------------
if not csv_file:
    st.info("Upload the required CSV to proceed.")
    st.stop()

csv_bytes = csv_file.getvalue()
dataset_id = dataset_fingerprint(csv_bytes)
st.session_state.dataset_id = dataset_id

# Load CSV
df = pd.read_csv(csv_file)
for c in REQUIRED_COLS:
    if c not in df.columns:
        st.error(f"Missing required column: {c}")
        st.stop()

df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
df = df[df['LATITUDE'].notna() & df['LONGITUDE'].notna()]

# ----------------- Ward Overlay -----------------
gdf_wards = None
if ward_file:
    gdf_wards = load_wards_uploaded(ward_file)

# ----------------- Batch Loop Example -----------------
batch_size = 10
pending_ids = [str(iid) for iid in df['ISSUE ID'].tolist() if iid not in st.session_state.visited_ticket_ids and iid not in st.session_state.skipped_ticket_ids]
if not pending_ids:
    st.success("✅ All tickets processed.")
else:
    batch_ids = pending_ids[st.session_state.batch_cursor:st.session_state.batch_cursor+batch_size]
    batch_df = df[df['ISSUE ID'].astype(str).isin(batch_ids)]
    for idx, row in batch_df.iterrows():
        iid = str(row['ISSUE ID'])
        st.subheader(f"Ticket {iid} | {row['SUBCATEGORY']} | {row['WARD']}")
        if is_url(row['BEFORE PHOTO']):
            st.image(row['BEFORE PHOTO'], width=300, caption="Before Photo")
        uploaded_file = st.file_uploader(f"Upload AFTER photo for {iid}", key=f"after_{iid}")
        if uploaded_file:
            st.session_state.uploaded_after_photos.setdefault(iid, []).append(uploaded_file)
            st.write(f"✅ Uploaded {len(st.session_state.uploaded_after_photos[iid])} photo(s)")
            autosave_local()  # auto-save
        col1, col2 = st.columns(2)
        if col1.button(f"Mark Visited {iid}"):
            st.session_state.visited_ticket_ids.add(iid)
            autosave_local()
            st.experimental_rerun()
        if col2.button(f"Skip {iid}"):
            st.session_state.skipped_ticket_ids.add(iid)
            autosave_local()
            st.experimental_rerun()

