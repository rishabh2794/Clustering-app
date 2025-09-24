import math, io, json, hashlib
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
from streamlit_folium import st_folium
from html import escape

# Optional GeoPandas
try:
    import geopandas as gpd
    HAVE_GPD = True
except:
    HAVE_GPD = False
try:
    import fiona
    HAS_FIONA = True
except:
    HAS_FIONA = False

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

# ----------------- Constants & Helpers -----------------
EARTH_RADIUS_M = 6_371_000.0
REQUIRED_COLS = {'ISSUE ID','CITY','ZONE','WARD','SUBCATEGORY','CREATED AT',
                 'STATUS','LATITUDE','LONGITUDE','BEFORE PHOTO','AFTER PHOTO','ADDRESS'}

def normalize_subcategory(series: pd.Series):
    return series.astype(str).str.strip().str.lower()

def haversine_m(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(float(lat1)), math.radians(float(lat2))
    dphi = math.radians(float(lat2)-float(lat1))
    dlmb = math.radians(float(lon2)-float(lon1))
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2)**2
    return 2*EARTH_RADIUS_M*math.asin(math.sqrt(a))

def google_maps_url(origin_lat, origin_lon, dest_lat, dest_lon, mode="driving", waypoints=None):
    base="https://www.google.com/maps/dir/?api=1"
    parts=[]
    if origin_lat is not None and origin_lon is not None:
        parts.append(f"origin={origin_lat},{origin_lon}")
    parts.append(f"destination={dest_lat},{dest_lon}")
    mode = mode if mode in {"driving","walking","bicycling","transit"} else "driving"
    parts.append(f"travelmode={mode}")
    if waypoints:
        from urllib.parse import quote
        wp = "|".join([f"{lat},{lon}" for (lat,lon) in waypoints])
        parts.append(f"waypoints={quote(wp,safe='|,')}")
    return base+"&".join(parts)

def is_url(u):
    return isinstance(u,str) and u.startswith(("http://","https://"))

def build_nearest_neighbor_sequence(df_like: pd.DataFrame, start_lat: float, start_lon: float, limit: int):
    seq=[]
    pool = df_like.copy()
    if pool.empty: return seq
    cur_lat, cur_lon = float(start_lat), float(start_lon)
    for _ in range(min(limit,len(pool))):
        pool['__dist'] = pool.apply(lambda r: haversine_m(cur_lat, cur_lon, r['LATITUDE'], r['LONGITUDE']), axis=1)
        nxt = pool.sort_values('__dist').iloc[0]
        seq.append(nxt)
        cur_lat, cur_lon = float(nxt['LATITUDE']), float(nxt['LONGITUDE'])
        pool = pool[pool['ISSUE ID']!=nxt['ISSUE ID']]
    return seq

# ----------------- Session State Initialization -----------------
for k in ["visited_ticket_ids","skipped_ticket_ids","uploaded_after_photos","restored_photo_counts"]:
    if k not in st.session_state:
        st.session_state[k] = {} if "photo" in k else set()
for k in ["batch_cursor","origin_lat","origin_lon","dataset_id"]:
    if k not in st.session_state:
        st.session_state[k] = 0 if k=="batch_cursor" else None

# ----------------- App -----------------
st.set_page_config(layout="wide", page_title="Clustering + Batch Navigation")
st.title("🗺️ Clustering + Batch Navigation")

# Uploads
csv_file = st.file_uploader("Upload CSV", type=["csv"])
ward_file = st.file_uploader("Upload Wards file (optional)", type=["geojson","json","kml"])

subcategory_option = st.selectbox(
    "Issue Subcategory",
    ["Pothole","Garbage dumped on public land","Unpaved Road","Broken Footpath / Divider",
     "Sand piled on roadsides + Mud/slit on roadside","Malba, bricks, bori, etc dumped on public land",
     "Construction/ demolition activity without safeguards","Encroachment-Building Materials Dumped on Road",
     "Burning of garbage, plastic, leaves, branches etc.","Overflowing Dustbins",
     "Barren land to be greened","Greening of Central Verges","Unsurfaced Parking Lots"]
)

radius_m = st.number_input("Clustering radius (m)",1,1000,15)
min_samples = st.number_input("Minimum per cluster",1,100,2)

if not csv_file: st.stop()

# Load CSV
csv_bytes = csv_file.getvalue()
dataset_id = dataset_fingerprint(csv_bytes)
st.session_state.dataset_id = dataset_id
st.caption(f"Dataset ID: `{dataset_id[:8]}…`")

df = pd.read_csv(io.BytesIO(csv_bytes))
missing = list(REQUIRED_COLS - set(df.columns))
if missing: st.error(f"Missing columns: {missing}"); st.stop()

# Restore progress
rest_visited, rest_skipped, rest_photo_counts = load_progress(dataset_id)
st.session_state.visited_ticket_ids |= rest_visited
st.session_state.skipped_ticket_ids |= rest_skipped
st.session_state.restored_photo_counts = rest_photo_counts

df['SUBCATEGORY_NORM'] = normalize_subcategory(df['SUBCATEGORY'])
df = df[df['SUBCATEGORY_NORM']==subcategory_option.lower()]
df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
df = df[df['LATITUDE'].between(-90,90) & df['LONGITUDE'].between(-180,180)]
df.dropna(subset=['LATITUDE','LONGITUDE'], inplace=True)
if df.empty: st.warning("No valid rows for selected subcategory"); st.stop()

# ----------------- Clustering -----------------
coords_rad = np.radians(df[['LATITUDE','LONGITUDE']].to_numpy())
eps_rad = radius_m/EARTH_RADIUS_M
db = DBSCAN(eps=eps_rad, min_samples=int(min_samples), metric='haversine', algorithm='ball_tree')
df['CLUSTER NUMBER'] = db.fit_predict(coords_rad)
df['IS_CLUSTERED'] = df['CLUSTER NUMBER']!=-1

# ----------------- Start Point -----------------
st.subheader("Step 1: Set Start Point")
colA,colB = st.columns(2)
with colA:
    addr_query = st.text_input("Search address / landmark (optional)")
    if st.button("Search & set start"):
        if addr_query.strip():
            try:
                r = requests.get("https://nominatim.openstreetmap.org/search", params={"q":addr_query,"format":"json","limit":1}, headers={"User-Agent":"cluster-app"})
                r.raise_for_status()
                res = r.json()
                if res:
                    st.session_state.origin_lat=float(res[0]["lat"])
                    st.session_state.origin_lon=float(res[0]["lon"])
                    st.success(f"Start point set: {res[0]['display_name']}")
            except Exception as e:
                st.error(f"Failed to geocode: {e}")
with colB:
    st.write(f"Current start point: {st.session_state.origin_lat},{st.session_state.origin_lon}")

# ----------------- Batch Preparation -----------------
pool = df[~df['ISSUE ID'].astype(str).isin(st.session_state.visited_ticket_ids)]
pool = pool[~pool['ISSUE ID'].astype(str).isin(st.session_state.skipped_ticket_ids)]
seed_lat = st.session_state.origin_lat or pool['LATITUDE'].mean()
seed_lon = st.session_state.origin_lon or pool['LONGITUDE'].mean()
seq_rows = build_nearest_neighbor_sequence(pool, seed_lat, seed_lon, limit=min(200,len(pool)))
seq_df = pd.DataFrame(seq_rows)
start,end = st.session_state.batch_cursor, min(st.session_state.batch_cursor+10,len(seq_df))
batch_df = seq_df.iloc[start:end]

st.subheader(f"Step 2: Current Batch ({start+1}-{end} of {len(seq_df)})")
st.dataframe(batch_df[['ISSUE ID','WARD','SUBCATEGORY','LATITUDE','LONGITUDE']])

# ----------------- Buttons -----------------
col1,col2,col3 = st.columns(3)
with col1:
    if st.button("✅ Mark first as Visited"):
        if not batch_df.empty:
            iid = str(batch_df.iloc[0]['ISSUE ID'])
            st.session_state.visited_ticket_ids.add(iid)
            save_progress(dataset_id, st.session_state.visited_ticket_ids,
                          st.session_state.skipped_ticket_ids,
                          {k:len(v) for k,v in st.session_state.uploaded_after_photos.items()})
            st.experimental_rerun()
with col2:
    if st.button("⏭ Skip first"):
        if not batch_df.empty:
            iid = str(batch_df.iloc[0]['ISSUE ID'])
            st.session_state.skipped_ticket_ids.add(iid)
            save_progress(dataset_id, st.session_state.visited_ticket_ids,
                          st.session_state.skipped_ticket_ids,
                          {k:len(v) for k,v in st.session_state.uploaded_after_photos.items()})
            st.experimental_rerun()
with col3:
    if st.button("➡ Next batch"):
        st.session_state.batch_cursor += 10
        st.experimental_rerun()

# ----------------- Map -----------------
st.subheader("Step 3: Map View")
m = folium.Map(location=[seed_lat,seed_lon], zoom_start=14)
marker_cluster = MarkerCluster().add_to(m)
for _,r in batch_df.iterrows():
    color="blue" if str(r['ISSUE ID']) not in st.session_state.visited_ticket_ids else "green"
    folium.Marker(location=[r['LATITUDE'],r['LONGITUDE']],
                  popup=f"ID: {r['ISSUE ID']}", icon=folium.Icon(color=color)).add_to(marker_cluster)
if st.session_state.origin_lat and st.session_state.origin_lon:
    folium.Marker(location=[st.session_state.origin_lat,st.session_state.origin_lon],
                  popup="Start", icon=folium.Icon(color="red")).add_to(m)
st_data = st_folium(m, width=700,height=500)

# ----------------- Photo Upload -----------------
st.subheader("Step 4: Upload Photos")
if not batch_df.empty:
    first_row = batch_df.iloc[0]
    iid = str(first_row['ISSUE ID'])
    uploaded_files = st.file_uploader(f"Upload photo for ISSUE ID {iid}", type=["png","jpg","jpeg"], accept_multiple_files=True)
    if uploaded_files:
        for f in uploaded_files:
            bytes_io = f.read()
            st.session_state.uploaded_after_photos.setdefault(iid,[]).append(bytes_io)
        save_progress(dataset_id, st.session_state.visited_ticket_ids,
                      st.session_state.skipped_ticket_ids,
                      {k:len(v) for k,v in st.session_state.uploaded_after_photos.items()})
        st.success(f"Uploaded {len(uploaded_files)} photo(s) for ISSUE ID {iid}")

# ----------------- Google Maps Link -----------------
st.subheader("Step 5: Directions")
if not batch_df.empty:
    dest_lat = batch_df.iloc[-1]['LATITUDE']
    dest_lon = batch_df.iloc[-1]['LONGITUDE']
    waypoints = list(batch_df[['LATITUDE','LONGITUDE']].itertuples(index=False, name=None))
    url = google_maps_url(st.session_state.origin_lat, st.session_state.origin_lon, dest_lat, dest_lon, waypoints=waypoints)
    st.markdown(f"[Open in Google Maps]({url})", unsafe_allow_html=True)

