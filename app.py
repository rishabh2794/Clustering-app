# app.py — Clustering + Batch Navigation + Skip/Mark + Clickable Image Popups
# ----------------------------------------------------------------------------
# Full version with local auto-save for progress & photo counts

import math, io, json, hashlib, tempfile
from pathlib import Path
from datetime import datetime
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
from html import escape

# ---------------- JSON Persistence ----------------
PROGRESS_ROOT = Path("./progress")
PROGRESS_ROOT.mkdir(parents=True, exist_ok=True)

def dataset_fingerprint(file_bytes: bytes) -> str:
    return hashlib.sha1(file_bytes).hexdigest()

def progress_path(dataset_id: str) -> Path:
    return PROGRESS_ROOT / f"{dataset_id}.json"

def load_progress(dataset_id: str):
    """Return (visited_set, skipped_set, photo_counts_dict)."""
    p = progress_path(dataset_id)
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding='utf-8'))
            visited = set(map(str, data.get("visited_ticket_ids", [])))
            skipped = set(map(str, data.get("skipped_ticket_ids", [])))
            photo_counts = data.get("uploaded_after_photos", {})
            return visited, skipped, photo_counts
        except Exception:
            pass
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
    progress_path(dataset_id).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

# ---------------- Constants ----------------
EARTH_RADIUS_M = 6_371_000.0
REQUIRED_COLS = {'ISSUE ID','CITY','ZONE','WARD','SUBCATEGORY','CREATED AT','STATUS','LATITUDE','LONGITUDE','BEFORE PHOTO','AFTER PHOTO','ADDRESS'}

# ---------------- Helpers ----------------
def normalize_subcategory(series: pd.Series):
    return series.astype(str).str.strip().str.lower()

def haversine_m(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(float(lat1)), math.radians(float(lat2))
    dphi = math.radians(float(lat2)-float(lat1))
    dlmb = math.radians(float(lon2)-float(lon1))
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2)**2
    return 2*EARTH_RADIUS_M*math.asin(math.sqrt(a))

def google_maps_url(origin_lat, origin_lon, dest_lat, dest_lon, mode="driving", waypoints=None):
    base = "https://www.google.com/maps/dir/?api=1"
    parts = []
    if origin_lat is not None and origin_lon is not None:
        parts.append(f"origin={origin_lat},{origin_lon}")
    parts.append(f"destination={dest_lat},{dest_lon}")
    parts.append(f"travelmode={mode}")
    if waypoints:
        from urllib.parse import quote
        wp = "|".join([f"{lat},{lon}" for lat,lon in waypoints])
        parts.append(f"waypoints={quote(wp, safe='|,')}")
    return base + "&" + "&".join(parts)

def is_url(u):
    return isinstance(u,str) and u.startswith(("http://","https://"))

def geocode_query(q: str, timeout=6):
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q":q,"format":"json","limit":1}
        headers = {"User-Agent":"cluster-app/1.0"}
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        if r.status_code==200:
            data = r.json()
            if data:
                return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception:
        pass
    return None, None

def build_nearest_neighbor_sequence(df_like: pd.DataFrame, start_lat: float, start_lon: float, limit: int):
    seq=[]
    pool=df_like.copy()
    if pool.empty:
        return seq
    cur_lat, cur_lon = float(start_lat), float(start_lon)
    for _ in range(min(limit,len(pool))):
        pool['__dist'] = pool.apply(lambda r: haversine_m(cur_lat,cur_lon,r['LATITUDE'],r['LONGITUDE']),axis=1)
        nxt = pool.sort_values('__dist').iloc[0]
        seq.append(nxt)
        cur_lat,cur_lon=float(nxt['LATITUDE']),float(nxt['LONGITUDE'])
        pool=pool[pool['ISSUE ID']!=nxt['ISSUE ID']]
    return seq

# ---------------- Streamlit Session State ----------------
st.set_page_config(layout="wide", page_title="Clustering + Batch Navigation")

for k in ["visited_ticket_ids","skipped_ticket_ids"]:
    if k not in st.session_state: st.session_state[k]=set()
for k in ["batch_cursor","origin_lat","origin_lon","dataset_id"]:
    if k not in st.session_state: st.session_state[k]=None
if "uploaded_after_photos" not in st.session_state:
    st.session_state.uploaded_after_photos={}
if "restored_photo_counts" not in st.session_state:
    st.session_state.restored_photo_counts={}

# ---------------- Inputs ----------------
st.title("🗺️ Clustering + Batch Navigation")
csv_file = st.file_uploader("Upload CSV", type=["csv"])
ward_file = st.file_uploader("Upload Wards (optional)", type=["geojson","json","kml"])

subcategory_option = st.selectbox("Issue Subcategory", [
    "Pothole","Garbage dumped on public land","Unpaved Road","Broken Footpath / Divider",
    "Sand piled on roadsides + Mud/slit on roadside","Malba, bricks, bori, etc dumped on public land",
    "Construction/ demolition activity without safeguards","Encroachment-Building Materials Dumped on Road",
    "Burning of garbage, plastic, leaves, branches etc.","Overflowing Dustbins",
    "Barren land to be greened","Greening of Central Verges","Unsurfaced Parking Lots"
])

radius_m = st.number_input("Clustering radius (m)",1,1000,15)
min_samples = st.number_input("Minimum per cluster",1,100,2)

if not csv_file:
    st.info("Upload CSV to proceed.")
    st.stop()

# ---------------- Load CSV ----------------
csv_bytes=csv_file.getvalue()
dataset_id=dataset_fingerprint(csv_bytes)
st.session_state.dataset_id=dataset_id
st.caption(f"Dataset ID: `{dataset_id[:8]}…`")

df=pd.read_csv(io.BytesIO(csv_bytes))
missing=list(REQUIRED_COLS-set(df.columns))
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

# ---------------- Restore JSON progress ----------------
rest_visited, rest_skipped, rest_photo_counts = load_progress(dataset_id)
st.session_state.visited_ticket_ids |= rest_visited
st.session_state.skipped_ticket_ids |= rest_skipped
st.session_state.restored_photo_counts = rest_photo_counts

df['SUBCATEGORY_NORM']=normalize_subcategory(df['SUBCATEGORY'])
df=df[df['SUBCATEGORY_NORM']==subcategory_option.lower()].copy()
df['LATITUDE']=pd.to_numeric(df['LATITUDE'],errors='coerce')
df['LONGITUDE']=pd.to_numeric(df['LONGITUDE'],errors='coerce')
df=df[df['LATITUDE'].between(-90,90) & df['LONGITUDE'].between(-180,180)]
df.dropna(subset=['LATITUDE','LONGITUDE'], inplace=True)
if df.empty:
    st.warning("No valid rows after cleaning.")
    st.stop()

# ---------------- Clustering ----------------
coords_rad=np.radians(df[['LATITUDE','LONGITUDE']].to_numpy(dtype=float))
eps_rad=float(radius_m)/EARTH_RADIUS_M
db=DBSCAN(eps=eps_rad,min_samples=int(min_samples),metric='haversine',algorithm='ball_tree')
labels=db.fit_predict(coords_rad)
df['CLUSTER NUMBER']=labels
df['IS_CLUSTERED']=df['CLUSTER NUMBER']!=-1

# Optional GeoPandas overlay
if HAVE_GPD:
    gdf_all=gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df['LONGITUDE'], df['LATITUDE']), crs="EPSG:4326")
    if ward_file:
        try:
            suffix=ward_file.name.split('.')[-1].lower()
            with tempfile.NamedTemporaryFile(delete=False,suffix='.'+suffix) as tmp:
                tmp.write(ward_file.read())
                tmp_path=tmp.name
            if suffix in ('geojson','json'): gdf=gpd.read_file(tmp_path)
            elif suffix=='kml' and HAS_FIONA:
                layers=fiona.listlayers(tmp_path)
                gdf=None
                for l in layers:
                    gdf_try=gpd.read_file(tmp_path,driver='KML',layer=l)
                    if not gdf_try.empty: gdf=gdf_try; break
            else: gdf=None
            if gdf is not None:
                gdf_all=gpd.sjoin(gdf_all,gdf,how='left',predicate='within')
        except Exception: pass
else:
    gdf_all=df.copy()

# ---------------- Step 4: Start point ----------------
st.subheader("Step 4: Set Start Point (No GPS Needed)")
colA,colB=st.columns(2)
with colA:
    q=st.text_input("🔎 Search address / landmark")
    if st.button("Search & set start"):
        if q.strip():
            g_lat,g_lon=geocode_query(q.strip())
            if g_lat: st.session_state.origin_lat, st.session_state.origin_lon=g_lat,g_lon; st.success(f"Start set: {g_lat:.6f},{g_lon:.6f}"); st.rerun()
with colB:
    man_lat=st.text_input("Manual latitude"); man_lon=st.text_input("Manual longitude")
    if st.button("Set manual start"):
        try: st.session_state.origin_lat, st.session_state.origin_lon=float(man_lat),float(man_lon); st.success(f"Manual start: {man_lat},{man_lon}"); st.rerun()
        except: st.error("Invalid coordinates.")

# Map click
st.caption("🗺️ Click mini map to set start")
center_lat, center_lon=float(df['LATITUDE'].mean()), float(df['LONGITUDE'].mean())
mini=folium.Map(location=[center_lat,center_lon],zoom_start=12)
if st.session_state.origin_lat:
    folium.Marker([st.session_state.origin_lat,st.session_state.origin_lon],popup="Start here",icon=folium.Icon(color="green",icon="flag")).add_to(mini)
click_state=st_folium(mini,height=320,returned_objects=["last_clicked"])
if isinstance(click_state,dict):
    clicked=click_state.get("last_clicked")
    if clicked and "lat" in clicked:
        st.session_state.origin_lat,st.session_state.origin_lon=float(clicked["lat"]),float(clicked["lng"]); st.success(f"Start set from map click"); st.rerun()
origin_lat,origin_lon=st.session_state.origin_lat,st.session_state.origin_lon
st.info(f"Using start: {origin_lat},{origin_lon}" if origin_lat else "No start set")

# ---------------- Step 5: Batches ----------------
st.subheader("Step 5: Batches of 10 — View • Skip • Mark")
pool = (gdf_all.drop(columns=["geometry"]) if "geometry" in getattr(gdf_all,"columns",[]) else gdf_all).copy()
pool = pool[~pool['ISSUE ID'].astype(str).isin(st.session_state.visited_ticket_ids)]
pool = pool[~pool['ISSUE ID'].astype(str).isin(st.session_state.skipped_ticket_ids)]
st.caption(f"Remaining: {len(pool)} | Visited: {len(st.session_state.visited_ticket_ids)} | Skipped: {len(st.session_state.skipped_ticket_ids)}")

if pool.empty:
    st.info("No tickets remaining")
    batch_df=pd.DataFrame(columns=['ISSUE ID','WARD','STATUS','LATITUDE','LONGITUDE','BEFORE PHOTO'])
else:
    seed_lat,seed_lon=origin_lat,origin_lon
    batch_seq=build_nearest_neighbor_sequence(pool,seed_lat,seed_lon,10)
    batch_df=pd.DataFrame(batch_seq)
    batch_ids=batch_df['ISSUE ID'].astype(str).tolist()
    st.session_state.batch_cursor=batch_ids

# ---------------- Step 6: Show batch & upload photos ----------------
for idx,row in batch_df.iterrows():
    ticket_id=str(row['ISSUE ID'])
    st.markdown(f"### Ticket ID: {ticket_id} — {row['WARD']} — {row['STATUS']}")
    col1,col2=st.columns([1,2])
    with col1:
        if is_url(row.get("BEFORE PHOTO","")):
            st.image(row['BEFORE PHOTO'], width=150)
    with col2:
        uploaded=st.file_uploader(f"Upload after-photo (optional) {ticket_id}", type=['jpg','jpeg','png'], key=f"after_{ticket_id}")
        if uploaded:
            st.session_state.uploaded_after_photos[ticket_id] = st.session_state.uploaded_after_photos.get(ticket_id,0)+1
            st.success(f"Uploaded ({st.session_state.uploaded_after_photos[ticket_id]} total)")
            # Save JSON immediately
            save_progress(dataset_id, st.session_state.visited_ticket_ids, st.session_state.skipped_ticket_ids, st.session_state.uploaded_after_photos)
    col3,col4=st.columns(2)
    with col3:
        if st.button(f"✅ Mark Visited {ticket_id}"):
            st.session_state.visited_ticket_ids.add(ticket_id)
            save_progress(dataset_id, st.session_state.visited_ticket_ids, st.session_state.skipped_ticket_ids, st.session_state.uploaded_after_photos)
            st.experimental_rerun()
    with col4:
        if st.button(f"⏩ Skip {ticket_id}"):
            st.session_state.skipped_ticket_ids.add(ticket_id)
            save_progress(dataset_id, st.session_state.visited_ticket_ids, st.session_state.skipped_ticket_ids, st.session_state.uploaded_after_photos)
            st.experimental_rerun()

# ---------------- Step 7: Download CSV ----------------
if st.button("💾 Download progress CSV"):
    full_copy=df.copy()
    full_copy['VISITED']=full_copy['ISSUE ID'].astype(str).apply(lambda x: x in st.session_state.visited_ticket_ids)
    full_copy['SKIPPED']=full_copy['ISSUE ID'].astype(str).apply(lambda x: x in st.session_state.skipped_ticket_ids)
    full_copy['AFTER_PHOTOS_UPLOADED']=full_copy['ISSUE ID'].astype(str).apply(lambda x: st.session_state.uploaded_after_photos.get(x,0))
    tmp = io.BytesIO()
    full_copy.to_csv(tmp,index=False)
    tmp.seek(0)
    st.download_button("Download CSV", tmp, file_name="progress.csv", mime="text/csv")

st.caption("Auto-save progress locally. Photos & visited/skipped tickets persist even on reload.")

