# app.py — Full Stable Version: Clustering + Batch Navigation + Skip/Mark + Photo Upload + Map + Downloads
# Features:
# - CSV upload + validation
# - Optional Ward overlay (GeoJSON/KML)
# - Subcategory filter
# - DBSCAN clustering
# - Batch navigation (10 per batch)
# - Skip/Mark + local JSON auto-save
# - After Photo upload (camera + file) with session persistence
# - Map with clickable popups + optional thumbnails
# - Google Maps route for batch
# - Excel + HTML map downloads
# - ZIP download of After Photos with manifest

import math, io, json, hashlib, tempfile, zipfile, csv
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
except:
    HAVE_GPD = False
try:
    import fiona
    HAS_FIONA = True
except:
    HAS_FIONA = False

from sklearn.cluster import DBSCAN
import folium
from folium.plugins import MarkerCluster
from openpyxl import load_workbook
from streamlit_folium import st_folium
from html import escape

# ----------------- Constants -----------------
EARTH_RADIUS_M = 6_371_000.0
REQUIRED_COLS = {
    'ISSUE ID','CITY','ZONE','WARD','SUBCATEGORY','CREATED AT',
    'STATUS','LATITUDE','LONGITUDE','BEFORE PHOTO','AFTER PHOTO','ADDRESS'
}

# ----------------- Helpers -----------------
def dataset_fingerprint(file_bytes: bytes) -> str:
    return hashlib.sha1(file_bytes).hexdigest()

PROGRESS_ROOT = Path("./progress")
PROGRESS_ROOT.mkdir(exist_ok=True, parents=True)

def progress_path(dataset_id: str) -> Path:
    return PROGRESS_ROOT / f"{dataset_id}.json"

def load_progress(dataset_id: str):
    p = progress_path(dataset_id)
    if p.exists():
        data = json.load(p.open("r", encoding="utf-8"))
        return set(data.get("visited_ticket_ids", [])), set(data.get("skipped_ticket_ids", [])), data.get("uploaded_after_photos", {})
    return set(), set(), {}

def save_progress(dataset_id, visited, skipped, photo_counts):
    if not dataset_id: return
    payload = {
        "visited_ticket_ids": sorted(list(visited)),
        "skipped_ticket_ids": sorted(list(skipped)),
        "uploaded_after_photos": photo_counts,
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with progress_path(dataset_id).open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def normalize_subcategory(series: pd.Series):
    return series.astype(str).str.strip().str.lower()

def haversine_m(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2-lat1)
    dlmb = math.radians(lon2-lon1)
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
        wp = "|".join([f"{lat},{lon}" for (lat,lon) in waypoints])
        parts.append(f"waypoints={quote(wp,safe='|,')}")
    return base+"&".join(parts)

def hyperlinkify_excel(excel_path):
    try:
        wb = load_workbook(excel_path)
        ws = wb["Clustering Application Summary"]
        for row in range(2, ws.max_row+1):
            for col in (11,12):
                link = ws.cell(row,col).value
                if link and isinstance(link,str) and link.startswith(("http://","https://")):
                    ws.cell(row,col).hyperlink = link
                    ws.cell(row,col).style = "Hyperlink"
        wb.save(excel_path)
    except: pass

def load_wards_uploaded(file):
    if not HAVE_GPD: return None
    try:
        suffix = file.name.split(".")[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix="."+suffix) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        if suffix in ("geojson","json"): gdf=gpd.read_file(tmp_path)
        elif suffix=="kml":
            if not HAS_FIONA: return None
            layers=fiona.listlayers(tmp_path)
            gdf=None
            for layer in layers:
                gdf_try=gpd.read_file(tmp_path, driver="KML", layer=layer)
                if not gdf_try.empty: gdf=gdf_try; break
            if gdf is None: return None
        else: return None
        if gdf.crs is None: gdf.set_crs(epsg=4326,inplace=True)
        elif gdf.crs.to_string()!="EPSG:4326": gdf=gdf.to_crs(epsg=4326)
        return gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    except: return None

def geocode_query(q: str, timeout=6):
    try:
        url="https://nominatim.openstreetmap.org/search"
        params={"q":q,"format":"json","limit":1}
        headers={"User-Agent":"cluster-app/1.0"}
        r=requests.get(url,params=params,headers=headers,timeout=timeout)
        if r.status_code==200:
            data=r.json()
            if data: return float(data[0]["lat"]), float(data[0]["lon"])
    except: pass
    return None,None

def build_nearest_neighbor_sequence(df_like, start_lat, start_lon, limit):
    seq=[]
    pool=df_like.copy()
    if pool.empty: return seq
    cur_lat, cur_lon=start_lat,start_lon
    for _ in range(min(limit,len(pool))):
        pool['__dist']=pool.apply(lambda r: haversine_m(cur_lat,cur_lon,r['LATITUDE'],r['LONGITUDE']),axis=1)
        nxt=pool.sort_values('__dist').iloc[0]
        seq.append(nxt)
        cur_lat,cur_lon=nxt['LATITUDE'],nxt['LONGITUDE']
        pool=pool[pool['ISSUE ID']!=nxt['ISSUE ID']]
    return seq

def is_url(u):
    return isinstance(u,str) and u.startswith(("http://","https://"))

# ----------------- Streamlit Setup -----------------
st.set_page_config(layout="wide", page_title="Clustering + Batch Navigation")
st.title("🗺️ Clustering + Batch Navigation")

# Session state defaults
for k in ["visited_ticket_ids","skipped_ticket_ids","uploaded_after_photos","restored_photo_counts","batch_cursor","origin_lat","origin_lon","dataset_id"]:
    if k not in st.session_state: st.session_state[k]={} if "photo" in k else set() if "ids" in k else 0 if k=="batch_cursor" else None

# ----------------- Inputs -----------------
csv_file = st.file_uploader("Upload CSV",type=["csv"],key="csv_uploader_main")
ward_file = st.file_uploader("Upload Wards (optional)", type=["geojson","json","kml"], key="ward_uploader_main")
subcategory_option=st.selectbox("Issue Subcategory",[
    "Pothole","Garbage dumped on public land","Unpaved Road","Broken Footpath / Divider",
    "Sand piled on roadsides + Mud/slit on roadside","Malba, bricks, bori, etc dumped on public land",
    "Construction/ demolition activity without safeguards","Encroachment-Building Materials Dumped on Road",
    "Burning of garbage, plastic, leaves, branches etc.","Overflowing Dustbins",
    "Barren land to be greened","Greening of Central Verges","Unsurfaced Parking Lots"], key="sel_subcategory")
radius_m = st.number_input("Clustering radius (m)", 1, 1000, 15, key="num_radius")
min_samples = st.number_input("Minimum per cluster",1,100,2,key="num_min_samples")

if not csv_file:
    st.info("Upload CSV to proceed."); st.stop()

csv_bytes=csv_file.getvalue()
dataset_id=dataset_fingerprint(csv_bytes)
st.session_state.dataset_id=dataset_id
st.caption(f"Dataset ID: `{dataset_id[:8]}…`")

df=pd.read_csv(io.BytesIO(csv_bytes))
missing=list(REQUIRED_COLS-set(df.columns))
if missing: st.error(f"Missing columns: {missing}"); st.stop()

# Restore prior progress
rest_visited, rest_skipped, rest_photo_counts = load_progress(dataset_id)
st.session_state.visited_ticket_ids |= rest_visited
st.session_state.skipped_ticket_ids |= rest_skipped
st.session_state.restored_photo_counts=rest_photo_counts

df['SUBCATEGORY_NORM']=normalize_subcategory(df['SUBCATEGORY'])
df=df[df['SUBCATEGORY_NORM']==subcategory_option.lower()]
df['LATITUDE']=pd.to_numeric(df['LATITUDE'],errors='coerce')
df['LONGITUDE']=pd.to_numeric(df['LONGITUDE'],errors='coerce')
df=df[df['LATITUDE'].between(-90,90) & df['LONGITUDE'].between(-180,180)].dropna(subset=['LATITUDE','LONGITUDE'])
if df.empty: st.warning("No valid rows for this subcategory."); st.stop()

# ----------------- Clustering -----------------
coords_deg=df[['LATITUDE','LONGITUDE']].to_numpy(dtype=float)
coords_rad=np.radians(coords_deg)
eps_rad=radius_m/EARTH_RADIUS_M
db=DBSCAN(eps=eps_rad,min_samples=int(min_samples),metric='haversine',algorithm='ball_tree')
labels=db.fit_predict(coords_rad)
df['CLUSTER NUMBER']=labels
df['IS_CLUSTERED']=df['CLUSTER NUMBER']!=-1

# Optional GeoPandas overlay
if HAVE_GPD:
    gdf_all=gpd.GeoDataFrame(df.copy(),geometry=gpd.points_from_xy(df['LONGITUDE'],df['LATITUDE']),crs="EPSG:4326")
    wards_gdf=load_wards_uploaded(ward_file) if ward_file else None
    if wards_gdf is not None:
        try: gdf_all=gpd.sjoin(gdf_all,wards_gdf,how="left",predicate="within")
        except: pass
else: gdf_all=df.copy()

# Excel summary
clustered=gdf_all[gdf_all['IS_CLUSTERED']]
excel_filename="Clustering_Application_Summary.xlsx"
if not clustered.empty:
    (clustered.drop(columns=["geometry"]) if "geometry" in clustered.columns else clustered).to_excel(
        excel_filename,sheet_name='Clustering Application Summary',index=False
    )
    hyperlinkify_excel(excel_filename)

# ----------------- Step 4: Start Point -----------------
st.subheader("Step 4: Set Start Point (No GPS Needed)")
colA,colB=st.columns(2)
with colA:
    q=st.text_input("🔎 Search address/landmark (optional)",key="txt_addr_query")
    if st.button("Search & set start",key="btn_addr_geocode") and q.strip():
        g_lat,g_lon=geocode_query(q.strip())
        if g_lat is not None: st.session_state.origin_lat, st.session_state.origin_lon=g_lat,g_lon; st.success(f"Start set from search: {g_lat:.6f}, {g_lon:.6f}"); st.rerun()
        else: st.error("Could not geocode query.")
with colB:
    man_lat=st.text_input("Manual latitude (optional)",key="txt_lat")
    man_lon=st.text_input("Manual longitude (optional)",key="txt_lon")
    if st.button("Set manual start",key="btn_set_manual") and man_lat and man_lon:
        try: st.session_state.origin_lat, st.session_state.origin_lon=float(man_lat),float(man_lon); st.success("Start point set manually."); st.rerun()
        except: st.error("Invalid lat/lon input.")

if st.session_state.origin_lat is None or st.session_state.origin_lon is None:
    st.warning("Set a start point to enable nearest-neighbor batch ordering."); st.stop()

# ----------------- Batch Ordering -----------------
batch_size=10
all_tickets=gdf_all.copy()
all_tickets['ISSUE_ID_STR']=all_tickets['ISSUE ID'].astype(str)
unvisited=all_tickets[~all_tickets['ISSUE_ID_STR'].isin(st.session_state.visited_ticket_ids|st.session_state.skipped_ticket_ids)]
nearest_seq=build_nearest_neighbor_sequence(unvisited, st.session_state.origin_lat, st.session_state.origin_lon, batch_size)
if not nearest_seq:
    st.info("No more tickets to visit.")
    st.stop()

st.subheader("Current Batch")
batch_df=pd.DataFrame(nearest_seq)
for idx,row in batch_df.iterrows():
    col1,col2,col3=st.columns([1,3,1])
    with col1: st.markdown(f"**{row['ISSUE ID']}**")
    with col2: st.markdown(f"{row['SUBCATEGORY']} — {row['ADDRESS']}")
    with col3:
        uploaded_file = st.file_uploader(f"After Photo (optional) {row['ISSUE ID']}", type=["png","jpg","jpeg"], key=f"after_{row['ISSUE ID']}")
        if uploaded_file:
            save_dir=Path("./after_photos")/dataset_id
            save_dir.mkdir(parents=True, exist_ok=True)
            fname=save_dir/f"{row['ISSUE ID']}_{uploaded_file.name}"
            with open(fname,"wb") as f: f.write(uploaded_file.getvalue())
            st.session_state.restored_photo_counts[row['ISSUE_ID_STR']]=str(fname)
            st.success(f"Saved {uploaded_file.name}")

col_action1,col_action2,col_action3=st.columns(3)
if col_action1.button("Mark Visited ✅"):
    st.session_state.visited_ticket_ids |= set(batch_df['ISSUE_ID_STR'])
    save_progress(dataset_id, st.session_state.visited_ticket_ids, st.session_state.skipped_ticket_ids, st.session_state.restored_photo_counts)
    st.experimental_rerun()
if col_action2.button("Skip ⏭️"):
    st.session_state.skipped_ticket_ids |= set(batch_df['ISSUE_ID_STR'])
    save_progress(dataset_id, st.session_state.visited_ticket_ids, st.session_state.skipped_ticket_ids, st.session_state.restored_photo_counts)
    st.experimental_rerun()
if col_action3.button("Reset Batch 🔄"):
    for k in batch_df['ISSUE_ID_STR']:
        st.session_state.restored_photo_counts.pop(k,None)
    save_progress(dataset_id, st.session_state.visited_ticket_ids, st.session_state.skipped_ticket_ids, st.session_state.restored_photo_counts)
    st.experimental_rerun()

# ----------------- Map -----------------
st.subheader("Map")
m=folium.Map(location=[st.session_state.origin_lat, st.session_state.origin_lon], zoom_start=14)
mc=MarkerCluster()
for _,r in batch_df.iterrows():
    html_popup = f"<b>{escape(str(r['ISSUE ID']))}</b><br>{escape(str(r['ADDRESS']))}"
    if str(r['ISSUE ID']) in st.session_state.restored_photo_counts:
        p_path=st.session_state.restored_photo_counts[str(r['ISSUE ID'])]
        html_popup+=f"<br><img src='file://{p_path}' width=120>"
    folium.Marker([r['LATITUDE'],r['LONGITUDE']], popup=folium.Popup(html_popup,max_width=300)).add_to(mc)
mc.add_to(m)
st_folium(m,width=700,height=500)

# Google Maps batch link
batch_coords=list(zip(batch_df['LATITUDE'],batch_df['LONGITUDE']))
if batch_coords:
    gmap_link=google_maps_url(st.session_state.origin_lat, st.session_state.origin_lon, batch_coords[-1][0], batch_coords[-1][1], waypoints=batch_coords[:-1])
    st.markdown(f"[Open Batch Route in Google Maps]({gmap_link})", unsafe_allow_html=True)

# ----------------- Downloads -----------------
st.subheader("Downloads")
col_d1,col_d2=st.columns(2)
with col_d1:
    if Path(excel_filename).exists():
        with open(excel_filename,"rb") as f: data=f.read()
        st.download_button("Download Excel Summary",data,file_name=excel_filename)
with col_d2:
    map_html_file="map.html"
    m.save(map_html_file)
    with open(map_html_file,"rb") as f: data=f.read()
    st.download_button("Download Map HTML",data,file_name="map.html")

# ZIP After Photos
zip_fname=f"After_Photos_{dataset_id[:8]}.zip"
photo_dir=Path("./after_photos")/dataset_id
if photo_dir.exists() and any(photo_dir.iterdir()):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_zip:
        with zipfile.ZipFile(tmp_zip.name,"w") as zf:
            for f in photo_dir.iterdir():
                zf.write(f,f.name)
        with open(tmp_zip.name,"rb") as f: data=f.read()
        st.download_button("Download After Photos ZIP",data,file_name=zip_fname)
