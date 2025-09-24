# app.py — Minimal Clustering + Batch Navigation + Photo Auto-save
import io, json, hashlib
from pathlib import Path
from datetime import datetime
import math
import zipfile, csv

import pandas as pd
import numpy as np
import streamlit as st
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from sklearn.cluster import DBSCAN
from openpyxl import Workbook

# ----------------- Constants -----------------
REQUIRED_COLS = {'ISSUE ID','SUBCATEGORY','STATUS','LATITUDE','LONGITUDE','BEFORE PHOTO'}
PROGRESS_ROOT = Path("./progress")
PROGRESS_ROOT.mkdir(exist_ok=True)

EARTH_RADIUS_M = 6_371_000.0

# ----------------- Helpers -----------------
def dataset_fingerprint(file_bytes: bytes) -> str:
    return hashlib.sha1(file_bytes).hexdigest()

def progress_path(dataset_id: str) -> Path:
    return PROGRESS_ROOT / f"{dataset_id}.json"

def load_progress(dataset_id: str):
    p = progress_path(dataset_id)
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        visited = set(data.get("visited_ticket_ids", []))
        skipped = set(data.get("skipped_ticket_ids", []))
        photo_counts = data.get("uploaded_after_photos", {})
        return visited, skipped, photo_counts
    return set(), set(), {}

def save_progress(dataset_id: str, visited, skipped, photo_counts):
    p = progress_path(dataset_id)
    payload = {
        "visited_ticket_ids": list(visited),
        "skipped_ticket_ids": list(skipped),
        "uploaded_after_photos": photo_counts,
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def haversine_m(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2)**2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))

def build_nearest_neighbor_sequence(df, start_lat, start_lon, limit=200):
    seq = []
    pool = df.copy()
    if pool.empty:
        return seq
    cur_lat, cur_lon = start_lat, start_lon
    for _ in range(min(limit,len(pool))):
        pool['__dist'] = pool.apply(lambda r: haversine_m(cur_lat,cur_lon,r['LATITUDE'],r['LONGITUDE']),axis=1)
        nxt = pool.sort_values('__dist').iloc[0]
        seq.append(nxt)
        cur_lat, cur_lon = nxt['LATITUDE'], nxt['LONGITUDE']
        pool = pool[pool['ISSUE ID'] != nxt['ISSUE ID']]
    return seq

def is_url(u):
    return isinstance(u,str) and u.startswith(("http://","https://"))

# ----------------- Streamlit setup -----------------
st.set_page_config(layout="wide", page_title="Clustering + Batch Navigation")
st.title("🗺️ Clustering + Batch Navigation (Minimal Version)")

# ----------------- Session state -----------------
for k in ["visited_ticket_ids","skipped_ticket_ids","uploaded_after_photos"]:
    if k not in st.session_state:
        st.session_state[k] = {} if k=="uploaded_after_photos" else set()
if "batch_cursor" not in st.session_state:
    st.session_state.batch_cursor = 0
if "dataset_id" not in st.session_state:
    st.session_state.dataset_id = None
if "origin_lat" not in st.session_state:
    st.session_state.origin_lat = None
if "origin_lon" not in st.session_state:
    st.session_state.origin_lon = None

# ----------------- Upload CSV -----------------
csv_file = st.file_uploader("Upload CSV", type=["csv"])
if not csv_file:
    st.stop()

csv_bytes = csv_file.getvalue()
dataset_id = dataset_fingerprint(csv_bytes)
st.session_state.dataset_id = dataset_id
st.caption(f"Dataset ID: {dataset_id[:8]}…")

df = pd.read_csv(io.BytesIO(csv_bytes))
missing = REQUIRED_COLS - set(df.columns)
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

# Convert lat/lon
df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
df.dropna(subset=['LATITUDE','LONGITUDE'], inplace=True)

# ----------------- Restore progress -----------------
visited, skipped, restored_counts = load_progress(dataset_id)
st.session_state.visited_ticket_ids |= visited
st.session_state.skipped_ticket_ids |= skipped
# restore counts
for k,v in restored_counts.items():
    if k not in st.session_state.uploaded_after_photos:
        st.session_state.uploaded_after_photos[k] = [None]*v  # placeholders

# ----------------- Clustering -----------------
radius_m = st.number_input("Clustering radius (m)",1,1000,15)
min_samples = st.number_input("Min per cluster",1,100,2)

coords_rad = np.radians(df[['LATITUDE','LONGITUDE']].to_numpy())
db = DBSCAN(eps=radius_m/EARTH_RADIUS_M,min_samples=int(min_samples),metric='haversine',algorithm='ball_tree')
df['CLUSTER NUMBER'] = db.fit_predict(coords_rad)
df['IS_CLUSTERED'] = df['CLUSTER NUMBER']!=-1

# ----------------- Start point -----------------
st.subheader("Step 4: Set Start Point (Optional)")
col1,col2 = st.columns(2)
with col1:
    lat_in = st.text_input("Latitude")
    lon_in = st.text_input("Longitude")
    if st.button("Set manual start"):
        try:
            st.session_state.origin_lat = float(lat_in)
            st.session_state.origin_lon = float(lon_in)
            st.success(f"Start set: {st.session_state.origin_lat},{st.session_state.origin_lon}")
        except:
            st.error("Invalid coordinates")
origin_lat = st.session_state.origin_lat or df['LATITUDE'].mean()
origin_lon = st.session_state.origin_lon or df['LONGITUDE'].mean()

# ----------------- Batch navigation -----------------
st.subheader("Step 5: Batch Navigation")

pool = df[~df['ISSUE ID'].astype(str).isin(st.session_state.visited_ticket_ids)]
pool = pool[~pool['ISSUE ID'].astype(str).isin(st.session_state.skipped_ticket_ids)]

st.caption(f"Remaining tickets: {len(pool)} | Visited: {len(st.session_state.visited_ticket_ids)} | Skipped: {len(st.session_state.skipped_ticket_ids)}")

if pool.empty:
    st.info("No tickets remaining")
    batch_df = pd.DataFrame(columns=df.columns)
else:
    seq_rows = build_nearest_neighbor_sequence(pool, origin_lat, origin_lon, limit=200)
    seq_df = pd.DataFrame(seq_rows)
    start = st.session_state.batch_cursor
    end = min(start+10,len(seq_df))
    batch_df = seq_df.iloc[start:end]

    st.dataframe(batch_df[['ISSUE ID','STATUS','LATITUDE','LONGITUDE','BEFORE PHOTO']])

    # ----------------- Photo uploads -----------------
    st.markdown("### After Photos")
    for _, row in batch_df.iterrows():
        issue_id = str(row['ISSUE ID'])
        ward = row.get('WARD','')
        status = row.get('STATUS','')

        cam = st.camera_input(f"Take photo ({issue_id})", key=f"cam_{issue_id}")
        if cam:
            st.session_state.uploaded_after_photos.setdefault(issue_id, []).append(cam.getvalue())

        up = st.file_uploader(f"Upload photo ({issue_id})",type=["jpg","jpeg","png"], key=f"upl_{issue_id}")
        if up:
            st.session_state.uploaded_after_photos.setdefault(issue_id, []).append(up.read())

        cnt = len(st.session_state.uploaded_after_photos.get(issue_id,[]))
        st.info(f"{cnt} photo(s) saved in this session")

    # ----------------- Batch actions -----------------
    c1,c2,c3,c4 = st.columns(4)
    first_id = str(batch_df.iloc[0]['ISSUE ID']) if not batch_df.empty else None

    def persist_counts():
        counts = {iid: len(v) for iid,v in st.session_state.uploaded_after_photos.items()}
        save_progress(st.session_state.dataset_id, st.session_state.visited_ticket_ids, st.session_state.skipped_ticket_ids, counts)

    with c1:
        if st.button("✅ Mark first visited"):
            if first_id:
                st.session_state.visited_ticket_ids.add(first_id)
                persist_counts()
                st.experimental_rerun()
    with c2:
        if st.button("⏭️ Skip first"):
            if first_id:
                st.session_state.skipped_ticket_ids.add(first_id)
                persist_counts()
                st.experimental_rerun()
    with c3:
        if st.button("✅ Mark batch visited"):
            for iid in batch_df['ISSUE ID'].astype(str):
                st.session_state.visited_ticket_ids.add(iid)
            persist_counts()
            st.experimental_rerun()
    with c4:
        if st.button("➡️ Next batch"):
            st.session_state.batch_cursor = end if end < len(seq_df) else 0
            st.experimental_rerun()

# ----------------- Map -----------------
st.subheader("Step 6: Map")
m = folium.Map(location=[origin_lat,origin_lon],zoom_start=12)
mc = MarkerCluster().add_to(m)
batch_ids = set(batch_df['ISSUE ID'].astype(str)) if not batch_df.empty else set()
first_in_batch = first_id

for _,row in df.iterrows():
    rid = str(row['ISSUE ID'])
    lat, lon = row['LATITUDE'], row['LONGITUDE']
    if rid==first_in_batch: color='green'
    elif rid in batch_ids: color='orange'
    elif rid in st.session_state.visited_ticket_ids: color='gray'
    elif rid in st.session_state.skipped_ticket_ids: color='purple'
    else: color='red'
    popup = folium.Popup(f"Issue: {rid}<br>Status: {row['STATUS']}", max_width=250)
    folium.CircleMarker([lat,lon],radius=7,color=color,fill=True,fill_color=color,popup=popup).add_to(mc)

m.save("map.html")
with open("map.html","rb") as f:
    st.download_button("Download Map",f,"map.html")

# ----------------- Excel summary -----------------
st.subheader("Step 7: Downloads")
if not df.empty:
    excel_buf = io.BytesIO()
    df.to_excel(excel_buf, index=False)
    excel_buf.seek(0)
    st.download_button("Download Excel",excel_buf,"summary.xlsx")

# ----------------- Download photos -----------------
if st.session_state.uploaded_after_photos:
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf,"w",compression=zipfile.ZIP_DEFLATED) as zf:
        manifest_rows=[]
        for iid,photos in st.session_state.uploaded_after_photos.items():
            for idx, b in enumerate(photos,1):
                fname=f"{iid}_after_{idx}.jpg"
                zf.writestr(f"after_photos/{fname}", b)
                manifest_rows.append([iid,fname])
        csv_io=io.StringIO()
        writer=csv.writer(csv_io)
        writer.writerow(["ISSUE_ID","FILENAME"])
        writer.writerows(manifest_rows)
        zf.writestr("after_photos/manifest.csv", csv_io.getvalue())
    zip_buf.seek(0)
    st.download_button("Download After Photos ZIP",zip_buf,"after_photos.zip")
