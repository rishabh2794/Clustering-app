# app.py — Clustering + Batch Navigation + Skip/Mark + Clickable Image Popups
# Cleaned: Removed Ward Overlay + Excel/HTML Map Downloads

import math
import io
import json
import hashlib
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import DBSCAN
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from html import escape
import requests
import zipfile, csv

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

# ----------------- App Setup -----------------
st.set_page_config(layout="wide", page_title="Clustering + Batch Navigation")
st.title("🗺️ Clustering + Batch Navigation (Skip/Mark • Map Click • After Photos)")

# Session state
for k in ["visited_ticket_ids","skipped_ticket_ids"]:
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
csv_file = st.file_uploader("Upload CSV", type=["csv"])
subcategory_option = st.selectbox(
    "Issue Subcategory",
    [
        "Pothole","Garbage dumped on public land","Unpaved Road","Broken Footpath / Divider",
        "Sand piled on roadsides + Mud/slit on roadside","Malba, bricks, bori, etc dumped on public land",
        "Construction/ demolition activity without safeguards","Encroachment-Building Materials Dumped on Road",
        "Burning of garbage, plastic, leaves, branches etc.","Overflowing Dustbins",
        "Barren land to be greened","Greening of Central Verges","Unsurfaced Parking Lots"
    ]
)
radius_m = st.number_input("Clustering radius (m)", 1, 1000, 15)
min_samples = st.number_input("Minimum per cluster", 1, 100, 2)

# ----------------- Main -----------------
if not csv_file:
    st.info("Upload the required CSV to proceed.")
    st.stop()

csv_bytes = csv_file.getvalue()
dataset_id = dataset_fingerprint(csv_bytes)
st.session_state.dataset_id = dataset_id
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
df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
df = df.dropna(subset=['LATITUDE','LONGITUDE'])
if df.empty:
    st.warning("No valid rows for the selected subcategory.")
    st.stop()

coords_deg = df[['LATITUDE','LONGITUDE']].to_numpy(dtype=float)
coords_rad = np.radians(coords_deg)
eps_rad = float(radius_m)/EARTH_RADIUS_M
db = DBSCAN(eps=eps_rad, min_samples=int(min_samples), metric='haversine', algorithm='ball_tree')
labels = db.fit_predict(coords_rad)
df['CLUSTER NUMBER'] = labels
df['IS_CLUSTERED'] = df['CLUSTER NUMBER'] != -1

# ---------------- Start Point ----------------
st.subheader("Step 4: Set Start Point")
colA, colB = st.columns(2)
with colA:
    q = st.text_input("🔎 Search address / landmark (optional)")
    if st.button("Search & set start"):
        if q.strip():
            try:
                r = requests.get("https://nominatim.openstreetmap.org/search",
                                 params={"q": q, "format": "json", "limit": 1},
                                 headers={"User-Agent": "cluster-app/1.0"},
                                 timeout=6)
                data = r.json()
                if data:
                    st.session_state.origin_lat = float(data[0]["lat"])
                    st.session_state.origin_lon = float(data[0]["lon"])
                    st.rerun()
            except Exception:
                st.error("Could not geocode that query.")
with colB:
    man_lat = st.text_input("Manual latitude")
    man_lon = st.text_input("Manual longitude")
    if st.button("Set manual start"):
        try:
            st.session_state.origin_lat = float(man_lat)
            st.session_state.origin_lon = float(man_lon)
            st.rerun()
        except Exception:
            st.error("Invalid coordinates.")

center_lat, center_lon = float(df['LATITUDE'].mean()), float(df['LONGITUDE'].mean())
mini = folium.Map(location=[center_lat, center_lon], zoom_start=12)
if st.session_state.origin_lat is not None and st.session_state.origin_lon is not None:
    folium.Marker([st.session_state.origin_lat, st.session_state.origin_lon],
                  popup="Start here", icon=folium.Icon(color="green", icon="flag")).add_to(mini)
click_state = st_folium(mini, height=320, returned_objects=["last_clicked"])
if isinstance(click_state, dict) and click_state.get("last_clicked"):
    st.session_state.origin_lat = float(click_state["last_clicked"]["lat"])
    st.session_state.origin_lon = float(click_state["last_clicked"]["lng"])
    st.rerun()

origin_lat, origin_lon = st.session_state.origin_lat, st.session_state.origin_lon

# ---------------- Batch + Skip/Mark ----------------
st.subheader("Step 5: Batches of 10 — View • Skip • Mark")
pool = df.copy()
pool = pool[~pool['ISSUE ID'].astype(str).isin(st.session_state.visited_ticket_ids)]
pool = pool[~pool['ISSUE ID'].astype(str).isin(st.session_state.skipped_ticket_ids)]
if pool.empty:
    st.info("No tickets remaining.")
    batch_df = pd.DataFrame()
else:
    if origin_lat is None or origin_lon is None:
        seed_lat, seed_lon = float(pool['LATITUDE'].mean()), float(pool['LONGITUDE'].mean())
    else:
        seed_lat, seed_lon = origin_lat, origin_lon
    long_seq_rows = build_nearest_neighbor_sequence(pool, seed_lat, seed_lon, limit=min(200,len(pool)))
    seq_df = pd.DataFrame(long_seq_rows)
    start = st.session_state.batch_cursor
    end = min(start+10, len(seq_df))
    batch_df = seq_df.iloc[start:end].copy()
    st.dataframe(batch_df[['ISSUE ID','WARD','STATUS','LATITUDE','LONGITUDE','BEFORE PHOTO']])

# ---------------- After Photo Uploads ----------------
if not batch_df.empty:
    st.subheader("After Photo uploads")
    def _save_photo_bytes(img_bytes, issue_id, ward, status, original_name, source):
        ts_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        st.session_state.uploaded_after_photos.setdefault(str(issue_id), []).append({
            "bytes": img_bytes,
            "saved_name": f"{issue_id}_after_{ts_str}.jpg",
            "original_name": original_name,
            "source": source,
            "ward": str(ward),
            "status": str(status),
            "ts_str": ts_str,
        })
        counts = {iid: len(items) for iid,items in st.session_state.uploaded_after_photos.items()}
        save_progress(st.session_state.dataset_id,
                      st.session_state.visited_ticket_ids,
                      st.session_state.skipped_ticket_ids,
                      counts)

    for _, row in batch_df.iterrows():
        issue_id, ward, status = str(row['ISSUE ID']), row.get('WARD',''), row.get('STATUS','')
        cam = st.camera_input(f"Take photo ({issue_id})")
        if cam is not None:
            _save_photo_bytes(cam.getvalue(), issue_id, ward, status, "camera.jpg", "camera")
        up = st.file_uploader(f"Upload photo ({issue_id})", type=["jpg","jpeg","png"])
        if up is not None:
            _save_photo_bytes(up.read(), issue_id, ward, status, up.name, "upload")

# ---------------- ZIP Download ----------------
if st.session_state.uploaded_after_photos:
    rows, zip_buf = [], io.BytesIO()
    with zipfile.ZipFile(zip_buf,"w",compression=zipfile.ZIP_DEFLATED) as zf:
        for issue_id, items in st.session_state.uploaded_after_photos.items():
            for it in items:
                rel_path = f"after_photos/{issue_id}/{it['saved_name']}"
                zf.writestr(rel_path, it["bytes"])
                rows.append([issue_id,it.get("ward",""),it.get("status",""),
                             rel_path,it.get("original_name",""),
                             it.get("source",""),it.get("ts_str","")])
        csv_io = io.StringIO()
        writer = csv.writer(csv_io)
        writer.writerow(["ISSUE_ID","WARD","STATUS","SAVED_FILENAME","ORIGINAL_NAME","SOURCE","TIMESTAMP"])
        writer.writerows(rows)
        zf.writestr("after_photos_manifest.csv", csv_io.getvalue())
    zip_buf.seek(0)
    st.download_button("⬇️ Download All After Photos (ZIP + manifest)",
                       data=zip_buf, file_name="after_photos.zip", mime="application/zip")

# ---------------- Map ----------------
st.subheader("Step 6: Map")
m = folium.Map(location=[df['LATITUDE'].mean(), df['LONGITUDE'].mean()], zoom_start=12)
mc = MarkerCluster().add_to(m)
batch_ids = set(batch_df['ISSUE ID'].astype(str).tolist()) if not batch_df.empty else set()
for _, row in df.iterrows():
    rid, lat, lon = str(row['ISSUE ID']), float(row['LATITUDE']), float(row['LONGITUDE'])
    color = 'red'
    if rid in batch_ids: color='orange'
    if rid in st.session_state.visited_ticket_ids: color='gray'
    if rid in st.session_state.skipped_ticket_ids: color='purple'
    folium.CircleMarker([lat,lon],radius=6,color=color,fill=True,fill_color=color,
                        popup=f"Issue {rid}").add_to(mc)
st_folium(m, height=500, use_container_width=True)
