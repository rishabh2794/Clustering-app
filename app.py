# app.py — Clustering + Batch Navigation + Skip/Mark + Clickable Image Popups
# ----------------------------------------------------------------------------
# JSON persistence for visited/skipped + photo counts
# Photos remain session-only (in memory); use ZIP download to keep them

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

# Optional GeoPandas (not mandatory now since maps removed)
try:
    import geopandas as gpd
    HAVE_GPD = True
except Exception:
    HAVE_GPD = False

from sklearn.cluster import DBSCAN

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
st.title("🗺️ Clustering + Batch Navigation (Skip/Mark • Downloads)")

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
df = df.dropna(subset=['LATITUDE','LONGITUDE'])

if df.empty:
    st.warning("No valid rows for the selected subcategory.")
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

colA, colB = st.columns(2)
with colA:
    q = st.text_input("🔎 Search address / landmark (optional)", key="txt_addr_query")
    if st.button("Search & set start", key="btn_addr_geocode"):
        try:
            url = "https://nominatim.openstreetmap.org/search"
            params = {"q": q, "format": "json", "limit": 1}
            headers = {"User-Agent": "cluster-app/1.0"}
            r = requests.get(url, params=params, headers=headers, timeout=6)
            if r.status_code == 200:
                data = r.json()
                if data:
                    g_lat, g_lon = float(data[0]["lat"]), float(data[0]["lon"])
                    st.session_state.origin_lat = g_lat
                    st.session_state.origin_lon = g_lon
                    st.success(f"Start set from search: {g_lat:.6f}, {g_lon:.6f}")
                    st.rerun()
                else:
                    st.error("No results found. Try again.")
        except Exception:
            st.error("Geocoding request failed. Try again later.")

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
            st.error("Invalid coordinates. Example: 26.8467, 80.9462")

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
    if origin_lat is None or origin_lon is None:
        seed_lat, seed_lon = float(pool['LATITUDE'].mean()), float(pool['LONGITUDE'].mean())
    else:
        seed_lat, seed_lon = origin_lat, origin_lon

    long_seq_rows = build_nearest_neighbor_sequence(pool, seed_lat, seed_lon, limit=min(200, len(pool)))
    seq_df = pd.DataFrame(long_seq_rows)

    if st.session_state.batch_cursor >= len(seq_df):
        st.session_state.batch_cursor = 0

    start = st.session_state.batch_cursor
    end = min(start + 10, len(seq_df))
    batch_df = seq_df.iloc[start:end].copy()

    if batch_df.empty and not seq_df.empty:
        st.session_state.batch_cursor = 0
        start, end = 0, min(10, len(seq_df))
        batch_df = seq_df.iloc[start:end].copy()

    batch_df_display = batch_df[['ISSUE ID','WARD','STATUS','LATITUDE','LONGITUDE','BEFORE PHOTO']].reset_index(drop=True)
    batch_df_display.index = batch_df_display.index + 1

    def as_link_or_none(x):
        s = str(x or "").strip()
        return s if is_url(s) else None
    batch_df_display['BEFORE PHOTO'] = batch_df_display['BEFORE PHOTO'].apply(as_link_or_none)

    st.dataframe(
        batch_df_display,
        use_container_width=True,
        column_config={
            'LATITUDE': st.column_config.NumberColumn('LATITUDE', format="%.6f"),
            'LONGITUDE': st.column_config.NumberColumn('LONGITUDE', format="%.6f"),
            'BEFORE PHOTO': st.column_config.LinkColumn('Before Photo', display_text='Open'),
        }
    )

    # After Photo uploads
    st.markdown("### After Photo uploads for current batch")
    from datetime import datetime
    def _save_photo_bytes(img_bytes: bytes, issue_id: str, ward: str, status: str, original_name: str, source: str):
        ts_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        saved_name = f"{issue_id}_after_{ts_str}.jpg"
        st.session_state.uploaded_after_photos.setdefault(str(issue_id), []).append({
            "bytes": img_bytes,
            "saved_name": saved_name,
            "original_name": original_name,
            "source": source,
            "ward": str(ward),
            "status": str(status),
            "ts_str": ts_str,
        })
        counts = {iid: len(items) for iid, items in st.session_state.uploaded_after_photos.items()}
        save_progress(st.session_state.dataset_id,
                      st.session_state.visited_ticket_ids,
                      st.session_state.skipped_ticket_ids,
                      counts)

    for _, row in batch_df.iterrows():
        issue_id = str(row['ISSUE ID'])
        ward     = row.get('WARD', '')
        status   = row.get('STATUS', '')

        st.markdown(f"**Issue {issue_id}** — Ward {ward}, Status: {status}")
        cam = st.camera_input(f"Take photo ({issue_id})", key=f"cam_{issue_id}")
        if cam is not None:
            _save_photo_bytes(cam.getvalue(), issue_id, ward, status, original_name="camera.jpg", source="camera")
            st.success("Captured ✅")

        up = st.file_uploader(
            f"Upload photo ({issue_id})",
            type=["jpg","jpeg","png"],
            key=f"upl_{issue_id}"
        )
        if up is not None:
            _save_photo_bytes(up.read(), issue_id, ward, status, original_name=up.name, source="upload")
            st.success("Uploaded ✅")

        if issue_id in st.session_state.uploaded_after_photos:
            cnt = len(st.session_state.uploaded_after_photos[issue_id])
            st.info(f"✅ {cnt} photo(s) saved in this session")
        elif issue_id in st.session_state.restored_photo_counts:
            cnt = st.session_state.restored_photo_counts[issue_id]
            st.info(f"✅ {cnt} photo(s) previously saved (no files in session)")
        else:
            st.warning("⚠️ No After Photo saved yet")

        st.divider()

    # ZIP download
    st.subheader("Download all clicked/uploaded After Photos")
    import zipfile, csv
    if st.session_state.uploaded_after_photos:
        rows = []
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for issue_id, items in st.session_state.uploaded_after_photos.items():
                for it in items:
                    rel_path = f"after_photos/{issue_id}/{it['saved_name']}"
                    zf.writestr(rel_path, it["bytes"])
                    rows.append([
                        issue_id,
                        it.get("ward",""),
                        it.get("status",""),
                        rel_path,
                        it.get("original_name",""),
                        it.get("source",""),
                        it.get("ts_str",""),
                    ])
            csv_io = io.StringIO()
            writer = csv.writer(csv_io)
            writer.writerow(["ISSUE_ID","WARD","STATUS","SAVED_FILENAME","ORIGINAL_NAME","SOURCE","TIMESTAMP"])
            writer.writerows(rows)
            zf.writestr("after_photos_manifest.csv", csv_io.getvalue())

        zip_buf.seek(0)
        zip_name = datetime.now().strftime("after_photos_%Y-%m-%d_%H%M.zip")
        st.download_button(
            "⬇️ Download All After Photos (ZIP + manifest)",
            data=zip_buf,
            file_name=zip_name,
            mime="application/zip",
            use_container_width=True,
        )
    else:
        st.info("No After Photos captured/uploaded in this session yet.")

    # Google Maps link
    if not batch_df.empty:
        waypoints = [(float(r['LATITUDE']), float(r['LONGITUDE'])) for _, r in batch_df.iterrows()]
        last = waypoints[-1]
        mids = waypoints[:-1] if len(waypoints) > 1 else []
        nav_url = google_maps_url(origin_lat, origin_lon, last[0], last[1], waypoints=mids)
        st.markdown(f"[🧭 Open route in Google Maps for this batch]({nav_url})")

    # Actions
    c1, c2, c3, c4 = st.columns(4)
    first_id = str(batch_df.iloc[0]['ISSUE ID']) if not batch_df.empty else None

    def _persist_counts_now():
        counts = {iid: len(items) for iid, items in st.session_state.uploaded_after_photos.items()}
        save_progress(st.session_state.dataset_id,
                      st.session_state.visited_ticket_ids,
                      st.session_state.skipped_ticket_ids,
                      counts)

    with c1:
        if st.button("✅ Mark first as Visited", key="btn
