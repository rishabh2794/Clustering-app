# app.py — Clustering + Batch Navigation + Skip/Mark + Clickable Image Popups
# ----------------------------------------------------------------------------

import math
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

# Optional geolocation helpers
try:
    from streamlit_geolocation import st_geolocation  # component
    HAVE_GEO = True
except Exception:
    HAVE_GEO = False

try:
    from streamlit_js_eval import streamlit_js_eval       # JS fallback
    HAVE_JS = True
except Exception:
    HAVE_JS = False

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
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb/2)**2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))

def google_maps_url(origin_lat, origin_lon, dest_lat, dest_lon, mode="driving", waypoints=None):
    base = "https://www.google.com/maps/dir/?api=1"
    parts = []
    # If origin is missing, omit it — Maps will start from live GPS on the device
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
    """Ward file reader with KML guard. Returns GeoDataFrame or None."""
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

def get_ip_location():
    try:
        r = requests.get("https://ipapi.co/json/", timeout=5)
        if r.status_code == 200:
            data = r.json()
            return float(data["latitude"]), float(data["longitude"])
    except Exception:
        pass
    return None, None

def build_nearest_neighbor_sequence(df_like: pd.DataFrame, start_lat: float, start_lon: float, limit: int):
    """Greedy NN sequence from a start; returns list of rows (dict-like)."""
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
st.title("🗺️ Clustering + Batch Navigation (Skip/Mark • Image Popups • Origin optional)")

# Session state
for k in ["visited_ticket_ids", "skipped_ticket_ids"]:
    if k not in st.session_state:
        st.session_state[k] = set()
if "batch_cursor" not in st.session_state:
    st.session_state.batch_cursor = 0  # pagination by 10
if "origin_lat" not in st.session_state:
    st.session_state.origin_lat = None
if "origin_lon" not in st.session_state:
    st.session_state.origin_lon = None

# ----------------- Inputs (unique keys) -----------------
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

# ----------------- Main -----------------
if not csv_file:
    st.info("Upload the required CSV to proceed.")
    st.stop()

# Data load + filter
df = pd.read_csv(csv_file)
missing = list(REQUIRED_COLS - set(df.columns))
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

df['SUBCATEGORY_NORM'] = normalize_subcategory(df['SUBCATEGORY'])
df = df[df['SUBCATEGORY_NORM'] == subcategory_option.lower()].copy()
df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
df.dropna(subset=['LATITUDE','LONGITUDE'], inplace=True)

# Clustering
coords_rad = np.radians(df[['LATITUDE','LONGITUDE']])
eps_rad = float(radius_m) / EARTH_RADIUS_M
db = DBSCAN(eps=eps_rad, min_samples=int(min_samples), metric='haversine', algorithm='ball_tree')
df['CLUSTER NUMBER'] = db.fit_predict(coords_rad)
df['IS_CLUSTERED'] = df['CLUSTER NUMBER'] != -1

# Optional GeoPandas overlay/join
if HAVE_GPD:
    gdf_all = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df['LONGITUDE'], df['LATITUDE']), crs="EPSG:4326")
    wards_gdf = load_wards_uploaded(ward_file) if ward_file else None
    if wards_gdf is not None and not wards_gdf.empty:
        try:
            gdf_all = gpd.sjoin(gdf_all, wards_gdf, how="left", predicate="within")
        except Exception:
            pass
else:
    gdf_all = df.copy()

# Excel summary for clusters
clustered = gdf_all[gdf_all['IS_CLUSTERED']]
excel_filename = "Clustering_Application_Summary.xlsx"
if not clustered.empty:
    (clustered.drop(columns=["geometry"]) if "geometry" in clustered.columns else clustered).to_excel(
        excel_filename, sheet_name='Clustering Application Summary', index=False
    )
    hyperlinkify_excel(excel_filename)

# ---------------- Location (robust: Browser GPS → JS → IP → Manual) ----------------
st.subheader("Step 4: Your Location")

origin_lat, origin_lon = st.session_state.origin_lat, st.session_state.origin_lon

c1, c2, c3, c4 = st.columns(4)
with c1:
    use_browser = st.button("📍 Use Browser GPS", key="btn_browser_gps")
with c2:
    do_refresh = st.button("🔄 Refresh", key="btn_refresh_gps")
with c3:
    do_clear = st.button("❌ Clear", key="btn_clear_gps")
with c4:
    use_ip = st.button("🌐 Use IP Location", key="btn_ip_gps")

if do_clear:
    st.session_state.origin_lat = None
    st.session_state.origin_lon = None
    origin_lat = origin_lon = None
    st.info("Cleared saved location.")

def try_browser_gps_once():
    # 1) streamlit_geolocation component (if installed)
    if HAVE_GEO:
        st.caption("Grant location permission in your browser if prompted.")
        loc = st_geolocation(key=f"geo_once")
        if loc and loc.get("latitude") and loc.get("longitude"):
            st.session_state.origin_lat = float(loc["latitude"])
            st.session_state.origin_lon = float(loc["longitude"])
            return True
    # 2) JS fallback (works on HTTPS)
    if HAVE_JS:
        js = """
        new Promise((resolve) => {
          if (!navigator.geolocation) { resolve(null); return; }
          navigator.geolocation.getCurrentPosition(
            (pos) => resolve([pos.coords.latitude, pos.coords.longitude]),
            () => resolve(null),
            { enableHighAccuracy: true, timeout: 12000, maximumAge: 0 }
          );
        })
        """
        coords = streamlit_js_eval(js_expressions=js, key=f"geo_js_{int(__import__('time').time())}")
        if coords and isinstance(coords, list) and len(coords) == 2:
            st.session_state.origin_lat = float(coords[0])
            st.session_state.origin_lon = float(coords[1])
            return True
    return False

if use_browser or do_refresh:
    ok = try_browser_gps_once()
    if ok:
        origin_lat, origin_lon = st.session_state.origin_lat, st.session_state.origin_lon
        st.success(f"Browser GPS: {origin_lat:.6f}, {origin_lon:.6f}")
    else:
        st.warning("Couldn’t get browser GPS (permission blocked or component not loaded).")

if use_ip and (origin_lat is None or origin_lon is None):
    ip_lat, ip_lon = get_ip_location()
    if ip_lat and ip_lon:
        st.session_state.origin_lat = ip_lat
        st.session_state.origin_lon = ip_lon
        origin_lat, origin_lon = ip_lat, ip_lon
        st.success(f"IP-based location: {origin_lat:.6f}, {origin_lon:.6f}")
    else:
        st.warning("IP-based lookup failed. You can still use the route link (Maps will use live GPS).")

# Manual fallback only if nothing saved
if origin_lat is None or origin_lon is None:
    m1, m2 = st.columns(2)
    with m1:
        man_lat = st.text_input("Manual latitude (optional)", key="txt_lat")
    with m2:
        man_lon = st.text_input("Manual longitude (optional)", key="txt_lon")
    if man_lat.strip() and man_lon.strip():
        try:
            st.session_state.origin_lat = float(man_lat)
            st.session_state.origin_lon = float(man_lon)
            origin_lat, origin_lon = st.session_state.origin_lat, st.session_state.origin_lon
            st.success(f"Manual location: {origin_lat:.6f}, {origin_lon:.6f}")
        except Exception:
            st.error("Invalid coordinates. Example: 26.8467, 80.9462")
else:
    st.info(f"Using saved location: {origin_lat:.6f}, {origin_lon:.6f}")

# ---------------- Batch + Skip/Mark ----------------
st.subheader("Step 5: Batches of 10 — View • Skip • Mark")

# Filter out visited/skipped from the pool
pool = (gdf_all.drop(columns=["geometry"]) if "geometry" in getattr(gdf_all, "columns", []) else gdf_all).copy()
pool = pool[~pool['ISSUE ID'].astype(str).isin(st.session_state.visited_ticket_ids)]
pool = pool[~pool['ISSUE ID'].astype(str).isin(st.session_state.skipped_ticket_ids)]

st.caption(f"Remaining eligible tickets: {len(pool)} | Visited: {len(st.session_state.visited_ticket_ids)} | Skipped: {len(st.session_state.skipped_ticket_ids)}")

if pool.empty:
    st.info("No tickets remaining after filters/visited/skipped.")
    batch_df = pd.DataFrame(columns=['ISSUE ID','WARD','STATUS','LATITUDE','LONGITUDE'])
else:
    # Build a long sequence (up to 200) using greedy NN, then show the current batch of 10 via cursor
    # Seed from origin if available, else from centroid
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

    # Show batch table
    batch_df_display = batch_df[['ISSUE ID','WARD','STATUS','LATITUDE','LONGITUDE']].reset_index(drop=True)
    batch_df_display.index = batch_df_display.index + 1  # 1-based row numbers
    st.dataframe(batch_df_display, use_container_width=True)

    # Google Maps link for this batch (origin optional)
    if not batch_df.empty:
        waypoints = [(float(r['LATITUDE']), float(r['LONGITUDE'])) for _, r in batch_df.iterrows()]
        last = waypoints[-1]
        mids = waypoints[:-1] if len(waypoints) > 1 else []
        nav_url = google_maps_url(origin_lat, origin_lon, last[0], last[1], waypoints=mids)
        st.markdown(f"[🧭 Open route in Google Maps for this batch]({nav_url})")

    # Actions
    c1, c2, c3, c4 = st.columns(4)
    first_id = str(batch_df.iloc[0]['ISSUE ID']) if len(batch_df) else None

    with c1:
        if st.button("✅ Mark first as Visited", key="btn_mark_first"):
            if first_id:
                st.session_state.visited_ticket_ids.add(first_id)
                st.experimental_rerun()

    with c2:
        if st.button("⏭️ Skip first", key="btn_skip_first"):
            if first_id:
                st.session_state.skipped_ticket_ids.add(first_id)
                st.experimental_rerun()

    with c3:
        if st.button("✅ Mark entire batch as Visited", key="btn_mark_batch"):
            for _id in batch_df['ISSUE ID'].astype(str).tolist():
                st.session_state.visited_ticket_ids.add(_id)
            st.experimental_rerun()

    with c4:
        if st.button("➡️ Next batch", key="btn_next_batch"):
            st.session_state.batch_cursor = end if end < len(seq_df) else 0
            st.experimental_rerun()

# ---------------- Map (+ clickable image links / thumbnails) ----------------
st.subheader("Step 6: Map")

from html import escape
show_thumbs = st.checkbox("Show image thumbnails in popups (may slow the map)", value=False, key="cb_thumbs")

center_lat = float(df['LATITUDE'].mean())
center_lon = float(df['LONGITUDE'].mean())
m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
mc = MarkerCluster(name="Tickets").add_to(m)

plot_df = (gdf_all.drop(columns=["geometry"]) if "geometry" in getattr(gdf_all, "columns", []) else gdf_all).copy()

# Color coding: visited = gray, skipped = purple, current batch = orange, first = green
batch_ids = set(batch_df['ISSUE ID'].astype(str).tolist()) if not batch_df.empty else set()
first_in_batch = str(batch_df.iloc[0]['ISSUE ID']) if not batch_df.empty else None

for _, row in plot_df.iterrows():
    rid = str(row['ISSUE ID'])
    lat, lon = float(row['LATITUDE']), float(row['LONGITUDE'])

    if rid == first_in_batch:
        color, size = 'green', 10
    elif rid in batch_ids:
        color, size = 'orange', 9
    elif rid in st.session_state.visited_ticket_ids:
        color, size = 'gray', 7
    elif rid in st.session_state.skipped_ticket_ids:
        color, size = 'purple', 7
    else:
        color, size = 'red', 7

    before_url = str(row.get('BEFORE PHOTO', '') or '').strip()
    after_url  = str(row.get('AFTER PHOTO', '') or '').strip()
    ward       = escape(str(row.get('WARD', '') or ''))
    status     = escape(str(row.get('STATUS', '') or ''))
    cluster    = row.get('CLUSTER NUMBER', '')
    cluster    = int(cluster) if isinstance(cluster, (int, float)) and not pd.isna(cluster) else ''

    parts = [
        f"<b>Issue ID:</b> {escape(rid)}",
        f"<b>Status:</b> {status}",
        f"<b>Ward:</b> {ward}",
        f"<b>Lat, Lon:</b> {lat:.6f}, {lon:.6f}",
    ]
    if cluster != '':
        parts.insert(0, f"<b>Cluster:</b> {cluster}")

    if is_url(before_url):
        parts.append(f"<a href='{before_url}' target='_blank'>Before photo</a>")
        if show_thumbs:
            parts.append(f"<div><img src='{before_url}' style='max-width:220px;border:1px solid #ccc;border-radius:6px;'/></div>")
    if is_url(after_url):
        parts.append(f"<a href='{after_url}' target='_blank'>After photo</a>")
        if show_thumbs:
            parts.append(f"<div><img src='{after_url}' style='max-width:220px;border:1px solid #ccc;border-radius:6px;'/></div>")

    popup_html = "<div style='font-size:13px;line-height:1.25'>" + "<br>".join(parts) + "</div>"
    popup = folium.Popup(popup_html, max_width=280)

    folium.CircleMarker(
        location=[lat, lon],
        radius=size,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.9,
        popup=popup
    ).add_to(mc)

m.save("Clustering_Application_Map.html")

# ---------------- Downloads ----------------
st.subheader("Step 7: Downloads")
if not clustered.empty:
    with open("Clustering_Application_Summary.xlsx", "rb") as f:
        st.download_button("Download Excel", f, file_name="Clustering_Application_Summary.xlsx", key="dl_excel")
with open("Clustering_Application_Map.html", "rb") as f:
    st.download_button("Download Map (HTML)", f, file_name="Clustering_Application_Map.html", key="dl_html")
