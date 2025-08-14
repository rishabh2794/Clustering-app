# Unified Clustering + Batch Navigation with Auto Geolocation (app.py)
# -------------------------------------------------------------------
# Features:
# - Upload CSV of tickets (required) and ward boundary (GeoJSON/JSON/KML, optional)
# - DBSCAN clustering (haversine)
# - Ward spatial join (optional)
# - Auto-location via browser (with manual fallback)
# - Build Google Maps route to next N tickets (nearest-neighbor; default 10)
# - Map shows ALL tickets; batch highlighted (first = green, rest = orange)
# - Downloads: Excel summary (clustered points) + HTML Folium map

import math
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import geopandas as gpd
from sklearn.cluster import DBSCAN
import folium
from folium.plugins import MarkerCluster
from shapely.geometry import Point
from openpyxl import load_workbook

# Auto geolocation (browser)
try:
    from streamlit_geolocation import geolocation
    HAVE_GEO = True
except Exception:
    HAVE_GEO = False

# Optional: improve KML handling
try:
    import fiona
    HAS_FIONA = True
except Exception:
    HAS_FIONA = False

# -------------------------
# Constants & Helpers
# -------------------------
EARTH_RADIUS_M = 6_371_000.0
REQUIRED_COLS = {
    'ISSUE ID', 'CITY', 'ZONE', 'WARD', 'SUBCATEGORY', 'CREATED AT',
    'STATUS', 'LATITUDE', 'LONGITUDE', 'BEFORE PHOTO', 'AFTER PHOTO', 'ADDRESS'
}

st.set_page_config(layout="wide", page_title="Clustering + Batch Navigation")
st.title("🗺️ Clustering + Batch Navigation (Auto-Location)")

with st.sidebar:
    st.markdown("### Tips")
    st.markdown(
        "- CSV must include **LATITUDE/LONGITUDE** (decimal degrees).  \n"
        "- Clustering **radius** is in meters.  \n"
        "- If KML ward reading fails, convert it to GeoJSON.  \n"
        "- Auto-location needs **HTTPS** + allow location in browser."
    )

subcategory_options = [
    "Pothole",
    "Sand piled on roadsides + Mud/slit on roadside",
    "Garbage dumped on public land",
    "Unpaved Road",
    "Broken Footpath / Divider",
    "Malba, bricks, bori, etc dumped on public land",
    "Construction/ demolition activity without safeguards",
    "Encroachment-Building Materials Dumped on Road",
    "Burning of garbage, plastic, leaves, branches etc.",
    "Overflowing Dustbins",
    "Barren land to be greened",
    "Greening of Central Verges",
    "Unsurfaced Parking Lots"
]

# -------------------------
# Utility functions
# -------------------------
def normalize_subcategory(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    phi1 = math.radians(float(lat1)); phi2 = math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlmb = math.radians(float(lon2) - float(lon1))
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2)**2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))

def google_maps_url(origin_lat, origin_lon, dest_lat, dest_lon, mode="driving", waypoints=None) -> str:
    base = "https://www.google.com/maps/dir/?api=1"
    origin = f"&origin={origin_lat},{origin_lon}"
    dest = f"&destination={dest_lat},{dest_lon}"
    travel = f"&travelmode={mode}"
    wp = ""
    if waypoints:
        wp_str = "|".join([f"{lat},{lon}" for (lat, lon) in waypoints])
        wp = f"&waypoints={wp_str}"
    return f"{base}{origin}{dest}{travel}{wp}"

def hyperlinkify_excel(excel_path: str, sheet_name: str = "Clustering Application Summary") -> None:
    try:
        wb = load_workbook(excel_path)
        ws = wb[sheet_name]
        for row in range(2, ws.max_row + 1):
            for col_idx in (11, 12):
                link = ws.cell(row, col_idx).value
                if link and isinstance(link, str) and link.startswith(("http://", "https://")):
                    ws.cell(row, col_idx).hyperlink = link
                    ws.cell(row, col_idx).style = "Hyperlink"
        wb.save(excel_path)
    except Exception as e:
        st.warning(f"Excel hyperlinking skipped: {e}")

def load_wards_uploaded(file) -> gpd.GeoDataFrame | None:
    try:
        suffix = file.name.split(".")[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix="." + suffix) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        if suffix in ("geojson", "json"):
            wards = gpd.read_file(tmp_path)
        elif suffix == "kml":
            if HAS_FIONA:
                layers = fiona.listlayers(tmp_path)
                if not layers:
                    st.error("KML has no layers.")
                    return None
                wards = None
                for lyr in layers:
                    gdf_try = gpd.read_file(tmp_path, driver="KML", layer=lyr)
                    if not gdf_try.empty:
                        wards = gdf_try
                        break
                if wards is None or wards.empty:
                    st.error("KML layers empty. Convert to GeoJSON.")
                    return None
            else:
                try:
                    wards = gpd.read_file(tmp_path, driver="KML")
                except Exception as e:
                    st.error(f"KML reading failed: {e}")
                    return None
        else:
            st.error("Unsupported file type for wards.")
            return None

        if wards is None or wards.empty:
            st.error("Ward file has no features.")
            return None
        if wards.crs is None:
            wards.set_crs(epsg=4326, inplace=True)
        elif wards.crs.to_string() != "EPSG:4326":
            wards = wards.to_crs(epsg=4326)

        wards = wards[wards.geometry.notna()].copy()
        wards = wards[~wards.geometry.is_empty]
        return wards
    except Exception as e:
        st.error(f"Error reading ward file: {e}")
        return None

# -------------------------
# Session State
# -------------------------
if "visited_ticket_ids" not in st.session_state:
    st.session_state.visited_ticket_ids = set()
if "skipped_ticket_ids" not in st.session_state:
    st.session_state.skipped_ticket_ids = set()
if "current_target_id" not in st.session_state:
    st.session_state.current_target_id = None
if "batch_target_ids" not in st.session_state:
    st.session_state.batch_target_ids = set()

# -------------------------
# Step 1: Upload Files
# -------------------------
st.subheader("Step 1: Upload Required Files")
csv_file = st.file_uploader("Upload CSV file with issues", type=["csv"])
ward_file = st.file_uploader("Upload WARD boundary file (optional)", type=["geojson", "json", "kml"])

st.subheader("Step 2: Select Issue Subcategory")
subcategory_option = st.selectbox("Choose issue subcategory to analyze:", subcategory_options)

st.subheader("Step 3: Set Clustering Parameters")
radius_m = st.number_input("Clustering Radius (meters)", min_value=1, max_value=1000, value=15)
min_samples = st.number_input("Minimum Issues per Cluster", min_value=1, max_value=100, value=2)
if radius_m < 10 or min_samples < 2:
    st.warning("⚠️ Low values may cause many small clusters.")

# -------------------------
# Core processing
# -------------------------
if csv_file:
    df = pd.read_csv(csv_file)
    missing = sorted(list(REQUIRED_COLS - set(df.columns)))
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    df['SUBCATEGORY_NORM'] = normalize_subcategory(df['SUBCATEGORY'])
    desired = subcategory_option.strip().lower()
    df = df[df['SUBCATEGORY_NORM'] == desired].copy()
    if df.empty:
        st.info("No matching subcategory.")
        st.stop()

    df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
    df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
    df = df.dropna(subset=['LATITUDE', 'LONGITUDE'])
    for col in ['CREATED AT', 'STATUS', 'ADDRESS', 'BEFORE PHOTO', 'AFTER PHOTO', 'ISSUE ID', 'ZONE', 'WARD', 'CITY']:
        if col in df.columns:
            df[col] = df[col].astype(str)

    coords_deg = df[['LATITUDE', 'LONGITUDE']].to_numpy()
    coords_rad = np.radians(coords_deg)
    eps_rad = radius_m / EARTH_RADIUS_M
    db = DBSCAN(eps=eps_rad, min_samples=int(min_samples), metric='haversine', algorithm='ball_tree')
    labels = db.fit_predict(coords_rad)
    df['CLUSTER NUMBER'] = labels
    df['IS_CLUSTERED'] = df['CLUSTER NUMBER'] != -1

    wards_gdf = load_wards_uploaded(ward_file) if ward_file else None
    gdf_all = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df['LONGITUDE'], df['LATITUDE']), crs="EPSG:4326")
    if wards_gdf is not None:
        try:
            gdf_all = gpd.sjoin(gdf_all, wards_gdf, how="left", predicate="within")
        except Exception as e:
            st.warning(f"Ward join failed: {e}")

    clustered = gdf_all[gdf_all['IS_CLUSTERED']].copy()
    if not clustered.empty:
        sizes = clustered.groupby('CLUSTER NUMBER')['ISSUE ID'].count().rename("NUMBER OF ISSUES")
        summary_sheet = clustered.merge(sizes, on='CLUSTER NUMBER', how='left')
        summary_sheet = summary_sheet[[
            'CLUSTER NUMBER', 'NUMBER OF ISSUES', 'ISSUE ID', 'ZONE', 'WARD',
            'SUBCATEGORY', 'CREATED AT', 'STATUS', 'LATITUDE', 'LONGITUDE',
            'BEFORE PHOTO', 'AFTER PHOTO', 'ADDRESS'
        ]].sort_values(['CLUSTER NUMBER', 'CREATED AT'])
    else:
        summary_sheet = pd.DataFrame(columns=[
            'CLUSTER NUMBER', 'NUMBER OF ISSUES', 'ISSUE ID', 'ZONE', 'WARD',
            'SUBCATEGORY', 'CREATED AT', 'STATUS', 'LATITUDE', 'LONGITUDE',
            'BEFORE PHOTO', 'AFTER PHOTO', 'ADDRESS'
        ])

    excel_filename = "Clustering_Application_Summary.xlsx"
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        summary_sheet.to_excel(writer, index=False, sheet_name='Clustering Application Summary')
    hyperlinkify_excel(excel_filename)

    # -------------------------
    # Step 4A — Navigate & Geolocation FIX
    # -------------------------
    st.subheader("Step 4A: Navigate to Nearest Tickets (Batch)")

    unique_statuses = sorted(gdf_all['STATUS'].dropna().astype(str).unique())
    default_statuses = [s for s in unique_statuses if s.lower() in ("open", "pending", "in progress")]
    include_statuses = st.multiselect("Eligible ticket statuses", options=unique_statuses, default=default_statuses or unique_statuses)

    wards_in_data = sorted(gdf_all['WARD'].dropna().astype(str).unique())
    ward_filter = st.multiselect("Limit to ward(s) (optional)", options=wards_in_data)

    travel_mode = st.selectbox("Travel mode", ["driving", "walking", "two_wheeler"], index=0)
    batch_size = st.slider("Batch size", 1, 10, 10)

    origin_lat = origin_lon = None
    st.markdown("### Your Location")
    colA, colB = st.columns(2)
    with colA:
        if HAVE_GEO:
            st.caption("Click the button and allow browser permission.")
            loc = geolocation()
            if loc and isinstance(loc, dict) and loc.get("latitude") and loc.get("longitude"):
                origin_lat, origin_lon = float(loc["latitude"]), float(loc["longitude"])
                st.session_state["origin_lat"] = origin_lat
                st.session_state["origin_lon"] = origin_lon
                st.success(f"Auto location: {origin_lat:.6f}, {origin_lon:.6f}")
            elif "origin_lat" in st.session_state:
                origin_lat = st.session_state["origin_lat"]
                origin_lon = st.session_state["origin_lon"]
                st.info(f"Using saved location: {origin_lat:.6f}, {origin_lon:.6f}")
            else:
                st.warning("Location not received — click button above.")
        else:
            st.info("Install geolocation: pip install streamlit-geolocation")
    with colB:
        manual_lat = st.text_input("Manual latitude", "")
        manual_lon = st.text_input("Manual longitude", "")
        if manual_lat.strip() and manual_lon.strip():
            try:
                origin_lat, origin_lon = float(manual_lat), float(manual_lon)
                st.session_state["origin_lat"] = origin_lat
                st.session_state["origin_lon"] = origin_lon
                st.success(f"Manual location: {origin_lat:.6f}, {origin_lon:.6f}")
            except Exception:
                st.error("Invalid manual coordinates.")

    pool = gdf_all.copy()
    if include_statuses:
        pool = pool[pool['STATUS'].isin(include_statuses)]
    if ward_filter:
        pool = pool[pool['WARD'].isin(ward_filter)]
    if st.session_state.visited_ticket_ids:
        pool = pool[~pool['ISSUE ID'].isin(st.session_state.visited_ticket_ids)]
    if st.session_state.skipped_ticket_ids:
        pool = pool[~pool['ISSUE ID'].isin(st.session_state.skipped_ticket_ids)]

    if origin_lat is not None and origin_lon is not None and not pool.empty:
        sequence_rows = []
        cur_lat, cur_lon = origin_lat, origin_lon
        pool2 = pool.copy()
        for _ in range(min(batch_size, len(pool2))):
            pool2['__dist_m'] = pool2.apply(lambda r: haversine_m(cur_lat, cur_lon, r['LATITUDE'], r['LONGITUDE']), axis=1)
            nxt = pool2.sort_values('__dist_m').iloc[0]
            sequence_rows.append(nxt)
            cur_lat, cur_lon = nxt['LATITUDE'], nxt['LONGITUDE']
            pool2 = pool2[pool2['ISSUE ID'] != nxt['ISSUE ID']]

        if sequence_rows:
            waypoints = [(r['LATITUDE'], r['LONGITUDE']) for r in sequence_rows]
            first, last = waypoints[0], waypoints[-1]
            mids = waypoints[:-1] if len(waypoints) > 1 else []
            nav_url = google_maps_url(origin_lat, origin_lon, last[0], last[1], mode=travel_mode, waypoints=mids)
            st.markdown(f"[🧭 Open continuous navigation in Google Maps]({nav_url})")

    # Step 5 and Step 6 (map display + downloads) — remain same from your previous code
