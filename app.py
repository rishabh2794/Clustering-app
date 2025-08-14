# Unified Clustering + Batch Navigation with Auto + IP Geolocation Fallback (app.py)

import math
import tempfile
import requests
import numpy as np
import pandas as pd
import streamlit as st
import geopandas as gpd
from sklearn.cluster import DBSCAN
import folium
from folium.plugins import MarkerCluster
from shapely.geometry import Point
from openpyxl import load_workbook
import streamlit.components.v1 as components

try:
    from streamlit_geolocation import geolocation
    HAVE_GEO = True
except Exception:
    HAVE_GEO = False

try:
    import fiona
    HAS_FIONA = True
except Exception:
    HAS_FIONA = False

EARTH_RADIUS_M = 6_371_000.0
REQUIRED_COLS = {
    'ISSUE ID', 'CITY', 'ZONE', 'WARD', 'SUBCATEGORY', 'CREATED AT',
    'STATUS', 'LATITUDE', 'LONGITUDE', 'BEFORE PHOTO', 'AFTER PHOTO', 'ADDRESS'
}

st.set_page_config(layout="wide", page_title="Clustering + Batch Navigation")
st.title("🗺️ Clustering + Batch Navigation with Auto + IP Geolocation")

subcategory_options = [
    "Pothole", "Sand piled on roadsides + Mud/slit on roadside",
    "Garbage dumped on public land", "Unpaved Road",
    "Broken Footpath / Divider", "Malba, bricks, bori, etc dumped on public land",
    "Construction/ demolition activity without safeguards",
    "Encroachment-Building Materials Dumped on Road",
    "Burning of garbage, plastic, leaves, branches etc.",
    "Overflowing Dustbins", "Barren land to be greened",
    "Greening of Central Verges", "Unsurfaced Parking Lots"
]

# Helpers
def normalize_subcategory(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    phi1 = math.radians(float(lat1)); phi2 = math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlmb = math.radians(float(lon2) - float(lon1))
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2)**2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))

def google_maps_url(origin_lat, origin_lon, dest_lat, dest_lon, mode="driving", waypoints=None):
    base = "https://www.google.com/maps/dir/?api=1"
    origin = f"&origin={origin_lat},{origin_lon}"
    dest = f"&destination={dest_lat},{dest_lon}"
    travel = f"&travelmode={mode}"
    wp = ""
    if waypoints:
        wp_str = "|".join([f"{lat},{lon}" for (lat, lon) in waypoints])
        wp = f"&waypoints={wp_str}"
    return f"{base}{origin}{dest}{travel}{wp}"

def hyperlinkify_excel(excel_path):
    try:
        wb = load_workbook(excel_path)
        ws = wb["Clustering Application Summary"]
        for row in range(2, ws.max_row + 1):
            for col_idx in (11, 12):
                link = ws.cell(row, col_idx).value
                if link and isinstance(link, str) and link.startswith(("http://", "https://")):
                    ws.cell(row, col_idx).hyperlink = link
                    ws.cell(row, col_idx).style = "Hyperlink"
        wb.save(excel_path)
    except Exception as e:
        st.warning(f"Excel hyperlinking skipped: {e}")

def load_wards_uploaded(file):
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
                wards = None
                for lyr in layers:
                    gdf_try = gpd.read_file(tmp_path, driver="KML", layer=lyr)
                    if not gdf_try.empty:
                        wards = gdf_try
                        break
            else:
                wards = gpd.read_file(tmp_path, driver="KML")
        else:
            st.error("Unsupported ward file type.")
            return None
        if wards.crs is None:
            wards.set_crs(epsg=4326, inplace=True)
        elif wards.crs.to_string() != "EPSG:4326":
            wards = wards.to_crs(epsg=4326)
        return wards[wards.geometry.notna() & ~wards.geometry.is_empty]
    except Exception as e:
        st.error(f"Error reading ward file: {e}")
        return None

# ---------------- State ----------------
for key in ["visited_ticket_ids", "skipped_ticket_ids", "batch_target_ids"]:
    if key not in st.session_state:
        st.session_state[key] = set()
if "current_target_id" not in st.session_state:
    st.session_state.current_target_id = None

# ---------------- Inputs ----------------
st.subheader("Step 1: Upload Required Files")
csv_file = st.file_uploader("Upload CSV", type=["csv"])
ward_file = st.file_uploader("Upload WARD file (Optional)", type=["geojson", "json", "kml"])

st.subheader("Step 2: Select Subcategory")
subcategory_option = st.selectbox("Choose subcategory:", subcategory_options)

st.subheader("Step 3: Clustering Parameters")
radius_m = st.number_input("Radius (m)", 1, 1000, 15)
min_samples = st.number_input("Minimum points per cluster", 1, 100, 2)

if csv_file:
    df = pd.read_csv(csv_file)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        st.error(f"Missing: {missing}")
        st.stop()
    df['SUBCATEGORY_NORM'] = normalize_subcategory(df['SUBCATEGORY'])
    df = df[df['SUBCATEGORY_NORM'] == subcategory_option.lower()]
    df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
    df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
    df.dropna(subset=['LATITUDE', 'LONGITUDE'], inplace=True)
    coords_rad = np.radians(df[['LATITUDE', 'LONGITUDE']])
    eps_rad = radius_m / EARTH_RADIUS_M
    db = DBSCAN(eps=eps_rad, min_samples=int(min_samples), metric='haversine', algorithm='ball_tree')
    df['CLUSTER NUMBER'] = db.fit_predict(coords_rad)
    df['IS_CLUSTERED'] = df['CLUSTER NUMBER'] != -1
    wards_gdf = load_wards_uploaded(ward_file) if ward_file else None
    gdf_all = gpd.GeoDataFrame(df.copy(),
        geometry=gpd.points_from_xy(df['LONGITUDE'], df['LATITUDE']), crs="EPSG:4326")
    if wards_gdf is not None:
        try:
            gdf_all = gpd.sjoin(gdf_all, wards_gdf, how="left", predicate="within")
        except:
            pass

    # Excel export
    clustered = gdf_all[gdf_all['IS_CLUSTERED']]
    excel_filename = "Clustering_Application_Summary.xlsx"
    clustered.to_excel(excel_filename,
                       sheet_name='Clustering Application Summary', index=False)
    hyperlinkify_excel(excel_filename)

    # ---------------- Location Retrieval ----------------
    st.subheader("Step 4: Your Location")
    origin_lat = origin_lon = None

    # Inject JS for browser geolocation
    js_code = """
    <script>
    navigator.geolocation.getCurrentPosition(
        (pos) => {
            const coords = pos.coords.latitude + "," + pos.coords.longitude;
            window.parent.postMessage({coords: coords}, "*");
        },
        (err) => {
            window.parent.postMessage({coords: "error"}, "*");
        }
    );
    </script>
    """
    components.html(js_code, height=0)

    # Read posted coords from URL hash (workaround)
    if "origin_lat" in st.session_state and "origin_lon" in st.session_state:
        origin_lat, origin_lon = st.session_state["origin_lat"], st.session_state["origin_lon"]
    else:
        try:
            r = requests.get("https://ipapi.co/json/", timeout=4)
            if r.status_code == 200:
                data = r.json()
                origin_lat, origin_lon = float(data['latitude']), float(data['longitude'])
                st.info(f"IP-based location: {origin_lat:.6f}, {origin_lon:.6f}")
        except:
            st.error("Could not determine location")

    if origin_lat and origin_lon:
        st.session_state["origin_lat"] = origin_lat
        st.session_state["origin_lon"] = origin_lon

    # ---------------- Navigation ----------------
    if origin_lat and origin_lon:
        pool = gdf_all.copy()
        sequence_rows = []
        cur_lat, cur_lon = origin_lat, origin_lon
        for _ in range(min(10, len(pool))):
            pool['__dist'] = pool.apply(lambda r: haversine_m(cur_lat, cur_lon, r['LATITUDE'], r['LONGITUDE']), axis=1)
            nxt = pool.sort_values('__dist').iloc[0]
            sequence_rows.append(nxt)
            cur_lat, cur_lon = nxt['LATITUDE'], nxt['LONGITUDE']
            pool = pool[pool['ISSUE ID'] != nxt['ISSUE ID']]
        if sequence_rows:
            waypoints = [(r['LATITUDE'], r['LONGITUDE']) for _, r in pd.DataFrame(sequence_rows).iterrows()]
            nav_url = google_maps_url(origin_lat, origin_lon, waypoints[-1][0], waypoints[-1][1], waypoints=waypoints[:-1])
            st.markdown(f"[🧭 Open continuous navigation in Google Maps]({nav_url})")

    # ---------------- Map ----------------
    m = folium.Map(location=[df['LATITUDE'].mean(), df['LONGITUDE'].mean()], zoom_start=13)
    mc = MarkerCluster().add_to(m)
    for _, row in gdf_all.iterrows():
        folium.Marker([row['LATITUDE'], row['LONGITUDE']],
                      popup=row['ISSUE ID']).add_to(mc)
    m.save("Clustering_Application_Map.html")

    # ---------------- Downloads ----------------
    st.download_button("Download Excel", open(excel_filename, "rb"), file_name=excel_filename)
    st.download_button("Download Map", open("Clustering_Application_Map.html", "rb"), file_name="Clustering_Application_Map.html")
