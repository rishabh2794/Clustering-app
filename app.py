# app.py — Clustering + Batch Navigation with Browser + IP Geolocation Fallback
# -------------------------------------------------------------------------------

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
from openpyxl import load_workbook
from shapely.geometry import Point

try:
    from streamlit_geolocation import geolocation
    HAVE_GEO = True
except:
    HAVE_GEO = False

try:
    import fiona
    HAS_FIONA = True
except:
    HAS_FIONA = False

# Constants
EARTH_RADIUS_M = 6_371_000.0
REQUIRED_COLS = {
    'ISSUE ID', 'CITY', 'ZONE', 'WARD', 'SUBCATEGORY', 'CREATED AT',
    'STATUS', 'LATITUDE', 'LONGITUDE', 'BEFORE PHOTO', 'AFTER PHOTO', 'ADDRESS'
}

# ----------------- Helper functions -----------------
def normalize_subcategory(series: pd.Series):
    return series.astype(str).str.strip().str.lower()

def haversine_m(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb/2)**2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))

def google_maps_url(origin_lat, origin_lon, dest_lat, dest_lon, mode="driving", waypoints=None):
    base = "https://www.google.com/maps/dir/?api=1"
    origin = f"&origin={origin_lat},{origin_lon}"
    dest = f"&destination={dest_lat},{dest_lon}"
    travel = f"&travelmode={mode}"
    wp = f"&waypoints={'|'.join([f'{lat},{lon}' for lat,lon in waypoints])}" if waypoints else ""
    return f"{base}{origin}{dest}{travel}{wp}"

def hyperlinkify_excel(excel_path):
    try:
        wb = load_workbook(excel_path)
        ws = wb["Clustering Application Summary"]
        for row in range(2, ws.max_row + 1):
            for col in (11, 12):  # photo URLs
                link = ws.cell(row, col).value
                if link and isinstance(link, str) and link.startswith(("http://", "https://")):
                    ws.cell(row, col).hyperlink = link
                    ws.cell(row, col).style = "Hyperlink"
        wb.save(excel_path)
    except:
        pass

def load_wards_uploaded(file):
    try:
        suffix = file.name.split(".")[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix="."+suffix) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        if suffix in ("geojson", "json"):
            gdf = gpd.read_file(tmp_path)
        elif suffix == "kml":
            if HAS_FIONA:
                layers = fiona.listlayers(tmp_path)
                gdf = None
                for layer in layers:
                    gdf_try = gpd.read_file(tmp_path, driver="KML", layer=layer)
                    if not gdf_try.empty:
                        gdf = gdf_try
                        break
            else:
                gdf = gpd.read_file(tmp_path, driver="KML")
        else:
            return None
        if gdf.crs is None: gdf.set_crs(epsg=4326, inplace=True)
        elif gdf.crs.to_string() != "EPSG:4326": gdf = gdf.to_crs(epsg=4326)
        return gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    except:
        return None

def get_ip_location():
    try:
        r = requests.get("https://ipapi.co/json/", timeout=5)
        if r.status_code == 200:
            data = r.json()
            return float(data["latitude"]), float(data["longitude"])
    except:
        pass
    return None, None

# ----------------- App Setup -----------------
st.set_page_config(layout="wide", page_title="Clustering + Batch Navigation")
st.title("🗺️ Clustering + Batch Navigation (Auto + IP Fallback Location)")

# State init
for k in ["visited_ticket_ids", "skipped_ticket_ids", "batch_target_ids"]:
    if k not in st.session_state: st.session_state[k] = set()
if "current_target_id" not in st.session_state: st.session_state.current_target_id = None

# Inputs
csv_file = st.file_uploader("Upload CSV", type=["csv"])
ward_file = st.file_uploader("Upload Wards file (optional)", type=["geojson","json","kml"])
subcategory_option = st.selectbox("Issue Subcategory", [
    "Pothole","Garbage dumped on public land","Unpaved Road","Broken Footpath / Divider",
    "Sand piled on roadsides + Mud/slit on roadside","Malba, bricks, bori, etc dumped on public land",
    "Construction/ demolition activity without safeguards","Encroachment-Building Materials Dumped on Road",
    "Burning of garbage, plastic, leaves, branches etc.","Overflowing Dustbins",
    "Barren land to be greened","Greening of Central Verges","Unsurfaced Parking Lots"
])
radius_m = st.number_input("Clustering radius (m)", 1, 1000, 15)
min_samples = st.number_input("Minimum per cluster", 1, 100, 2)

if csv_file:
    # Data load + filter
    df = pd.read_csv(csv_file)
    if REQUIRED_COLS - set(df.columns):
        st.error("Missing required columns.")
        st.stop()

    df['SUBCATEGORY_NORM'] = normalize_subcategory(df['SUBCATEGORY'])
    df = df[df['SUBCATEGORY_NORM'] == subcategory_option.lower()].copy()
    df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
    df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
    df.dropna(subset=['LATITUDE','LONGITUDE'], inplace=True)

    coords_rad = np.radians(df[['LATITUDE','LONGITUDE']])
    eps_rad = radius_m / EARTH_RADIUS_M
    db = DBSCAN(eps=eps_rad, min_samples=int(min_samples), metric='haversine', algorithm='ball_tree')
    df['CLUSTER NUMBER'] = db.fit_predict(coords_rad)
    df['IS_CLUSTERED'] = df['CLUSTER NUMBER'] != -1

    gdf_all = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df['LONGITUDE'], df['LATITUDE']), crs="EPSG:4326")
    wards_gdf = load_wards_uploaded(ward_file) if ward_file else None
    if wards_gdf is not None and not wards_gdf.empty:
        try: gdf_all = gpd.sjoin(gdf_all, wards_gdf, how="left", predicate="within")
        except: pass

    # Excel summary for clusters
    clustered = gdf_all[gdf_all['IS_CLUSTERED']]
    excel_filename = "Clustering_Application_Summary.xlsx"
    if not clustered.empty:
        clustered.to_excel(excel_filename, sheet_name='Clustering Application Summary', index=False)
        hyperlinkify_excel(excel_filename)

    # ---------------- Location Step ----------------
    st.subheader("Step 4: Your Location")
    origin_lat = origin_lon = None

    if 'origin_lat' in st.session_state and 'origin_lon' in st.session_state:
        origin_lat, origin_lon = st.session_state['origin_lat'], st.session_state['origin_lon']
        st.info(f"Using saved location: {origin_lat:.6f}, {origin_lon:.6f}")
    else:
        if HAVE_GEO:
            st.caption("Click to get browser GPS location ↓")
            loc = geolocation()
            if loc and loc.get("latitude") and loc.get("longitude"):
                origin_lat, origin_lon = float(loc["latitude"]), float(loc["longitude"])
                st.session_state["origin_lat"], st.session_state["origin_lon"] = origin_lat, origin_lon
                st.success(f"Browser location: {origin_lat:.6f}, {origin_lon:.6f}")
        if origin_lat is None:
            if st.button("Use my approximate (IP-based) location"):
                ip_lat, ip_lon = get_ip_location()
                if ip_lat and ip_lon:
                    origin_lat, origin_lon = ip_lat, ip_lon
                    st.session_state["origin_lat"], st.session_state["origin_lon"] = origin_lat, origin_lon
                    st.success(f"IP-based location: {origin_lat:.6f}, {origin_lon:.6f}")
# app.py — Clustering + Batch Navigation with Browser + IP Geolocation Fallback
# -------------------------------------------------------------------------------

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
from openpyxl import load_workbook
from shapely.geometry import Point

try:
    from streamlit_geolocation import geolocation
    HAVE_GEO = True
except:
    HAVE_GEO = False

try:
    import fiona
    HAS_FIONA = True
except:
    HAS_FIONA = False

# Constants
EARTH_RADIUS_M = 6_371_000.0
REQUIRED_COLS = {
    'ISSUE ID', 'CITY', 'ZONE', 'WARD', 'SUBCATEGORY', 'CREATED AT',
    'STATUS', 'LATITUDE', 'LONGITUDE', 'BEFORE PHOTO', 'AFTER PHOTO', 'ADDRESS'
}

# ----------------- Helper functions -----------------
def normalize_subcategory(series: pd.Series):
    return series.astype(str).str.strip().str.lower()

def haversine_m(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb/2)**2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))

def google_maps_url(origin_lat, origin_lon, dest_lat, dest_lon, mode="driving", waypoints=None):
    base = "https://www.google.com/maps/dir/?api=1"
    origin = f"&origin={origin_lat},{origin_lon}"
    dest = f"&destination={dest_lat},{dest_lon}"
    travel = f"&travelmode={mode}"
    wp = f"&waypoints={'|'.join([f'{lat},{lon}' for lat,lon in waypoints])}" if waypoints else ""
    return f"{base}{origin}{dest}{travel}{wp}"

def hyperlinkify_excel(excel_path):
    try:
        wb = load_workbook(excel_path)
        ws = wb["Clustering Application Summary"]
        for row in range(2, ws.max_row + 1):
            for col in (11, 12):  # photo URLs
                link = ws.cell(row, col).value
                if link and isinstance(link, str) and link.startswith(("http://", "https://")):
                    ws.cell(row, col).hyperlink = link
                    ws.cell(row, col).style = "Hyperlink"
        wb.save(excel_path)
    except:
        pass

def load_wards_uploaded(file):
    try:
        suffix = file.name.split(".")[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix="."+suffix) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        if suffix in ("geojson", "json"):
            gdf = gpd.read_file(tmp_path)
        elif suffix == "kml":
            if HAS_FIONA:
                layers = fiona.listlayers(tmp_path)
                gdf = None
                for layer in layers:
                    gdf_try = gpd.read_file(tmp_path, driver="KML", layer=layer)
                    if not gdf_try.empty:
                        gdf = gdf_try
                        break
            else:
                gdf = gpd.read_file(tmp_path, driver="KML")
        else:
            return None
        if gdf.crs is None: gdf.set_crs(epsg=4326, inplace=True)
        elif gdf.crs.to_string() != "EPSG:4326": gdf = gdf.to_crs(epsg=4326)
        return gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    except:
        return None

def get_ip_location():
    try:
        r = requests.get("https://ipapi.co/json/", timeout=5)
        if r.status_code == 200:
            data = r.json()
            return float(data["latitude"]), float(data["longitude"])
    except:
        pass
    return None, None

# ----------------- App Setup -----------------
st.set_page_config(layout="wide", page_title="Clustering + Batch Navigation")
st.title("🗺️ Clustering + Batch Navigation (Auto + IP Fallback Location)")

# State init
for k in ["visited_ticket_ids", "skipped_ticket_ids", "batch_target_ids"]:
    if k not in st.session_state: st.session_state[k] = set()
if "current_target_id" not in st.session_state: st.session_state.current_target_id = None

# Inputs
csv_file = st.file_uploader("Upload CSV", type=["csv"])
ward_file = st.file_uploader("Upload Wards file (optional)", type=["geojson","json","kml"])
subcategory_option = st.selectbox("Issue Subcategory", [
    "Pothole","Garbage dumped on public land","Unpaved Road","Broken Footpath / Divider",
    "Sand piled on roadsides + Mud/slit on roadside","Malba, bricks, bori, etc dumped on public land",
    "Construction/ demolition activity without safeguards","Encroachment-Building Materials Dumped on Road",
    "Burning of garbage, plastic, leaves, branches etc.","Overflowing Dustbins",
    "Barren land to be greened","Greening of Central Verges","Unsurfaced Parking Lots"
])
radius_m = st.number_input("Clustering radius (m)", 1, 1000, 15)
min_samples = st.number_input("Minimum per cluster", 1, 100, 2)

if csv_file:
    # Data load + filter
    df = pd.read_csv(csv_file)
    if REQUIRED_COLS - set(df.columns):
        st.error("Missing required columns.")
        st.stop()

    df['SUBCATEGORY_NORM'] = normalize_subcategory(df['SUBCATEGORY'])
    df = df[df['SUBCATEGORY_NORM'] == subcategory_option.lower()].copy()
    df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
    df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
    df.dropna(subset=['LATITUDE','LONGITUDE'], inplace=True)

    coords_rad = np.radians(df[['LATITUDE','LONGITUDE']])
    eps_rad = radius_m / EARTH_RADIUS_M
    db = DBSCAN(eps=eps_rad, min_samples=int(min_samples), metric='haversine', algorithm='ball_tree')
    df['CLUSTER NUMBER'] = db.fit_predict(coords_rad)
    df['IS_CLUSTERED'] = df['CLUSTER NUMBER'] != -1

    gdf_all = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df['LONGITUDE'], df['LATITUDE']), crs="EPSG:4326")
    wards_gdf = load_wards_uploaded(ward_file) if ward_file else None
    if wards_gdf is not None and not wards_gdf.empty:
        try: gdf_all = gpd.sjoin(gdf_all, wards_gdf, how="left", predicate="within")
        except: pass

    # Excel summary for clusters
    clustered = gdf_all[gdf_all['IS_CLUSTERED']]
    excel_filename = "Clustering_Application_Summary.xlsx"
    if not clustered.empty:
        clustered.to_excel(excel_filename, sheet_name='Clustering Application Summary', index=False)
        hyperlinkify_excel(excel_filename)

    # ---------------- Location Step ----------------
    st.subheader("Step 4: Your Location")
    origin_lat = origin_lon = None

    if 'origin_lat' in st.session_state and 'origin_lon' in st.session_state:
        origin_lat, origin_lon = st.session_state['origin_lat'], st.session_state['origin_lon']
        st.info(f"Using saved location: {origin_lat:.6f}, {origin_lon:.6f}")
    else:
        if HAVE_GEO:
            st.caption("Click to get browser GPS location ↓")
            loc = geolocation()
            if loc and loc.get("latitude") and loc.get("longitude"):
                origin_lat, origin_lon = float(loc["latitude"]), float(loc["longitude"])
                st.session_state["origin_lat"], st.session_state["origin_lon"] = origin_lat, origin_lon
                st.success(f"Browser location: {origin_lat:.6f}, {origin_lon:.6f}")
        if origin_lat is None:
            if st.button("Use my approximate (IP-based) location"):
                ip_lat, ip_lon = get_ip_location()
                if ip_lat and ip_lon:
                    origin_lat, origin_lon = ip_lat, ip_lon
                    st.session_state["origin_lat"], st.session_state["origin_lon"] = origin_lat, origin_lon
                    st.success(f"IP-based location: {origin_lat:.6f}, {origin_lon:.6f}")

    # ---------------- Routing ----------------
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
            waypoints = [(r['LATITUDE'],r['LONGITUDE']) for _,r in pd.DataFrame(sequence_rows).iterrows()]
            nav_url = google_maps_url(origin_lat, origin_lon, waypoints[-1][0], waypoints[-1][1], waypoints=waypoints[:-1])
            st.markdown(f"[🧭 Open route in Google Maps]({nav_url})")

    # ---------------- Map ----------------
    m = folium.Map(location=[df['LATITUDE'].mean(), df['LONGITUDE'].mean()], zoom_start=13)
    mc = MarkerCluster().add_to(m)
    for _, row in gdf_all.iterrows():
        folium.Marker([row['LATITUDE'], row['LONGITUDE']], popup=str(row['ISSUE ID'])).add_to(mc)
    m.save("Clustering_Application_Map.html")

    # ---------------- Downloads ----------------
    if not clustered.empty:
        st.download_button("Download Excel", open(excel_filename, "rb"), file_name=excel_filename)
    st.download_button("Download Map (HTML)", open("Clustering_Application_Map.html", "rb"), file_name="Clustering_Application_Map.html")
    # ---------------- Routing ----------------
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
            waypoints = [(r['LATITUDE'],r['LONGITUDE']) for _,r in pd.DataFrame(sequence_rows).iterrows()]
            nav_url = google_maps_url(origin_lat, origin_lon, waypoints[-1][0], waypoints[-1][1], waypoints=waypoints[:-1])
            st.markdown(f"[🧭 Open route in Google Maps]({nav_url})")

    # ---------------- Map ----------------
    m = folium.Map(location=[df['LATITUDE'].mean(), df['LONGITUDE'].mean()], zoom_start=13)
    mc = MarkerCluster().add_to(m)
    for _, row in gdf_all.iterrows():
        folium.Marker([row['LATITUDE'], row['LONGITUDE']], popup=str(row['ISSUE ID'])).add_to(mc)
    m.save("Clustering_Application_Map.html")

    # ---------------- Downloads ----------------
    if not clustered.empty:
        st.download_button("Download Excel", open(excel_filename, "rb"), file_name=excel_filename)
    st.download_button("Download Map (HTML)", open("Clustering_Application_Map.html", "rb"), file_name="Clustering_Application_Map.html")

