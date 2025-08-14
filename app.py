# Cloud‑Safe Clustering + Batch Routing v16 (app.py)
# --------------------------------------------------
# Zero‑friction on Streamlit Cloud: no GeoPandas/Fiona. Origin is OPTIONAL.
# If origin is omitted, the Google Maps link starts from the user's live GPS
# (Maps handles location), so no in‑app geolocation is required.
#
# Features
# - Upload CSV of issues (required columns below)
# - Filter by subcategory
# - DBSCAN clustering with Haversine distance
# - Ward overlay (GeoJSON/JSON) on the Folium map (no spatial join)
# - Batch routing to next N tickets (nearest‑neighbor)
# - Google Maps deep link with waypoints (omits origin if not provided)
# - Optional manual origin for ETA & polyline; NOT needed for navigation
# - Exports: Excel summary of clustered points + HTML map
#
# How to run (Cloud or local):
#   requirements.txt →
#       streamlit
#       pandas
#       numpy
#       scikit-learn
#       folium
#       openpyxl
#   streamlit run app.py

import json
import math
import urllib.parse
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import DBSCAN
import folium
from folium.plugins import MarkerCluster
from openpyxl import load_workbook

# -------------------------
# Constants & Inputs
# -------------------------
EARTH_RADIUS_M = 6_371_000.0
REQUIRED_COLS = {
    'ISSUE ID', 'CITY', 'ZONE', 'WARD', 'SUBCATEGORY', 'CREATED AT',
    'STATUS', 'LATITUDE', 'LONGITUDE', 'BEFORE PHOTO', 'AFTER PHOTO', 'ADDRESS'
}

st.set_page_config(layout="wide")
st.title("Cloud‑Safe Clustering + Batch Routing — Origin Optional")

with st.sidebar:
    st.markdown("### Tips")
    st.markdown(
        "- CSV must include LATITUDE/LONGITUDE in decimal degrees.\n"
        "- Clustering radius is in **meters** (internally converted to radians).\n"
        "- Ward overlay accepts **GeoJSON/JSON** (no KML in this build).\n"
        "- Navigation link works from your **current GPS** even if you don't enter origin."
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
# Helpers
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
    parts = []
    # If origin is None → omit; Maps will use the user's current GPS automatically
    if origin_lat is not None and origin_lon is not None:
        parts.append(f"origin={origin_lat},{origin_lon}")
    parts.append(f"destination={dest_lat},{dest_lon}")
    mode = mode if mode in {"driving","walking","bicycling","transit"} else "driving"
    parts.append(f"travelmode={mode}")
    if waypoints:
        wp_str = "|".join([f"{lat},{lon}" for (lat, lon) in waypoints])
        parts.append(f"waypoints={urllib.parse.quote(wp_str, safe='|,')}")
    return base + "&" + "&".join(parts)

def hyperlinkify_excel(excel_path: str, sheet_name: str = "Clustering Application Summary") -> None:
    try:
        wb = load_workbook(excel_path)
        ws = wb[sheet_name]
        for row in range(2, ws.max_row + 1):
            for col_idx in (11, 12):  # BEFORE (K), AFTER (L)
                link = ws.cell(row, col_idx).value
                if link and isinstance(link, str) and link.startswith(("http://", "https://")):
                    ws.cell(row, col_idx).hyperlink = link
                    ws.cell(row, col_idx).style = "Hyperlink"
        wb.save(excel_path)
    except Exception as e:
        st.warning(f"Excel hyperlinking skipped: {e}")

# -------------------------
# Uploads & Parameters
# -------------------------
st.subheader("Step 1: Upload Required Files")
csv_file = st.file_uploader("Upload CSV file with issues", type=["csv"])
ward_file = st.file_uploader("Upload WARD boundary file (GeoJSON/JSON overlay, optional)", type=["geojson", "json"])

st.subheader("Step 2: Select Issue Subcategory")
subcategory_option = st.selectbox("Choose issue subcategory to analyze:", subcategory_options)

st.subheader("Step 3: Set Clustering Parameters")
radius_m = st.number_input("Clustering Radius (meters)", min_value=1, max_value=2000, value=15)
min_samples = st.number_input("Minimum Issues per Cluster", min_value=1, max_value=100, value=2)
if radius_m < 10 or min_samples < 2:
    st.warning("⚠️ Low values may lead to too many tiny clusters. Proceed with caution.")

# Optional: manual origin for ETA/polyline (NOT required for navigation)
with st.expander("Optional: Enter a manual origin (for ETA & map polyline)"):
    col_o1, col_o2 = st.columns(2)
    with col_o1:
        manual_lat = st.text_input("Origin latitude (optional)", value="")
    with col_o2:
        manual_lon = st.text_input("Origin longitude (optional)", value="")
    origin_lat = origin_lon = None
    if manual_lat.strip() and manual_lon.strip():
        try:
            origin_lat = float(manual_lat)
            origin_lon = float(manual_lon)
            st.success(f"Using manual origin: {origin_lat:.6f}, {origin_lon:.6f}")
        except Exception:
            st.error("Invalid origin coordinates. Example: 26.8467 (lat), 80.9462 (lon)")

# -------------------------
# Core
# -------------------------
if csv_file:
    try:
        df = pd.read_csv(csv_file)
        missing = sorted(list(REQUIRED_COLS - set(df.columns)))
        if missing:
            st.error(f"Missing required columns: {missing}")
            st.stop()

        # Filter & clean
        df = df.copy()
        df['SUBCATEGORY_NORM'] = normalize_subcategory(df['SUBCATEGORY'])
        desired = subcategory_option.strip().lower()
        df = df[df['SUBCATEGORY_NORM'] == desired].copy()
        if df.empty:
            st.info("No rows found for the selected subcategory.")
            st.stop()

        df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
        df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
        df = df.dropna(subset=['LATITUDE', 'LONGITUDE'])
        if df.empty:
            st.info("All rows had invalid/missing coordinates after cleaning.")
            st.stop()

        # String-friendly columns
        for col in ['CREATED AT', 'STATUS', 'ADDRESS', 'BEFORE PHOTO', 'AFTER PHOTO', 'ISSUE ID', 'ZONE', 'WARD', 'CITY']:
            if col in df.columns:
                df[col] = df[col].astype(str)

        # --- DBSCAN (Haversine) ---
        coords_deg = df[['LATITUDE', 'LONGITUDE']].to_numpy()
        if coords_deg.shape[0] < max(1, int(min_samples)):
            st.info(f"Need at least {min_samples} points to form a cluster. Only {coords_deg.shape[0]} rows available.")
            st.stop()
        coords_rad = np.radians(coords_deg)
        eps_rad = float(radius_m) / EARTH_RADIUS_M
        db = DBSCAN(eps=eps_rad, min_samples=int(min_samples), metric='haversine', algorithm='ball_tree')
        labels = db.fit_predict(coords_rad)
        df['CLUSTER NUMBER'] = labels  # -1 = noise
        df['IS_CLUSTERED'] = df['CLUSTER NUMBER'] != -1

        # --- Excel summary (clustered only) ---
        clustered = df[df['IS_CLUSTERED']].copy()
        if clustered.empty:
            st.warning("DBSCAN found no clusters (all points are noise). You can still route & map with all points.")
            summary_sheet = pd.DataFrame(columns=[
                'CLUSTER NUMBER', 'NUMBER OF ISSUES', 'ISSUE ID', 'ZONE', 'WARD',
                'SUBCATEGORY', 'CREATED AT', 'STATUS', 'LATITUDE', 'LONGITUDE',
                'BEFORE PHOTO', 'AFTER PHOTO', 'ADDRESS'
            ])
        else:
            sizes = clustered.groupby('CLUSTER NUMBER')['ISSUE ID'].count().rename("NUMBER OF ISSUES")
            summary_sheet = clustered.merge(sizes, on='CLUSTER NUMBER', how='left')
            summary_sheet = summary_sheet[[
                'CLUSTER NUMBER', 'NUMBER OF ISSUES', 'ISSUE ID', 'ZONE', 'WARD',
                'SUBCATEGORY', 'CREATED AT', 'STATUS', 'LATITUDE', 'LONGITUDE',
                'BEFORE PHOTO', 'AFTER PHOTO', 'ADDRESS'
            ]].sort_values(['CLUSTER NUMBER', 'CREATED AT'])

        excel_filename = "Clustering_Application_Summary.xlsx"
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            summary_sheet.to_excel(writer, index=False, sheet_name='Clustering Application Summary')
        hyperlinkify_excel(excel_filename)

        # -------------------------
        # Batch routing (nearest‑neighbor)
        # -------------------------
        st.subheader("Step 4A: Navigate to Nearest Tickets (Batch)")

        unique_statuses = sorted(df['STATUS'].dropna().astype(str).unique().tolist()) if 'STATUS' in df.columns else []
        default_statuses = [s for s in unique_statuses if s.lower() in ("open", "pending", "in progress")] or unique_statuses
        include_statuses = st.multiselect("Eligible ticket statuses", options=unique_statuses, default=default_statuses)

        wards_in_data = sorted(df['WARD'].dropna().astype(str).unique().tolist()) if 'WARD' in df.columns else []
        ward_filter = st.multiselect("Limit to ward(s) (optional)", options=wards_in_data, default=[])

        travel_mode = st.selectbox("Travel mode", ["driving", "walking", "bicycling", "transit"], index=0)
        batch_size = st.slider("Batch size (next N tickets)", min_value=1, max_value=10, value=10)

        # Build pool
        pool = df.copy()
        if include_statuses:
            pool = pool[pool['STATUS'].astype(str).isin(include_statuses)]
        if ward_filter:
            pool = pool[pool['WARD'].astype(str).isin([str(w) for w in ward_filter])]

        st.write(f"Eligible tickets after filters: **{len(pool)}**")

        # Greedy NN: seed from origin if provided else from centroid
        sequence_rows = []
        if not pool.empty:
            if origin_lat is not None and origin_lon is not None:
                cur_lat, cur_lon = origin_lat, origin_lon
            else:
                cur_lat, cur_lon = float(pool['LATITUDE'].mean()), float(pool['LONGITUDE'].mean())

            pool2 = pool.copy()
            for _ in range(min(batch_size, len(pool2))):
                pool2['__dist_m'] = pool2.apply(lambda r: haversine_m(cur_lat, cur_lon, r['LATITUDE'], r['LONGITUDE']), axis=1)
                nxt = pool2.sort_values('__dist_m', ascending=True).iloc[0]
                sequence_rows.append(nxt)
                cur_lat, cur_lon = float(nxt['LATITUDE']), float(nxt['LONGITUDE'])
                pool2 = pool2[pool2['ISSUE ID'] != nxt['ISSUE ID']]

        # Build navigation URL (omit origin if not provided)
        if sequence_rows:
            waypoints = [(float(r['LATITUDE']), float(r['LONGITUDE'])) for r in sequence_rows]
            last = waypoints[-1]
            mids = waypoints[:-1] if len(waypoints) > 1 else []
            nav_url = google_maps_url(origin_lat, origin_lon, last[0], last[1], mode=travel_mode, waypoints=mids)
            st.markdown(f"[🧭 Open continuous navigation in Google Maps]({nav_url})")

            # Optional ETA (needs origin)
            if origin_lat is not None and origin_lon is not None:
                total_m = 0.0
                prev = (origin_lat, origin_lon)
                for (lat, lon) in waypoints:
                    total_m += haversine_m(prev[0], prev[1], lat, lon)
                    prev = (lat, lon)
                eta_min = total_m / 8.3 / 60.0  # ~30 km/h baseline
                st.caption(f"Approx distance ≈ {int(total_m)} m | ETA ≈ {eta_min:.1f} min (estimate)")

            list_df = pd.DataFrame({
                "#": list(range(1, len(sequence_rows)+1)),
                "ISSUE ID": [str(r['ISSUE ID']) for r in sequence_rows],
                "WARD": [str(r.get('WARD', '')) for r in sequence_rows],
                "STATUS": [str(r.get('STATUS', '')) for r in sequence_rows]
            })
            st.dataframe(list_df, use_container_width=True)
        else:
            st.info("No tickets found for routing with the current filters.")

        # -------------------------
        # Map (with optional ward overlay & polyline)
        # -------------------------
        st.subheader("Step 5: Map Display")
        center = [float(df['LATITUDE'].mean()), float(df['LONGITUDE'].mean())]
        m = folium.Map(location=center, zoom_start=13)

        # Ward overlay (GeoJSON/JSON only)
        if ward_file is not None:
            try:
                data = json.load(ward_file)
                folium.GeoJson(data, name="Wards").add_to(m)
            except Exception as e:
                st.warning(f"Ward overlay failed: {e}")

        # Markers
        batch_ids = {str(r['ISSUE ID']) for r in sequence_rows} if sequence_rows else set()
        first_id = str(sequence_rows[0]['ISSUE ID']) if sequence_rows else None

        use_cluster = st.radio("Marker mode", ["All markers", "Dynamic clustering"], index=0)
        tgt = m if use_cluster == "All markers" else MarkerCluster(name="Tickets").add_to(m)

        for _, row in df.iterrows():
            rid = str(row['ISSUE ID'])
            lat, lon = float(row['LATITUDE']), float(row['LONGITUDE'])
            is_first = (rid == first_id)
            in_batch = (rid in batch_ids)
            color = 'green' if is_first else ('orange' if in_batch else 'red')
            popup = (
                f"Cluster {int(row['CLUSTER NUMBER']) if 'CLUSTER NUMBER' in row else ''}<br>"
                f"Issue ID: {row['ISSUE ID']}<br>Ward: {row['WARD']}<br>"
                f"Lat: {row['LATITUDE']}, Lon: {row['LONGITUDE']}"
            )
            marker = folium.CircleMarker(location=[lat, lon], radius=9 if in_batch else 7,
                                         color=color, fill=True, fill_color=color, fill_opacity=0.9, popup=popup)
            (tgt if use_cluster == "Dynamic clustering" else m).add_child(marker)

        # Optional polyline from origin (if provided) through batch
        if sequence_rows and origin_lat is not None and origin_lon is not None:
            route_coords = [[origin_lat, origin_lon]] + [[float(r['LATITUDE']), float(r['LONGITUDE'])] for r in sequence_rows]
            try:
                folium.PolyLine(route_coords, weight=4, opacity=0.85).add_to(m)
            except Exception:
                pass

        folium.LayerControl().add_to(m)
        html_filename = "Clustering_Application_Map.html"
        m.save(html_filename)

        # -------------------------
        # Downloads
        # -------------------------
        st.subheader("Step 6: Download Outputs")
        with open("Clustering_Application_Summary.xlsx", "rb") as f:
            st.download_button("Download Clustering Application Summary (Excel)", f, file_name="Clustering_Application_Summary.xlsx")
        with open(html_filename, "rb") as f:
            st.download_button("Download Clustering Application Map (HTML)", f, file_name=html_filename)

        st.success("✅ Ready: clustering, batch route, and downloads are available.")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload the required CSV file to proceed.")
