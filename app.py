# Unified Clustering + Batch Routing (No External Geo Component) v14 (app.py)
# ---------------------------------------------------------------------------
# What this app does
# - Ingests a CSV of issue/ticket points (lat/lon, status, photos, etc.)
# - Clusters them with DBSCAN (haversine distance)
# - (Optional) Spatially joins to ward boundaries (GeoJSON/JSON/KML)
# - Auto-detects your location via **browser JavaScript** (no separate geolocation component)
#   with manual lat/lon fallback
# - Builds a continuous Google Maps route to the next N tickets (nearest-neighbor, default 10)
# - Shows a Folium map of ALL filtered tickets and highlights the batch
#   (first = green, rest = orange), optional polyline drawn between stops
# - Exports: Excel summary of clustered points + HTML map
#
# How to run locally:
#   pip install streamlit pandas numpy geopandas scikit-learn shapely folium openpyxl fiona streamlit-js-eval
#   streamlit run app.py
#
# On Streamlit Cloud:
#   - Add the packages above to requirements.txt (at repo root)
#   - (Optional) runtime.txt with: python-3.11

import math
import tempfile
import urllib.parse

import numpy as np
import pandas as pd
import streamlit as st
import geopandas as gpd
from sklearn.cluster import DBSCAN

import folium
from folium.plugins import MarkerCluster
from shapely.geometry import Point
from openpyxl import load_workbook

# JavaScript eval for in-browser geolocation (no external Streamlit component)
try:
    from streamlit_js_eval import streamlit_js_eval
    HAVE_JS_EVAL = True
except Exception:
    HAVE_JS_EVAL = False

# Optional: improve KML handling if available
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

st.set_page_config(layout="wide")
st.title("Unified Clustering + Batch Routing — Auto Geo (No External Component)")

with st.sidebar:
    st.markdown("### Tips")
    st.markdown(
        "- CSV must include LATITUDE/LONGITUDE in decimal degrees.\n"
        "- Clustering radius is in **meters** (converted to **radians** for haversine).\n"
        "- If KML ward reading fails, convert to GeoJSON and try again.\n"
        "- Browser will ask for Location permission for auto geolocation."
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
    # Google supports driving/walking/bicycling/transit
    mode = mode if mode in {"driving","walking","bicycling","transit"} else "driving"
    travel = f"&travelmode={mode}"
    wp = ""
    if waypoints:
        wp_str = "|".join([f"{lat},{lon}" for (lat, lon) in waypoints])
        # Encode but keep separators
        wp_enc = urllib.parse.quote(wp_str, safe="|,")
        wp = f"&waypoints={wp_enc}"
    return f"{base}{origin}{dest}{travel}{wp}"

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

def load_wards_uploaded(file) -> gpd.GeoDataFrame | None:
    """Read GeoJSON/JSON/KML wards into EPSG:4326. Return None on failure and show an error."""
    try:
        suffix = file.name.split(".")[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix="." + suffix) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        if suffix in ("geojson", "json"):
            wards = gpd.read_file(tmp_path)
        elif suffix == "kml":
            if HAS_FIONA:
                import fiona
                layers = fiona.listlayers(tmp_path)
                if not layers:
                    st.error("KML has no layers. Convert to GeoJSON and try again.")
                    return None
                wards = None
                for lyr in layers:
                    gdf_try = gpd.read_file(tmp_path, driver="KML", layer=lyr)
                    if not gdf_try.empty:
                        wards = gdf_try
                        break
                if wards is None or wards.empty:
                    st.error("KML layers are empty. Convert to GeoJSON and try again.")
                    return None
            else:
                try:
                    wards = gpd.read_file(tmp_path, driver="KML")
                except Exception as e:
                    st.error(
                        "Could not read KML. Install `fiona`/`libkml`, or convert KML to GeoJSON and upload.\n\n"
                        f"Error: {e}"
                    )
                    return None
        else:
            st.error("Unsupported ward file type. Please upload GeoJSON/JSON/KML.")
            return None

        if wards is None or wards.empty:
            st.error("Ward file loaded but contains no features.")
            return None

        if wards.crs is None:
            st.info("Ward file had no CRS. Assuming EPSG:4326 (WGS84).")
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
# UI — Inputs
# -------------------------
st.subheader("Step 1: Upload Required Files")
csv_file = st.file_uploader("Upload CSV file with issues", type=["csv"])
ward_file = st.file_uploader("Upload WARD boundary file (GeoJSON/JSON/KML, optional)", type=["geojson", "json", "kml"])

st.subheader("Step 2: Select Issue Subcategory")
subcategory_option = st.selectbox("Choose issue subcategory to analyze:", subcategory_options)

st.subheader("Step 3: Set Clustering Parameters")
radius_m = st.number_input("Clustering Radius (meters)", min_value=1, max_value=2000, value=15)
min_samples = st.number_input("Minimum Issues per Cluster", min_value=1, max_value=100, value=2)
if radius_m < 10 or min_samples < 2:
    st.warning("⚠️ Low values may lead to too many tiny clusters. Proceed with caution.")

# -------------------------
# Core logic
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

        # Ensure string-friendly columns
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

        # --- Optional Ward Join ---
        wards_gdf = None
        if ward_file is not None:
            wards_gdf = load_wards_uploaded(ward_file)

        # GeoDataFrame of ALL points (not only clusters)
        gdf_all = gpd.GeoDataFrame(
            df.copy(), geometry=gpd.points_from_xy(df['LONGITUDE'].astype(float), df['LATITUDE'].astype(float)), crs="EPSG:4326"
        )
        if wards_gdf is not None and not wards_gdf.empty:
            try:
                gdf_all = gpd.sjoin(gdf_all, wards_gdf, how="left", predicate="within")
            except Exception as e:
                st.warning(f"Spatial join failed; proceeding without ward attribution. Error: {e}")

        # --- Clustered subset for Excel summary ONLY ---
        clustered = gdf_all[gdf_all['IS_CLUSTERED']].copy()
        if clustered.empty:
            st.warning("DBSCAN found no clusters (all points are noise). You can still use navigation and map with all points.")

        # Cluster sizes for non-noise
        if not clustered.empty:
            sizes = clustered.groupby('CLUSTER NUMBER')['ISSUE ID'].count().rename("NUMBER OF ISSUES")
            summary_sheet = clustered.drop(columns=['SUBCATEGORY_NORM']).copy()
            summary_sheet = summary_sheet.merge(sizes, on='CLUSTER NUMBER', how='left')
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

        # --- Excel export (clustered points only) ---
        excel_filename = "Clustering_Application_Summary.xlsx"
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            summary_sheet.to_excel(writer, index=False, sheet_name='Clustering Application Summary')
        hyperlinkify_excel(excel_filename)

        # -------------------------
        # Step 4A — Navigate to Nearest Tickets (Batch up to 10)
        # -------------------------
        st.subheader("Step 4A: Navigate to Nearest Tickets (Batch)")

        unique_statuses = sorted(gdf_all['STATUS'].dropna().astype(str).unique().tolist()) if 'STATUS' in gdf_all.columns else []
        default_statuses = [s for s in unique_statuses if s.lower() in ("open", "pending", "in progress")] or unique_statuses
        include_statuses = st.multiselect("Eligible ticket statuses", options=unique_statuses, default=default_statuses)

        wards_in_data = sorted(gdf_all['WARD'].dropna().astype(str).unique().tolist()) if 'WARD' in gdf_all.columns else []
        ward_filter = st.multiselect("Limit to ward(s) (optional)", options=wards_in_data, default=[])

        travel_mode = st.selectbox("Travel mode", ["driving", "walking", "bicycling"], index=0)
        batch_size = st.slider("Batch size (next N tickets)", min_value=1, max_value=10, value=10)

        # Get current location (JS auto + manual fallback)
        origin_lat = origin_lon = None
        st.markdown("### Your Location")
        colA, colB = st.columns(2)
        with colA:
            if HAVE_JS_EVAL:
                st.caption("Attempting auto location (allow browser permission)…")
                js = """
                new Promise((resolve) => {
                  if (!navigator.geolocation) { resolve(null); return; }
                  navigator.geolocation.getCurrentPosition(
                    (pos) => resolve([pos.coords.latitude, pos.coords.longitude]),
                    (err) => resolve(null),
                    { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
                  );
                })
                """
                coords = streamlit_js_eval(js_expressions=js, key="geo_fallback")
                if coords and isinstance(coords, list) and len(coords) == 2:
                    origin_lat, origin_lon = float(coords[0]), float(coords[1])
                    st.success(f"Auto location: {origin_lat:.6f}, {origin_lon:.6f}")
            else:
                st.info("Install 'streamlit-js-eval' for auto geolocation, or use manual entry.")
        with colB:
            manual_lat = st.text_input("Manual latitude (fallback)", value="")
            manual_lon = st.text_input("Manual longitude (fallback)", value="")
            if manual_lat.strip() and manual_lon.strip():
                try:
                    origin_lat = float(manual_lat)
                    origin_lon = float(manual_lon)
                    st.success(f"Manual location: {origin_lat:.6f}, {origin_lon:.6f}")
                except Exception:
                    st.error("Invalid manual coordinates. Example: 26.8467 (lat), 80.9462 (lon)")

        # Build pool from ALL filtered points
        pool = gdf_all.copy()
        if include_statuses:
            pool = pool[pool['STATUS'].astype(str).isin(include_statuses)]
        if ward_filter:
            pool = pool[pool['WARD'].astype(str).isin([str(w) for w in ward_filter])]
        if st.session_state.visited_ticket_ids:
            pool = pool[~pool['ISSUE ID'].astype(str).isin(st.session_state.visited_ticket_ids)]
        if st.session_state.skipped_ticket_ids:
            pool = pool[~pool['ISSUE ID'].astype(str).isin(st.session_state.skipped_ticket_ids)]

        st.write(f"Eligible tickets remaining: **{len(pool)}**")

        # Greedy nearest-neighbor batch
        sequence_rows = []
        route_coords = None
        if origin_lat is not None and origin_lon is not None and not pool.empty:
            cur_lat, cur_lon = origin_lat, origin_lon
            pool2 = pool.copy()
            for _ in range(min(batch_size, len(pool2))):
                pool2['__dist_m'] = pool2.apply(lambda r: haversine_m(cur_lat, cur_lon, r['LATITUDE'], r['LONGITUDE']), axis=1)
                nxt = pool2.sort_values('__dist_m', ascending=True).iloc[0]
                sequence_rows.append(nxt)
                cur_lat, cur_lon = float(nxt['LATITUDE']), float(nxt['LONGITUDE'])
                pool2 = pool2[pool2['ISSUE ID'] != nxt['ISSUE ID']]

            # Build navigation URL with waypoints
            waypoints = [(float(r['LATITUDE']), float(r['LONGITUDE'])) for r in sequence_rows]
            first, last = waypoints[0], waypoints[-1]
            mids = waypoints[:-1] if len(waypoints) > 1 else []
            nav_url = google_maps_url(origin_lat, origin_lon, last[0], last[1], mode=travel_mode, waypoints=mids)

            # Approx total distance & list
            total_m = 0.0
            prev = (origin_lat, origin_lon)
            leg_dists = []
            for (lat, lon) in waypoints:
                d = haversine_m(prev[0], prev[1], lat, lon)
                leg_dists.append(int(d))
                total_m += d
                prev = (lat, lon)
            eta_min = total_m / 8.3 / 60.0  # ~30 km/h baseline

            st.success(
                f"Batch route ready: **{len(sequence_rows)}** tickets | Total distance ≈ **{int(total_m)} m** | ETA ≈ **{eta_min:.1f} min**"
            )
            st.markdown(f"[🧭 Open continuous navigation in Google Maps]({nav_url})")

            # Show ordered list
            list_df = pd.DataFrame({
                "#": list(range(1, len(sequence_rows)+1)),
                "ISSUE ID": [str(r['ISSUE ID']) for r in sequence_rows],
                "WARD": [str(r.get('WARD', '')) for r in sequence_rows],
                "STATUS": [str(r.get('STATUS', '')) for r in sequence_rows],
                "Leg dist (m)": leg_dists
            })
            st.dataframe(list_df, use_container_width=True)

            # For map polyline: origin + stops
            route_coords = [[origin_lat, origin_lon]] + [[float(r['LATITUDE']), float(r['LONGITUDE'])] for r in sequence_rows]

            # Highlight on map (first = green, rest = orange)
            st.session_state.current_target_id = str(sequence_rows[0]['ISSUE ID'])
            st.session_state.batch_target_ids = {str(r['ISSUE ID']) for r in sequence_rows}

            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ Mark first as Done (visited)"):
                    st.session_state.visited_ticket_ids.add(str(sequence_rows[0]['ISSUE ID']))
                    st.rerun()
            with c2:
                if st.button("⏭️ Skip first ticket"):
                    st.session_state.skipped_ticket_ids.add(str(sequence_rows[0]['ISSUE ID']))
                    st.rerun()
        else:
            st.info("Provide your location to compute a batch route.")

        # -------------------------
        # Step 5 — Map Display (ALL filtered points, batch highlighted)
        # -------------------------
        st.subheader("Step 5: Map Display Options")
        st.write(f"**Total clusters found (non-noise)**: {df[df['CLUSTER NUMBER']!=-1]['CLUSTER NUMBER'].nunique()}")
        center_on_first = st.checkbox("Center map on first stop (if any)", value=False)
        draw_polyline = st.checkbox("Draw route polyline on map (origin → stops)", value=True)

        map_type = st.radio(
            "Select map type:",
            [
                "Show all markers (Type 1)",
                "Use Dynamic Clustering (Type 2)"
            ],
            index=0
        )

        # Map center
        if center_on_first and sequence_rows:
            map_center = [float(sequence_rows[0]['LATITUDE']), float(sequence_rows[0]['LONGITUDE'])]
            zoom_level = 16
        else:
            map_center = [float(df['LATITUDE'].mean()), float(df['LONGITUDE'].mean())]
            zoom_level = 13

        m = folium.Map(location=map_center, zoom_start=zoom_level)

        # Ward overlay (if provided)
        if wards_gdf is not None and not wards_gdf.empty:
            try:
                folium.GeoJson(wards_gdf, name="Wards").add_to(m)
            except Exception:
                pass

        target_id = st.session_state.get('current_target_id')
        batch_ids = st.session_state.get('batch_target_ids', set())

        if map_type == "Show all markers (Type 1)":
            for _, row in gdf_all.iterrows():
                rid = str(row['ISSUE ID'])
                is_first = (rid == str(target_id)) if target_id else False
                in_batch = rid in batch_ids
                if is_first:
                    color, size = 'green', 10
                elif in_batch:
                    color, size = 'orange', 9
                else:
                    color, size = 'red', 7
                folium.CircleMarker(
                    location=[float(row['LATITUDE']), float(row['LONGITUDE'])],
                    radius=size,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.9 if (is_first or in_batch) else 0.85,
                    popup=(
                        f"Cluster {int(row['CLUSTER NUMBER']) if 'CLUSTER NUMBER' in row else ''}<br>"
                        f"Issue ID: {row['ISSUE ID']}<br>"
                        f"Ward: {row['WARD']}<br>"
                        f"Lat: {row['LATITUDE']}, Lon: {row['LONGITUDE']}"
                    )
                ).add_to(m)
        else:
            mc = MarkerCluster(name="Tickets").add_to(m)
            for _, row in gdf_all.iterrows():
                rid = str(row['ISSUE ID'])
                is_first = (rid == str(target_id)) if target_id else False
                in_batch = rid in batch_ids
                icon_color = 'green' if is_first else ('orange' if in_batch else 'red')
                folium.Marker(
                    location=[float(row['LATITUDE']), float(row['LONGITUDE'])],
                    popup=(
                        f"Cluster {int(row['CLUSTER NUMBER']) if 'CLUSTER NUMBER' in row else ''}<br>"
                        f"Issue ID: {row['ISSUE ID']}<br>"
                        f"Ward: {row['WARD']}<br>"
                        f"Lat: {row['LATITUDE']}, Lon: {row['LONGITUDE']}"
                    ),
                    icon=folium.Icon(color=icon_color, icon='info-sign')
                ).add_to(mc)

        # Draw route polyline if requested
        if draw_polyline and route_coords:
            try:
                folium.PolyLine(route_coords, weight=4, opacity=0.85).add_to(m)
            except Exception:
                pass

        folium.LayerControl().add_to(m)
        html_filename = "Clustering_Application_Map.html"
        m.save(html_filename)

        # -------------------------
        # Step 6 — Downloads
        # -------------------------
        st.subheader("Step 6: Download Outputs")
        with open(excel_filename, "rb") as f:
            st.download_button("Download Clustering Application Summary (Excel)", f, file_name=excel_filename)
        with open(html_filename, "rb") as f:
            st.download_button("Download Clustering Application Map (HTML)", f, file_name=html_filename)

        st.success("✅ Processing complete. Batch route + map + downloads are ready.")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload the required CSV file to proceed.")
