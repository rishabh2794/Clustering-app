import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.cluster import DBSCAN
import folium
from folium.plugins import MarkerCluster
from shapely.geometry import Point
import tempfile
from openpyxl import load_workbook

st.set_page_config(layout="wide")
st.title("Clustering Application v10")

# File uploads
st.subheader("Step 1: Upload Required Files")
csv_file = st.file_uploader("Upload CSV file with issues", type=["csv"])
ward_file = st.file_uploader("Upload WARD boundary file (GeoJSON/JSON/KML, optional)", type=["geojson", "json", "kml"])

# Subcategory selection
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
st.subheader("Step 2: Select Issue Subcategory")
subcategory_option = st.selectbox("Choose issue subcategory to analyze:", subcategory_options)

# Parameters
st.subheader("Step 3: Set Clustering Parameters")
radius_m = st.number_input("Clustering Radius (meters)", min_value=1, max_value=100, value=15)
min_samples = st.number_input("Minimum Issues per Cluster", min_value=1, max_value=100, value=2)

if radius_m < 10 or min_samples < 2:
    st.warning("⚠️ Low values may lead to too many tiny clusters. Proceed with caution.")

# Run logic
if csv_file:
    try:
        df = pd.read_csv(csv_file)
        required_cols = {
            'ISSUE ID', 'CITY', 'ZONE', 'WARD', 'SUBCATEGORY', 'CREATED AT',
            'STATUS', 'LATITUDE', 'LONGITUDE', 'BEFORE PHOTO', 'AFTER PHOTO', 'ADDRESS'
        }
        if not required_cols.issubset(df.columns):
            st.error(f"Missing required columns. Expected: {required_cols}")
        else:
            # Filter and clean
            df = df[df['SUBCATEGORY'].str.strip().str.lower() == subcategory_option.lower()]
            df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
            df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
            df = df.dropna(subset=['LATITUDE', 'LONGITUDE'])
            df['CREATED AT'] = df['CREATED AT'].astype(str)
            df['STATUS'] = df['STATUS'].astype(str)
            df['ADDRESS'] = df['ADDRESS'].astype(str)

            # Clustering
            coords = df[['LATITUDE', 'LONGITUDE']].to_numpy()
            eps_rad = radius_m / 6371000
            db = DBSCAN(eps=eps_rad, min_samples=min_samples, metric='haversine', algorithm='ball_tree')
            df['Cluster'] = db.fit_predict(np.radians(coords))
            clustered = df[df['Cluster'] != -1].copy()

            # Spatial join with wards if available
            if ward_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix="." + ward_file.name.split(".")[-1]) as tmp:
                    tmp.write(ward_file.read())
                    tmp_path = tmp.name
                wards = gpd.read_file(tmp_path)
                if wards.crs != "EPSG:4326":
                    wards = wards.to_crs("EPSG:4326")
                gdf_points = gpd.GeoDataFrame(clustered, geometry=gpd.points_from_xy(clustered['LONGITUDE'], clustered['LATITUDE']), crs="EPSG:4326")
                joined = gpd.sjoin(gdf_points, wards, how="left", predicate="within")
            else:
                joined = clustered.copy()

            # Prepare output summary
            cluster_sizes = joined.groupby('Cluster')['ISSUE ID'].count().rename("NUMBER OF ISSUES")
            summary = joined.copy()
            summary = summary.merge(cluster_sizes, left_on='Cluster', right_index=True, how='left')

            # Rename and select output columns in correct order/case
            summary_sheet = summary[[
                'Cluster', 'NUMBER OF ISSUES', 'ISSUE ID', 'ZONE', 'WARD',
                'SUBCATEGORY', 'CREATED AT', 'STATUS', 'LATITUDE', 'LONGITUDE',
                'BEFORE PHOTO', 'AFTER PHOTO', 'ADDRESS'
            ]].rename(columns={
                'Cluster': 'CLUSTER NUMBER',
                'NUMBER OF ISSUES': 'NUMBER OF ISSUES',
                'ISSUE ID': 'ISSUE ID',
                'ZONE': 'ZONE',
                'WARD': 'WARD',
                'SUBCATEGORY': 'SUBCATEGORY',
                'CREATED AT': 'CREATED AT',
                'STATUS': 'STATUS',
                'LATITUDE': 'LATITUDE',
                'LONGITUDE': 'LONGITUDE',
                'BEFORE PHOTO': 'BEFORE PHOTO',
                'AFTER PHOTO': 'AFTER PHOTO',
                'ADDRESS': 'ADDRESS'
            }).sort_values(['CLUSTER NUMBER', 'CREATED AT'])

            # Excel export with custom sheet name
            excel_filename = "Clustering_Application_Summary.xlsx"
            with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
                summary_sheet.to_excel(writer, index=False, sheet_name='Clustering Application Summary')

            # Make BEFORE PHOTO and AFTER PHOTO hyperlinks if URLs
            wb = load_workbook(excel_filename)
            ws = wb['Clustering Application Summary']
            for row in range(2, ws.max_row + 1):
                for col_idx in (11, 12):  # BEFORE PHOTO (K), AFTER PHOTO (L)
                    link = ws.cell(row, col_idx).value
                    if link and isinstance(link, str) and link.startswith('http'):
                        ws.cell(row, col_idx).hyperlink = link
                        ws.cell(row, col_idx).style = "Hyperlink"
            wb.save(excel_filename)

            # Map display options
            st.subheader("Step 4: Map Display Options")
            st.write(f"**Total clusters found:** {summary_sheet['CLUSTER NUMBER'].nunique()}")
            if summary_sheet['CLUSTER NUMBER'].nunique() > 100:
                st.info("We recommend **Dynamic Clustering (Type 2 Map)** for maps with more than 100 clusters.")
            else:
                st.info("For ≤ 100 clusters, all markers can be shown at once.")

            map_type = st.radio(
                "Select map type:",
                [
                    "Show all cluster markers (Type 1 Map)",
                    "Use Dynamic Clustering (Type 2 Map)"
                ],
                index=1 if summary_sheet['CLUSTER NUMBER'].nunique() > 100 else 0
            )

            m = folium.Map(location=[summary_sheet['LATITUDE'].mean(), summary_sheet['LONGITUDE'].mean()], zoom_start=13)
            if ward_file:
                folium.GeoJson(wards, name="Wards").add_to(m)

            if map_type == "Show all cluster markers (Type 1 Map)":
                for _, row in summary_sheet.iterrows():
                    folium.CircleMarker(
                        location=[row['LATITUDE'], row['LONGITUDE']],
                        radius=7,
                        color='red',
                        fill=True,
                        fill_color='red',
                        fill_opacity=0.85,
                        popup=(
                            f"Cluster {row['CLUSTER NUMBER']}<br>"
                            f"Issue ID: {row['ISSUE ID']}<br>"
                            f"Ward: {row['WARD']}<br>"
                            f"Lat: {row['LATITUDE']}, Lon: {row['LONGITUDE']}"
                        )
                    ).add_to(m)
            else:
                mc = MarkerCluster(name="Hotspots").add_to(m)
                for _, row in summary_sheet.iterrows():
                    folium.Marker(
                        location=[row['LATITUDE'], row['LONGITUDE']],
                        popup=(
                            f"Cluster {row['CLUSTER NUMBER']}<br>"
                            f"Issue ID: {row['ISSUE ID']}<br>"
                            f"Ward: {row['WARD']}<br>"
                            f"Lat: {row['LATITUDE']}, Lon: {row['LONGITUDE']}"
                        ),
                        icon=folium.Icon(color='red', icon='info-sign')
                    ).add_to(mc)

            folium.LayerControl().add_to(m)
            html_filename = "Clustering_Application_Map.html"
            m.save(html_filename)

            # Download buttons
            st.subheader("Step 5: Download Outputs")
            with open(excel_filename, "rb") as f:
                st.download_button("Download Clustering Application Summary (Excel)", f, file_name=excel_filename)

            with open(html_filename, "rb") as f:
                st.download_button("Download Clustering Application Map (HTML)", f, file_name=html_filename)

            st.success(f"✅ Generated summary for {summary_sheet['CLUSTER NUMBER'].nunique()} clusters.")
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload the required CSV file to proceed.")
