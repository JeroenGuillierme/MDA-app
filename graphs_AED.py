import pandas as pd
import numpy as np
import geopandas as gpd
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objs as go
from sklearn.cluster import DBSCAN


# ----------------------------------------------------------------------------------------------------------------------
# IMPORTING DATA
# ----------------------------------------------------------------------------------------------------------------------

url2 = 'https://raw.githubusercontent.com/JeroenGuillierme/Project-MDA/main/Data/'

gdf_interventions_with_both_count = pd.read_csv(
    f'{url2}gdf_interventions.csv')

# Load Belgium with regions shapefile
belgium_with_provinces_boundary = gpd.read_file(f'{url2}BELGIUM_-_Provinces.geojson')

# ----------------------------------------------------------------------------------------------------------------------
# HIGH-RISK-AREAS
# ----------------------------------------------------------------------------------------------------------------------

# Define high-risk based on response time, incident frequency and aed density
response_time_threshold = 8  # minutes
incident_density_threshold = 5  # more than five previous interventions on that location area
aed_density_threshold = 5  # less than 5 aeds in the 3 km³ grid for that location


# Identify high-risk areas
high_risk_areas = gdf_interventions_with_both_count[
    (gdf_interventions_with_both_count['T3-T0'] > response_time_threshold) &
    (gdf_interventions_with_both_count['incident_count'] > incident_density_threshold) &
    (gdf_interventions_with_both_count['aed_count'] < aed_density_threshold)
    ]

# Reset indices
high_risk_areas.reset_index(drop=True, inplace = True)

# Drop duplicates because same locations multiple times in dataset for representing response time of different vector types
df = high_risk_areas[['Latitude', 'Longitude']].drop_duplicates()

# Reset indices
df.reset_index(drop=True, inplace=True)


# ----------------------------------------------------------------------------------------------------------------------
# DBSCAN
# ----------------------------------------------------------------------------------------------------------------------

# Hyper parameters
min_samples = 4
eps = 0.023

# Initialize DBSCAN
dbscan = DBSCAN(eps=eps, min_samples=min_samples)

# Perform clustering
df.loc[:, 'cluster'] = dbscan.fit_predict(df)

# Drop noise points (cluster == -1)
# Filter out noise points (cluster == -1)
df_filtered = df[df['cluster'] != -1]

cluster_centers = df_filtered.groupby('cluster')[['Latitude', 'Longitude']].mean().reset_index()

# Determine cluster sizes
cluster_sizes = df_filtered['cluster'].value_counts().reset_index()
cluster_sizes.columns = ['cluster', 'size']

# Combine locations of clusters centers and their sizes
clusters_df = pd.merge(cluster_centers, cluster_sizes, on='cluster')


# ----------------------------------------------------------------------------------------------------------------------
# Plot Results
# ----------------------------------------------------------------------------------------------------------------------

# Re-project to a suitable projected CRS (e.g., UTM zone 31N for Belgium)
belgium_proj = belgium_with_provinces_boundary.to_crs(epsg=32631)

# Calculate the centroid of the re-projected geometry
centroid = belgium_proj.geometry.centroid

# Now, we convert this projected centroid back to geographic CRS for map display purposes
centroid_geo = centroid.to_crs(epsg=4326)
mean_lat = centroid_geo.y.mean()
mean_lon = centroid_geo.x.mean()

# Create the figure
fig = go.Figure()


# Get unique clusters
unique_clusters = sorted(df_filtered['cluster'].unique())


# Plot high-risk areas and color by cluster
for cluster_id in unique_clusters:
    cluster_data = df_filtered[df_filtered['cluster'] == cluster_id].copy()
    lon = cluster_data['Longitude']
    lat = cluster_data['Latitude']
    fig.add_trace(go.Scattermapbox(
        lon=lon,
        lat=lat,
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=12,
            opacity=0.5
        ),
        name=f'Cluster {cluster_id}'
    ))

# Cluster centers
lat = clusters_df['Latitude']
lon = clusters_df['Longitude']
sizes = np.asarray(clusters_df['size'])

# Plot the cluster centers with size proportional to cluster size
fig.add_trace(go.Scattermapbox(
    lon=lon,
    lat=lat,
    mode='markers',
    marker=go.scattermapbox.Marker(
        size=sizes*2,
        opacity=1,
        color='red'
    ),
    name='New AED'
))

# Update the layout for map display
fig.update_layout(
    mapbox=dict(
        style="carto-positron",
        zoom=7,
        center=dict(lat=mean_lat, lon=mean_lon)
    ),
    title="High-Risk Areas Clusters and Proposed AED Locations",
    margin={"r": 50, "t": 50, "l": 50, "b": 50},
    legend=dict(x=1, y=1)
)

# Show the figure
fig.show()