import pandas as pd
import numpy as np
import geopandas as gpd
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from scipy.spatial import ConvexHull


# ----------------------------------------------------------------------------------------------------------------------
# FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------


def load_data():
    """
    Loads the required datasets for the analysis.

    :return:
        - gdf_interventions_with_both_count (pd.DataFrame): A dataframe containing intervention data with incident counts and AED counts.
        - belgium_with_provinces_boundary (gpd.GeoDataFrame): A GeoDataFrame containing the geographical boundaries of Belgian provinces.
        - aed_data (pd.DataFrame): A dataframe containing data on AED (Automated External Defibrillator) locations.

    """

    try:
        url2 = 'https://raw.githubusercontent.com/JeroenGuillierme/Project-MDA/main/Data/'

        gdf_interventions_with_both_count = pd.read_csv(
            f'{url2}gdf_interventions.csv')
        # Load Belgium with regions shapefile
        belgium_with_provinces_boundary = gpd.read_file(f'{url2}BELGIUM_-_Provinces.geojson')
        
        df = pd.read_csv(f'{url2}total_df_with_distances.csv')
        aed_data = df[df['AED'] == 1].reset_index(drop=True)

    except:
        gdf_interventions_with_both_count = None
        belgium_with_provinces_boundary = None

    return gdf_interventions_with_both_count, belgium_with_provinces_boundary, aed_data


def high_risk_areas(response_time_threshold=8, incident_density_threshold=5, aed_density_threshold=5):
    """
    Identify high-risk areas based on thresholds.

    This function filters the intervention data to identify high-risk areas where the response time is above the
    threshold, the incident density is above the threshold, and the AED density is below the threshold.

    :param response_time_threshold: (int) The response time threshold in minutes. Default is 8 minutes.
    :param incident_density_threshold: (int) The minimum incident density to qualify as a high-risk area. Default is 5.
    :param aed_density_threshold: (int) The maximum AED density for a location to be considered high-risk. Default is 5.
    :return: pd.DataFrame containing the latitude and longitude of identified high-risk areas.
    """
    gdf = load_data()[0]
    hra = gdf.loc[
          (gdf['T3-T0'] > response_time_threshold) &
          (gdf['incident_count'] > incident_density_threshold) &
          (gdf['aed_count'] < aed_density_threshold), :]
    hra.reset_index(drop=True, inplace=True)

    # Drop duplicates because same locations multiple times in dataset for representing response time of different vector types
    df = hra[['Latitude', 'Longitude']].drop_duplicates()
    df.reset_index(drop=True, inplace=True)

    return df


def dbscan(df, min_samples=4, eps=0.023):
    """
    Apply DBSCAN clustering to the high-risk areas.

    This function applies the DBSCAN clustering algorithm to the identified high-risk areas to find clusters and
    their centers. It also calculates the silhouette coefficient to evaluate clustering performance.

    :param df: pd.DataFrame containing the latitude and longitude of high-risk areas.
    :param min_samples: (int) The minimum number of samples required to form a cluster. Default is 4.
    :param eps: (float) The maximum distance between two samples for them to be considered as in the
                same neighborhood. Default is 0.023.
    :return:
        - df_filtered (pd.DataFrame): Dataframe with clustered high-risk areas.
        - df_noise (pd.DataFrame): Dataframe with noise points that were not assigned to any cluster.
        - clusters_df (pd.DataFrame): Dataframe with cluster centers and their sizes.
        - silhouette_coefficient (float): The silhouette score of the clustering.
    """

    # Initialize DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples)

    # Perform clustering
    df.loc[:, 'cluster'] = clustering.fit_predict(df)

    # Drop noise points (cluster == -1)
    # Filter out noise points (cluster == -1)
    df_filtered = df[df['cluster'] != -1]
    df_noise = df[df['cluster'] == -1]

    cluster_centers = df_filtered.groupby('cluster')[['Latitude', 'Longitude']].mean().reset_index()

    # Determine cluster sizes
    cluster_sizes = df_filtered['cluster'].value_counts().reset_index()
    cluster_sizes.columns = ['cluster', 'size']
    labels = clustering.labels_

    # Combine locations of clusters centers and their sizes
    clusters_df = pd.merge(cluster_centers, cluster_sizes, on='cluster')

    # Evaluation metrics

    silhouette_coefficient = silhouette_score(df[['Latitude', 'Longitude']], labels)

    return df_filtered, df_noise, clusters_df, silhouette_coefficient


def geom_belgium():
    """
    Calculates the geographic centroid of Belgium based on its provinces' boundaries.

    :return:
        - mean_lat (float): The latitude of the centroid.
        - mean_lon (float): The longitude of the centroid.
    """
    belgium_with_provinces_boundary = load_data()[1]
    # Re-project to a suitable projected CRS (e.g., UTM zone 31N for Belgium)
    belgium_proj = belgium_with_provinces_boundary.to_crs(epsg=32631)

    # Calculate the centroid of the re-projected geometry
    centroid = belgium_proj.geometry.centroid

    # Now, we convert this projected centroid back to geographic CRS for map display purposes
    centroid_geo = centroid.to_crs(epsg=4326)
    mean_lat = centroid_geo.y.mean()
    mean_lon = centroid_geo.x.mean()
    return mean_lat, mean_lon


def find_neighbours(df, n_neighbours):
    """
    Find the k-nearest neighbors of each point in the dataframe.

    This function calculates the distances to the k-nearest neighbors for each point in the dataframe, sorting them
    in ascending order.

    :param df: pd.DataFrame containing the latitude and longitude of the points.
    :param n_neighbours: (int) The number of nearest neighbors to find for each point.
    :return: np.ndarray containing the sorted distances to the 4th nearest neighbor for each point.
    """
    neighbors = NearestNeighbors(n_neighbors=n_neighbours)
    neighbors_fit = neighbors.fit(df)

    # Find the k-neighbors of a point
    distances, indices = neighbors_fit.kneighbors(df)

    # Sort the neighbor distances (lengths to points) in ascending order
    distances_sorted = np.sort(distances, axis=0)  # axis=0 represents sort along the first axis i.e. sort along rows

    # Get the 4th nearest neighbor distances
    k_dist = distances_sorted[:, 4]

    return k_dist


def make_aed_plots(response_time_threshold=8, incident_density_threshold=5, aed_density_threshold=5,
                   dbscan_min_samples=4, dbscan_eps=0.023):
    """
    Create visualizations for AED analysis.

    This function generates a set of visualizations, including a k-NN distance plot, a bar plot of cluster sizes, a
    map of high-risk areas with cluster centers, and an evaluation metric plot (silhouette score).

    :param response_time_threshold: (int, optional) The response time threshold in minutes. Default is 8 minutes.
    :param incident_density_threshold: (int, optional) The minimum incident density to qualify as a high-risk area. Default is 5.
    :param aed_density_threshold: (int, optional) The maximum AED density for a location to be considered high-risk. Default is 5.
    :param dbscan_min_samples: (int, optional) The minimum number of samples required to form a cluster using DBSCAN. Default is 4.
    :param dbscan_eps: (float, optional) The maximum distance between two samples for them to be considered as in the
                       same neighborhood using DBSCAN. Default is 0.023.
    :return: Plotly figure object containing the visualizations.
    """

    df = high_risk_areas(response_time_threshold=response_time_threshold,
                         incident_density_threshold=incident_density_threshold,
                         aed_density_threshold=aed_density_threshold)

    df_filtered, df_noise, clusters_df, silhouette_coefficient = dbscan(df, min_samples=dbscan_min_samples,
                                                                        eps=dbscan_eps)

    mean_lat, mean_lon = geom_belgium()
    # Create the figure
    fig = go.Figure()
    # Create subplots with 1 row and 2 columns
    fig = make_subplots(rows=3, cols=2, subplot_titles=('k-NN Distance plot',
                                                        'Bar Plot Showing Cluster Sizes',
                                                        'High-Risk-Area Clusters & Proposed AED Locations',
                                                        'Evaluation Metric: Silhouette Score'
                                                        ),
                        specs=[[{}, {}], [{"colspan": 2, "type": 'mapbox'}, None], [{"colspan": 2}, None]],
                        vertical_spacing=0.1,  # Adjust vertical spacing between rows
                        horizontal_spacing=0.1,  # Adjust horizontal spacing between columns
                        row_heights=[0.4, 0.5, 0.1]  # Adjust row heights to allocate more space for the map
                        )

    # First Plot: k-NN Distances Plot
    # --------------------------------

    k_dist = find_neighbours(df, n_neighbours=dbscan_min_samples + 1)

    # Add the k-NN distance plot
    knn_lineplot = go.Scatter(
        y=k_dist,
        mode='lines',
        name='k-NN distance'
    )
    fig.add_trace(knn_lineplot, row=1, col=1)

    # Add the horizontal line
    fig.add_shape(type="line",
                  x0=0, y0=dbscan_eps, x1=len(k_dist) - 1, y1=dbscan_eps,
                  line=dict(color="Red", width=2, dash="dash"),
                  name='Threshold',
                  row=1, col=1)

    # Second Plot: Bar plot of Cluster Sizes
    # --------------------------------------

    # Get unique clusters
    unique_clusters = sorted(df_filtered['cluster'].unique())

    np.random.seed(32)

    # Generate a color mapping for each cluster
    color_map = {
        cluster_id: f'rgb({np.random.randint(0, 255)}, {np.random.randint(0, 255)}, {np.random.randint(0, 255)})'
        for cluster_id in unique_clusters
    }

    cluster_nr = clusters_df['cluster'].astype('str')
    cluster_size = clusters_df['size']

    barplot = go.Bar(
        x=cluster_nr,
        y=cluster_size,
        name='Bar Plot Showing Cluster Sizes',
        marker=dict(color=[color_map[cluster_id] for cluster_id in clusters_df['cluster']]),
        hovertext=[f'Cluster: {cluster_nr[i]} <br>Cluster Size: {cluster_size[i]}'
                   for i in range(len(clusters_df))],
        hoverinfo='text'
    )
    fig.add_trace(barplot, row=1, col=2)

    # Third Plot: Map of High Risk Areas and Cluster Centers (and AEDs)
    # ------------------------------------------------------------------

    # Add already existing AED locations to map
    aed_locations = load_data()[2]

    lat = aed_locations['Latitude']
    lon = aed_locations['Longitude']

    aeds = go.Scattermapbox(
        lon=lon,
        lat=lat,
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=4,
            color='grey',
            opacity=0.6
        ),
        name='AED Location',
        hovertext=[f'Current AED Location: ({lat[i]}°, {lon[i]}°)' for i in range(len(aed_locations))],
        hoverinfo='text'
    )
    fig.add_trace(aeds, row=2, col=1)

    # Add Cluster centers
    lat = clusters_df['Latitude']
    lon = clusters_df['Longitude']
    sizes = np.asarray(clusters_df['size'])

    # Plot the cluster centers in red
    scatter_centers = go.Scattermapbox(
        lon=lon,
        lat=lat,
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=12,
            color='red'
        ),
        name='New AED',
        hovertext=[f'Centroid of Cluster {i} as proposed location for new AED' for i in range(len(clusters_df))],
        hoverinfo='text'
    )
    fig.add_trace(scatter_centers, row=2, col=1)

    # Plot high-risk areas and color by cluster
    for cluster_id in unique_clusters:
        cluster_data = df_filtered[df_filtered['cluster'] == cluster_id].copy()
        lon = cluster_data['Longitude'].reset_index(drop=True)
        lat = cluster_data['Latitude'].reset_index(drop=True)

        cluster_color = color_map[cluster_id]  # Get the color for the current cluster

        # Plot cluster points
        scatter = go.Scattermapbox(
            lon=lon,
            lat=lat,
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=12,
                opacity=1,
                color=cluster_color
            ),
            name=f'Cluster {cluster_id}',
            hovertext=[f'Incident Location: ({lat[i]}°, {lon[i]}°) '
                       f'<br>Inside High-Risk-Area Cluster {cluster_id}' for i in range(len(cluster_data))],
            hoverinfo='text'
        )
        fig.add_trace(scatter, row=2, col=1)
        # Calculate Convex Hull for each cluster to encircle the clusters on the map
        if len(cluster_data) > 2:  # ConvexHull needs at least 3 points
            points = np.column_stack((lon, lat))
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]

            # Append the first point to the end to close the polygon
            hull_points = np.vstack([hull_points, hull_points[0]])
            # Create a polygon
            lon_hull_points = hull_points[:, 0]
            lat_hull_points = hull_points[:, 1]
            polygon = go.Scattermapbox(
                lon=lon_hull_points,
                lat=lat_hull_points,
                mode='lines',
                fill='toself',
                fillcolor=f'rgba{cluster_color[3:-1]}, 0.3)',  # Match the color and adjust transparency
                line=dict(color=cluster_color, width=2),
                name=f'Cluster {cluster_id}',
                hovertext=[f'Incident Location: ({lat_hull_points[i]}°, {lon_hull_points[i]}°) '
                           f'<br>In High-Risk-Area Cluster {cluster_id}' for i in range(len(hull_points))],
                hoverinfo='text'
            )
            fig.add_trace(polygon, row=2, col=1)

    # Add Noise Points to Map
    lat = df_noise['Latitude']
    lon = df_noise['Longitude']

    # Plot the noise points in black
    noise_points = go.Scattermapbox(
        lon=lon,
        lat=lat,
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=6,
            color='black'
        ),
        name='Noise Point',
        hovertext=f'One of the {len(df_noise)} noise points.'
                  f'<br> These points are not assigned to any cluster.',
        hoverinfo='text'
    )
    fig.add_trace(noise_points, row=2, col=1)


    # Fourth Plot: Silhouette score
    # -------------------------------
    silhouette_line = go.Scatter(
        x=[0, silhouette_coefficient, 1], y=[0, 0, 0], mode='markers+text',
        marker=dict(
            size=[10, 20, 10],
            color=['rgb(0, 0, 0)', 'rgb(255, 0, 0)', 'rgb(0, 0, 0)']
        ),
        text=['Worst Case', "Silhouette Coefficient:%0.2f" % silhouette_coefficient, 'Ideal Case'],
        textposition='top center'

    )
    fig.add_trace(silhouette_line, row=3, col=1)

    # Update the layout for map display
    fig.update_layout(
        autosize=True,
        hovermode='closest',
        mapbox=dict(
            style="carto-positron",
            zoom=7,
            center=dict(lat=mean_lat, lon=mean_lon)
        ),
        title="AED Analysis Plots",
        height=1000,
        margin=dict(l=20, r=20, t=40, b=20),  # Adjust the figure's margins
        showlegend=False
    )

    fig.update_xaxes(title_text="Sorted Observations (4th NN)", row=1, col=1)
    fig.update_yaxes(title_text="k-NN Distance", row=1, col=1)

    fig.update_xaxes(title_text="Cluster Number", row=1, col=2)
    fig.update_yaxes(title_text="Cluster Size", row=1, col=2)

    fig.update_xaxes(showgrid=False, row=3, col=1,
                     showticklabels=True,
                     fixedrange=True)
    fig.update_yaxes(showgrid=False, row=3, col=1,
                     zeroline=True, zerolinecolor='black', zerolinewidth=4,
                     showticklabels=False)

    return fig

