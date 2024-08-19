import pandas as pd
import numpy as np
import geopandas as gpd
import pingouin as pg
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objs as go

# ----------------------------------------------------------------------------------------------------------------------
# IMPORTING DATA
# ----------------------------------------------------------------------------------------------------------------------

url2 = 'https://raw.githubusercontent.com/JeroenGuillierme/Project-MDA/main/Data/'

interventions_data = pd.read_csv(
    f'{url2}rta_df.csv')
aed_data =pd.read_csv(f'{url2}total_df_with_distances.csv')

# Load Belgium shapefile
belgium_boundary = gpd.read_file(f'{url2}Belgi%C3%AB.json')
# Load Belgium with regions shapefile
belgium_with_provinces_boundary = gpd.read_file(f'{url2}BELGIUM_-_Provinces.geojson')


# ----------------------------------------------------------------------------------------------------------------------
# FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------
def load_data():
    try:
        url2 = 'https://raw.githubusercontent.com/JeroenGuillierme/Project-MDA/main/Data/'

        interventions_data = pd.read_csv(
            f'{url2}rta_df.csv')
    except:
        df=None

    return interventions_data


def make_plots(df, x, y, hue=None):
    # Get data
    df = load_data()

    # Prepare data for graphs
    df['response_time'] = df['T3-T0']
    df['vector_type'] = df['Vector type']

    # Make Event level string
    df['Eventlevel'] = df['Eventlevel'].astype('str')

    if x != 'vector_type':
        rta_df = df.sort_values(by=['response_time', 'vector_type'], ascending=[True, True]).drop_duplicates(
            subset='Mission ID', keep='first')
        rta_df = rta_df[[y, x]].dropna()

    else:
        rta_df = df[[y, x]].dropna()

    # Log-Transform the variable Response Time, because is right-skewed
    rta_df['log_response_time'] = rta_df['response_time'].transform(np.log10)

    # Reset indices
    rta_df.reset_index(drop=True, inplace=True)

    # Create subplots with 1 row and 2 columns
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f'Pointplot of Response Time vs {x}',
                                                        'Pairwise p-values (Games-Howell)'))

    # Calculate the mean and standard error of the mean (SEM) for each group
    grouped = rta_df.groupby([x])[y].agg(
        mean_y='mean',
        sem_y=lambda x: 1.96*x.std() / (len(x) ** 0.5)
    ).reset_index()

    # Pointplot
    scatter_fig = px.scatter(
        grouped,
        x=x,
        y='mean_y',
        color=hue,
        error_y='sem_y',  # Standard error for confidence intervals for normal distribution
        labels={'mean_y': 'Average Response Time'},
    )

    for trace in scatter_fig.data:
        fig.add_trace(trace, row=1, col=1)

    fig.update_layout(showlegend=False)


    # Post hoc Games Howell test
    gh = pg.pairwise_gameshowell(dv='log_response_time', between=x, data=rta_df)

    # Pivot the results to get the p-values
    pval_matrix = gh.pivot(index='A', columns='B', values='pval').round(3)

    # Heatmap
    heatmap = go.Heatmap(
        z=pval_matrix.values,
        x=pval_matrix.columns,
        y=pval_matrix.index,
        colorscale='RdBu_r',
        colorbar=dict(title='p-value'),
        zmin=0, zmax=1,  # p-values range from 0 to 1
        text=pval_matrix.values,
        hoverinfo='text'
    )
    fig.add_trace(heatmap, row=1, col=2)

    # Update layout to adjust titles, size, etc.
    fig.update_layout(
        title_text="Response Time Analysis Plots",
        xaxis_tickangle=-90,
        height=600,
        width=1500,
    )

    return fig