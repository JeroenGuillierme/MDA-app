import pandas as pd
import numpy as np
import pingouin as pg
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objs as go

# ----------------------------------------------------------------------------------------------------------------------
# FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------


def load_data():
    """
    Load the dataset containing intervention data from a CSV file from online repository.

    :return: DataFrame containing the intervention data, or None if the data could not be loaded.
    """
    try:
        url2 = 'https://raw.githubusercontent.com/JeroenGuillierme/Project-MDA/main/Data/'

        interventions_data = pd.read_csv(
            f'{url2}rta_df.csv')
    except:
        interventions_data = None

    return interventions_data


def prepare_data(df, x, y):
    """
    Prepare the data for analysis and plotting by calculating the response time,
    transforming variables, and handling missing values.

    :param df: DataFrame containing the raw data.
    :param x: Column name to be used for the x-axis in the plots.
    :param y: Column name to be used for the y-axis in the plots.
    :return: DataFrame prepared for plotting, with necessary transformations and filtering applied.
    """
    try:
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

    except:
        df = None

    return rta_df


def make_rta_plots(x, y, hue=None):
    """
    Generate plots for response time analysis, including a point plot with confidence intervals
    and a heatmap of pairwise p-values from a Games-Howell test.

    :param x: Column name to be used for the x-axis in the plots.
    :param y: Column name to be used for the y-axis in the plots.
    :param hue: Optional; column name to be used for color encoding in the scatter plot.
    :return: Plotly figure object containing the generated plots.
    """
    # Get data
    df = load_data()
    rta_df = prepare_data(df, x=x, y=y)

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
