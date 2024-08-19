import dash
from dash import html
from dash import dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import geopandas as gpd
import pingouin as pg
import plotly.express as px

from Plotly_RTA import make_plots


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
# APP
# ----------------------------------------------------------------------------------------------------------------------

app = dash.Dash(__name__, title='Modern Data Analytics (G0Z39a)', external_stylesheets=[dbc.themes.BOOTSTRAP],
                serve_locally = True)


figure = make_plots(df=interventions_data, x='vector_type', y='response_time', hue='vector_type')

# Comparison Options
dropdown = dcc.Dropdown(
    id='groups',
    options=[{"label":'Province','value':'Province'},
             {"label":'Event Level','value':'Eventlevel'},
             {"label":'Vector Type','value':'vector_type'},
             ],
    value='vector_type')

# Sliders for AED Optimization
slider_rt = html.Div(
    [
        html.Label("Select Response Time Threshold", htmlFor="slider1"),
        dcc.Slider(min=0, max=30, step=1,
                       value=8, marks=None,
                       tooltip={"placement": "bottom", "always_visible": True},
                       id='slider1')
    ]
)
slider_int_dens = html.Div(
    [
        html.Label("Select Incident Density Threshold", htmlFor="slider2"),
        dcc.Slider(min=0, max=30, step=1,
                       value=5, marks=None,
                       tooltip={"placement": "bottom", "always_visible": True},
                       id='slider2')
    ]
)
slider_aed_dens = html.Div(
    [
        html.Label("Select AED Desnity Threshold", htmlFor="slider3"),
        dcc.Slider(min=0, max=30, step=1,
                       value=5, marks=None,
                       tooltip={"placement": "bottom", "always_visible": True},
                       id='slider3')
    ]
)
slider_eps = html.Div(
    [
        html.Label("Select Epsilon", htmlFor="slider4"),
        dcc.Slider(min=0, max=1, step=0.001,
                       value=0.023, marks=None,
                       tooltip={"placement": "bottom", "always_visible": True},
                       id='slider4')
    ]
)

input_groups = dbc.Row(dbc.Col(
    html.Div([
    slider_rt,
    slider_int_dens,
    slider_aed_dens,
    slider_eps]
)))

app.layout = dbc.Container(
    [
        html.Div(children=[html.H1(children='Response Time Analysis (RTA)'),
                           html.H2(children='Results Welch ANOVA'),
                           html.H4(children='', id='rta_title')],
                 style={'textAlign':'center', 'color':'black'}),
        html.Hr(),
        dbc.Row(
            [
                dropdown,
            ],
            align="center"),
        dbc.Row(
            [
                dcc.Graph(id='rta_graph', figure = figure),
             ],
            align="center",
        ),
        html.Hr(),
        html.Div(children=[html.H1(children='AED Optimization'),
                           html.H2(children='Results Clustering High-Risk-Areas'),
                          ],
                 style={'textAlign': 'center', 'color': 'black'}),
        html.Hr(),
        input_groups,

    ],
    fluid=True,
)

@app.callback(
    Output('rta_title','children'),
    Output('rta_graph','figure'),
    [Input('groups', 'value'),
     ]
)
def update_chart(input_value):
    fig=make_plots(df=interventions_data, x=input_value, y='response_time', hue=input_value)

    return 'Response Time vs. ' + input_value, fig


if __name__ == '__main__':
    app.run_server(debug=True)