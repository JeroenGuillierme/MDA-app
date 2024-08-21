import dash
from dash import html
from dash import dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

from graphs_RTA import make_rta_plots
from graphs_AED import make_aed_plots

# ----------------------------------------------------------------------------------------------------------------------
# APP
# ----------------------------------------------------------------------------------------------------------------------

app = dash.Dash(__name__, title='Modern Data Analytics (G0Z39a)', external_stylesheets=[dbc.themes.BOOTSTRAP],
                serve_locally=True)

rta_figure = make_rta_plots(x='vector_type', y='response_time', hue='vector_type')
aed_figure = make_aed_plots(response_time_threshold=8, incident_density_threshold=5, aed_density_threshold=5,
                            dbscan_min_samples=4, dbscan_eps=0.023)

# RTA Comparison Options
dropdown = dcc.Dropdown(
    id='groups',
    options=[{"label": 'Vector Type', 'value': 'vector_type'},
             {"label": 'Province', 'value': 'Province'},
             {"label": 'Event Level', 'value': 'Eventlevel'},
             ],
    value='vector_type')

# Sliders for AED Optimization
slider_rt = html.Div(
    [
        html.Label(children="Select Response Time Threshold", htmlFor="slider1",
                   style={"font-weight": "bold"}),
        dcc.Slider(min=0, max=30, step=1,
                   value=8, marks=None,
                   tooltip={"placement": "bottom", "always_visible": True},
                   id='slider1')
    ]
)
slider_int_dens = html.Div(
    [
        html.Label(children="Select Incident Density Threshold", htmlFor="slider2",
                   style={"font-weight": "bold"}),
        dcc.Slider(min=0, max=30, step=1,
                   value=5, marks=None,
                   tooltip={"placement": "bottom", "always_visible": True},
                   id='slider2')
    ]
)
slider_aed_dens = html.Div(
    [
        html.Label(children="Select AED Desnity Threshold", htmlFor="slider3",
                   style={"font-weight": "bold"}),
        dcc.Slider(min=0, max=30, step=1,
                   value=5, marks=None,
                   tooltip={"placement": "bottom", "always_visible": True},
                   id='slider3')
    ]
)
slider_eps = html.Div(
    [
        html.Label(children="Select Epsilon", htmlFor="slider4",
                   style={"font-weight": "bold"}),
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
                 style={'textAlign': 'center', 'color': 'black'}),
        html.Hr(),
        dbc.Row(
            [
                dropdown,
            ],
            align="center"),
        dbc.Row(
            [
                dcc.Graph(id='rta_graph', figure=rta_figure),
            ],
            align="center",
        ),
        html.Hr(),
        html.Div(children=[html.H1(children='AED Optimization'),
                           html.H2(children='Results Clustering High-Risk-Areas'),
                           html.H4(children='', id='aed_title')
                           ],
                 style={'textAlign': 'center', 'color': 'black'}),
        html.Hr(),
        dbc.Row(
            [
                input_groups,
            ],
            align='center',
        ),
        dbc.Row(
            [
                dcc.Graph(id='aed_graph', figure=aed_figure),
            ],
            align='center',
        )

    ],
    fluid=True,
)


@app.callback(
    Output('rta_title', 'children'),
    Output('rta_graph', 'figure'),
    [Input('groups', 'value'),
     ]
)
def update_chart(group):
    fig = make_rta_plots(x=group, y='response_time', hue=group)
    if group == 'vector_type':
        group_name = 'Vector Type'
    elif group == 'Eventlevel':
        group_name = 'Event Level'
    else:
        group_name = 'Province'

    return 'Response Time vs. ' + group_name, fig


@app.callback(
    Output('aed_title', 'children'),
    Output('aed_graph', 'figure'),
    [Input('slider1', 'value'),
     Input('slider2', 'value'),
     Input('slider3', 'value'),
     Input('slider4', 'value')
     ]
)
def update_chart(rt_threshold, int_dens_threshold, aed_dens_threshold, eps):
    fig = make_aed_plots(response_time_threshold=rt_threshold, incident_density_threshold=int_dens_threshold,
                         aed_density_threshold=aed_dens_threshold, dbscan_eps=eps)

    hyper_params = [
        f'Response Time Threshold: {rt_threshold}',
        f'Intervention Density Threshold: {int_dens_threshold}',
        f'AED Density Threshold: {aed_dens_threshold}',
        f'Optimal epsilon for DBSCAN: {eps}',
    ]

    return html.Div(children=[
        html.P('Hyperparameters chosen for Determining High-Risk-Area Intervention locations'
               ' and DBSCAN clustering analysis:'),
        html.Ul([html.Li(param) for param in hyper_params])
    ], style={'textAlign': 'left'}), fig


if __name__ == '__main__':
    app.run_server(debug=True)
