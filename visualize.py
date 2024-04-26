import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import numpy as np
from skimage.io import imread
import os
import pandas as pd

app = dash.Dash(__name__)

gt_path = "./data/seg/"
pd_segs = os.listdir("./results/pred_segs/")
results = pd.read_csv("./results/results.csv")
results = results[results["name"].isin(gt_segs)]

app.layout = html.Div([

    html.Div([
        html.H2('File selector:'),
        dcc.Dropdown(
            id='dropdown',
            options=[{'label': i, 'value': i} for i in gt_segs],
            value=gt_segs[0]
        ),
        html.H2('Slice Number:'),
        dcc.Slider(
            id='slice-slider',
            min=0,
            max=64,  # Vous devez définir cette valeur en fonction de la taille de vos données
            value=0,
            marks={str(i): str(i) for i in range(0, 129, 4)},
            step=1
        ),
        
    ], style={'width': '50%'}),

    html.Div([
        html.Div(style={'width': '33%'}, children=[dcc.Graph(id='2d-seg1')]),
        html.Div(style={'width': '33%'}, children=[dcc.Graph(id='2d-seg2')]),
        html.Div(style={'width': '33%'}, children=[dcc.Graph(id='2d-seg3')])
    ], style={'display': 'flex'}),

    html.H2('Informations:'),

    html.Div([
        dash_table.DataTable(
            id='table',
            columns=[{"name": i, "id": i} for i in results.columns],
            data=results.to_dict('records'),
        )
    ])
])


@app.callback(
    [Output('2d-seg1', 'figure'), Output('2d-seg2', 'figure'), Output('2d-seg3', 'figure'),
     Output('table', 'style_data_conditional')],
    [Input('dropdown', 'value'), Input('slice-slider', 'value')])
def update_graph(value, slice_num):
    gt_seg = imread(f"./results/gt_segs/{value}")
    pd_seg = imread(f"./results/pred_segs/{value}")

    gt_slice = gt_seg[:, :, slice_num]
    pd_slice = pd_seg[:, :, slice_num]

    fig1 = go.Figure()
    fig1.add_trace(go.Heatmap(z=gt_slice, name='Ground Truth', 
                             colorscale='Viridis'))
    fig1.update_layout(autosize=False, width=500, height=500, title='Ground truth')

    fig2 = go.Figure()
    fig2.add_trace(go.Heatmap(z=pd_slice, name='Prediction', 
                             colorscale='Viridis'))
    fig2.update_layout(autosize=False, width=500, height=500, title='Prediction')

    fig3 = go.Figure()
    gt_seg = np.where(gt_slice > 0, 1, 0)
    pd_seg = np.where(pd_slice > 0, 1, 0)
    diff = np.abs(pd_seg - gt_seg)
    fig3.add_trace(go.Heatmap(z=diff, name='Difference', 
                             colorscale='Viridis'))
    fig3.update_layout(autosize=False, width=500, height=500, title='Difference')

    table = [{'if': {'filter_query': '{name} eq "%s"' % value}, 'backgroundColor': '#3D9970'}]

    return fig1, fig2, fig3, table


if __name__ == '__main__':
    app.run_server(debug=True)