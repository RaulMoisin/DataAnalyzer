import os
import numpy
from dash import Dash, dcc, html, dash_table, Output, Input, State, MATCH, ALL
import pandas as pd
import numpy as np
#from dash.dependencies import Output, Input, State, MATCH, ALL

import plotly.graph_objects as go
from dash.exceptions import PreventUpdate

import plotly.express as px
import csv
import base64
import datetime
import io
from io import BytesIO
import plotly.graph_objs as go
import cufflinks as cf

import matplotlib
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller

import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from math import sqrt
import itertools
import flask


from helpers import *
from models import *

import re
import time

# anomalies

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from numpy import sqrt, argsort
from sklearn.preprocessing import scale
from sklearn.metrics import silhouette_score
import pmdarima as pmd

from pyspark.sql import SparkSession
import dash_uploader as du



data = None

DEFAULT_OPACITY = 0.8

def configurations():

    spark = SparkSession.builder.config("spark.ui.enabled", "false").getOrCreate()

    external_stylesheets = [
        {
            "href": "https://fonts.googleapis.com/css2?"
            "family=Lato:wght@400;700&display=swap",
            "rel": "stylesheet",
        },
    ]
    current_directory = os.getcwd()
    app = Dash(__name__, external_stylesheets=external_stylesheets,)
    du.configure_upload(app, f"{current_directory}/data")
    server = app.server
    app.config['suppress_callback_exceptions'] = True
    app.title = "Addable"

    app.layout = html.Div(
        children=[
            html.Div(
            id="header",
            children=[
                html.H4(children="Time-Series Data Analyzer"),
                html.P(
                    id="description",
                    children=" It is advisable that you upload your file first.",
                ),
                html.Div(
                        children=[
                            dbc.Button('Add a configuration block',outline=True, id='submit-val',n_clicks=0,size="lg")
                        ],
                    ),

                html.Div([
                            du.Upload(
                                id='upload-data',
                                text='Drag and Drop files here',
                                text_completed='Completed: ',
                                pause_button=False,
                                cancel_button=True,
                                max_file_size=1800,  # 1800 Mb
                                filetypes=['csv', 'rar'],
                                # children=html.Div([
                                # html.A('Upload your csv file')
                                #                  ]),

                                # multiple=True
                            )
                        ]
                        ),
            ],
        ),

            html.Div( id = "addable", children = [
                ],
            ),

            html.Div(id='output-data-upload')
        ]
    )

    @app.callback(Output({'type' : 'decompose-output', 'index': MATCH}, 'data'),
        [
            Input({"type" : "Decompose-Radio", "index" : MATCH}, 'value'),
            State({'type' : 'decompose-output', 'index': MATCH}, 'data')

        ]#, prevent_initial_call=True)
    )
    def decompose_params(value,decompose_data):

        decompose_data = {'type':'additive'}

        if value:
            decompose_data['type']= value

        return decompose_data

    @app.callback(Output({'type' : 'correlation-output', 'index': MATCH}, 'data'),
        [
            Input({"type" : "Correlation-Radio", "index" : MATCH}, 'value'),
            Input({"type" : "lag-option", "index" : MATCH}, 'value'),
            State({'type' : 'correlation-output', 'index': MATCH}, 'data')
        ]#,prevent_initial_call=True
    )
    def correlation_params(option,value,correlation_data):

        correlation_data = {'ype' : 'auto', 'value' : 25}

        if option:
            correlation_data['type'] = option
        if value:
            correlation_data['value'] = value

        return correlation_data

    for store in ('arima','sarima'):

        @app.callback(Output({'type' : 'model-' + store + '-output', 'index': MATCH}, 'data'),
            [
                Input({"type" : "p-" + store + "-option", "index" : MATCH}, 'value'),
                Input({"type" : "d-" + store + "-option", "index" : MATCH}, 'value'),
                Input({"type" : "q-" + store + "-option", "index" : MATCH}, 'value'),
                Input({"type" : "P-" + store + "-option", "index" : MATCH}, 'value'),
                Input({"type" : "D-" + store + "-option", "index" : MATCH}, 'value'),
                Input({"type" : "Q-" + store + "-option", "index" : MATCH}, 'value'),
                Input({"type" : "m-" + store + "-option", "index" : MATCH}, 'value'),
            ],prevent_initial_call=True)
        def models_params(p_value,d_value,q_value,P_value,D_value,Q_value,m_value):

            ctx = dash.callback_context
            if not ctx.triggered:
                field_id = 'No params yet'
            else:
                field_id = ctx.triggered[0]['prop_id'].split('.')[0]

            if store == 'arima':
                return {'p-value':p_value, 'd-value':d_value,'q-value':q_value}
            if store == 'sarima':
                return {'p-value':p_value, 'd-value':d_value,'q-value':q_value,'P-value':P_value, 'D-value':D_value,"Q-value":Q_value,"m-value":m_value}
            return {}

    @app.callback(Output({'type' : 'store-output', "index" : MATCH}, 'data'), 
        [
            Input({'type' : 'decompose-output', 'index': MATCH}, 'data'),
            Input({'type' : 'correlation-output', 'index': MATCH}, 'data'),
            Input({'type' : 'model-arima-output', 'index': MATCH}, 'data'),
            Input({'type' : 'model-sarima-output', 'index': MATCH}, 'data'),
            State({'type' : 'store-output', "index" : MATCH}, 'data')
        ],prevent_initial_call=True
    )
    def combine_store_parameters(decompose, correlation, arima, sarima, output):

        ctx = dash.callback_context
        if not ctx.triggered:
            field_id = 'No params yet'
        else:

            field_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if "decompose-output" in field_id :
            output['decomposition'] = decompose

        if "correlation-output" in field_id :
            output['correlation'] = correlation

        if  "model-arima-output" in field_id :
            output['arima'] = arima

        if "model-sarima-output" in field_id :
            output['sarima'] = sarima

        return output

    # upload file callback
    @app.callback(Output('output-data-upload', 'children'),
            [
                #Input('upload-data', 'contents'),
                Input('upload-data', 'isCompleted'),
                State('upload-data', 'fileNames'),
            ],prevent_initial_call=True
    )
    def update_table(isCompleted, filenames):

        if not isCompleted:
            raise PreventUpdate

        table = html.Div()

        if isCompleted:
            filename = filenames[0]

            filename = 'data/'+filename
            df = pd.read_csv(filename)

            df = pd.concat([df.head(10), df.tail(10)], axis=0)

            table = html.Div([
                html.H5(filename),
                dash_table.DataTable(
                    data=df.to_dict('records'),
                    columns=[{'name': i, 'id': i} for i in df.columns],

                    style_header={'backgroundColor': 'rgb(30, 30, 30)'},
                    style_cell={
                        'backgroundColor': '#1f2630',
                        'color': '#2cfec1',
                        'textAlign': 'left'
                    },
                ),
                html.Hr(),
            ])

            print("Data uploaded")

            return table

# function for appending the configuration block to the main layout
    @app.callback(Output("addable","children"),
        (
            Input("submit-val", "n_clicks"),
            State("addable","children"),
        ), prevent_initial_call=True
    )
    def create_second(n_clicks,kids):

        print(n_clicks)

        if n_clicks == 0:
            raise PreventUpdate

        print("do we get here 1")

        addable =  kids + [ html.Div( children = [
            dbc.Row( id = "menu",
                    children=[
                        html.Div(
                            children=[
                                html.Div(children="Separator-column", className="menu-title", id= f"id-separator-column-tooltip-{n_clicks}"),
                                dcc.Dropdown(
                                    id={"type" : "Parameter-column", "index" : n_clicks},
                                    style={'height': '45px', 'width': '170px', 'display': 'inline-block', 'color': "black"},
                                    value = '',
                                    clearable=False,
                                    className="dropdown",
                                ),
                                dbc.Tooltip(
                                    "Choose the column which contains the separators",
                                    target=f"id-separator-column-tooltip-{n_clicks}",
                                ),
                            ],
                        ),

                        html.Div(
                            children=[
                                html.Div(children="Separator", className="menu-title", id= f"id-separator-tooltip-{n_clicks}"),
                                dcc.Dropdown(
                                    id={"type":"Parameter-filter", "index": n_clicks},
                                    style={'height': '45px', 'width': '170px', 'display': 'inline-block', 'color': 'black'},
                                    value = '',
                                    clearable=False,
                                    className="dropdown",
                                ),
                                dbc.Tooltip(
                                    "Separators are contextual attributes which can posses the same index values(time values)",
                                    target=f"id-separator-tooltip-{n_clicks}",
                                ),
                            ],
                        ),

                        html.Div(
                            children=[
                                html.Div(children="Comparable-Separator", className="menu-title", id=f"id-comparable-separator-tooltip-{n_clicks}"),
                                dcc.Dropdown(
                                    id={"type":"Extra-Parameter-filter", "index": n_clicks},
                                    style={'height': '45px', 'width': '170px', 'display': 'inline-block', 'color': 'black'},
                                    value = '',
                                    clearable=False,
                                    className="dropdown",
                                ),
                                dbc.Tooltip(
                                    "Choose a separator if you wish to compare it with the previous one",
                                    target=f"id-comparable-separator-tooltip-{n_clicks}",
                                ),
                            ],
                        ),

                        html.Div(
                            children=[
                                html.Div(
                                    children="Attribute", className="menu-title", id= f"id-attribute-tooltip-{n_clicks}"),
                                dcc.Dropdown(
                                    id={"type":"YAxis-filter", "index":n_clicks},
                                    style={'height': '45px', 'width': '170px', 'display': 'inline-block', 'color': 'black'},
                                    value = '',
                                    clearable=False,
                                    className="dropdown",
                                ),
                                dbc.Tooltip(
                                    "A behavioral attribute represents the other axis on the plot",
                                    target=f"id-attribute-tooltip-{n_clicks}",
                                ),
                            ],
                            #className = 'models'
                        ),
                        html.Div( id = 'separator-button-box',
                            children=[
                            dbc.Button('Add Separator', id={"type" : "add-separator", "index" : n_clicks},n_clicks=0,size="lg",className = 'separator-title'),
                            ]
                        )
                    ],
                    align = 'center',
                ),

            html.Div(id = {"type" : "separator-block", "index" : n_clicks}, children = []),

            dbc.Row([
                    dbc.Col(
                        id = "col1",
                        children = [
                            html.Div(
                                children = [
                                    html.Div(children="Config-Checklist", className="menu-title"),
                                    dcc.Checklist(
                                        id={"type" : "Config-Checklist", "index" : n_clicks},
                                        options=[
                                            {'label': 'Mean', 'value': 'mean'},
                                            {'label': 'Rolling Mean', 'value': 'rmean'},
                                            {'label': 'Standard Deviation', 'value': 'std'}
                                        ],
                                        value = [],
                                        className="checklist",
                                    ),
                                ],
                                className="configs-container",
                            ),

                            html.Div(
                                children = [
                                    html.Div(children="Models-Checklist", className="menu-title"),
                                    dcc.Checklist(
                                        id={"type" : "Models-Checklist", "index" : n_clicks},
                                        options=[
                                            {'label': 'ARIMA', 'value': 'arima'},
                                            {'label': 'SARIMA', 'value': 'sarima'},
                                            {'label': 'Decomposition', 'value': 'decompose'},
                                            {'label': 'Correlation', 'value': 'correlation'}
                                        ],
                                        value = [],
                                        style = {'display' : 'inline-block'},
                                        className="modelList",
                                    ),
                                ],

                                className="configs-container",
                            ),

                            html.Div(
                                children = [
                                    html.Div(children="Anomaly-Detection-Checklist", className="menu-title"),
                                    dcc.Checklist(
                                        id={"type" : "Anomaly-Checklist", "index" : n_clicks},
                                        options=[
                                            {'label': 'K-Means Clustering', 'value': 'kmean'},
                                            {'label': 'Isolation Forest', 'value': 'forest'},
                                        ],
                                        value = [],
                                        style = {'display' : 'inline-block'},
                                        className="anomalyList",
                                    ),
                                ],

                                className="configs-container",
                            ),

                            html.Div(
                                children=[
                                    dbc.Button("Go Plot", id={'type' : 'do-plots', 'index' : n_clicks },className="menu-title",n_clicks=0),
                                    dbc.Button("Compare", id={'type' : 'compare', 'index' : n_clicks },className="menu-title",n_clicks=0),
                                ],
                                className = "configs-container"
                            ),

                        ],
                    ),

            ],
            ),

            dbc.Row(
                        html.Div(
                            id={"type" : "model-blocks", "index" : n_clicks},
                            children = [],
                            className = 'other_containers'
                        ),
                    ),

            html.Div(
                children=[
                    html.Div(
                        id="slider-container",
                        children=[
                            html.P(
                                id="slider-text",
                                children="Drag the slider to change the date:",
                            ),
                            dcc.RangeSlider(min=0, max=0,
                                id={"type" : "years-slider", "index" : n_clicks},
                            ),
                            dcc.Store(id = {'type' : 'slider-output', 'index' : n_clicks}, data = {}, storage_type='session'),
                        ],
                    ),
                    html.Div(id = "graph-container",
                        children=dcc.Graph(
                            id={"type" : "price-chart", "index" : n_clicks},
                            figure=dict(
                                data=[dict(x=0, y=0)],
                                layout=dict(
                                    paper_bgcolor="#1f2630",
                                    plot_bgcolor="#1f2630",
                                    autofill=True,
                                    margin=dict(t=100, r=100, b=100, l=100)),
                                )
                        ),
                    ),
                    html.Div(
                        id = {"type" : "other-charts", "index" : n_clicks},
                    ),

                ],
            ),

            dcc.Store(
                id = {'type' : 'store-output', 'index' : n_clicks},
                data = {
                    'decomposition' : {'type' : 'additive'},
                    'correlation' : {'type' : 'auto', 'value' : 25},
                    'arima' : {'p-value' : 2, 'd-value' : 1,'q-value' : 0},
                    'sarima' : {'p-value' : 0, 'd-value' : 0,'q-value' : 1, 'P-value' : 1, "D-value" : 1, 'Q-value' : 1, 'm-value' : None},
                }
            ),
            dcc.Store(id = {'type' : 'decompose-output', 'index' : n_clicks}, data = {}),
            dcc.Store(id = {'type' : 'correlation-output', 'index' : n_clicks}, data = {}),
            dcc.Store(id = {'type' : 'model-arima-output', 'index' : n_clicks}, data = {}),
            dcc.Store(id = {'type' : 'model-sarima-output', 'index' : n_clicks}, data = {}),
            dcc.Store(id = {'type' : 'separators-output', 'index' : n_clicks}, data = {}),
        ],

        className = "big-container"
    )
]
        print("do we get here 2")
        return addable

    # function for appending the parameter blocks to the configuration block
    @app.callback(Output({"type" : "model-blocks", "index" : MATCH}, "children"),
        (
            Input({"type" : "Models-Checklist", "index" : MATCH},"value"),
            State({"type" : "Models-Checklist", "index" : MATCH},"id"),
            State({"type" : "model-blocks", "index" : MATCH}, "children"),
        ),prevent_initial_call=True

    )
    def addModelConfigBlock(models,id,kids):

        if models == []:
            PreventUpdate

        for value in models:
            addable =  kids + checkModelValue(value,id.get("index"))

        return addable

    # function for generating a separator dropdown block
    @app.callback(Output({"type" : "separator-block", "index" : MATCH}, 'children'),
        [
            Input({"type" : "add-separator", "index" : MATCH}, 'n_clicks'),
            State({"type" : "separator-block", "index" : MATCH}, 'children'),
            State({"type" : "Parameter-column", "index" : MATCH}, "options"),
            State({"type" : "add-separator", "index" : MATCH}, 'id'),
        ]
    )
    def display_extra_dropdowns(n_clicks, children, separator_column_options, index_id):

        if n_clicks == 0:
            raise PreventUpdate

        children.append(dbc.Row(id = "menu",
            children=[
                        html.Div(
                            children=[
                                html.Div(children="Extra-Separator-column", className="menu-title"),
                                dcc.Dropdown(
                                    id={"type" : "Extra-Separator-column", "index" : index_id.get("index"), "second_index" : n_clicks},
                                    style={'height': '45px', 'width': '170px', 'display': 'inline-block', 'color': "black"},
                                    options = separator_column_options,
                                    value = '',
                                    clearable=False,
                                    className="dropdown",
                                ),
                            ],
                        ),

                        html.Div(
                            children=[
                                html.Div(children="Extra-Separator", className="menu-title"),
                                dcc.Dropdown(
                                    id={"type":"separator-parameter-filter", "index": index_id.get("index"), "second_index" : n_clicks},
                                    style={'height': '45px', 'width': '170px', 'display': 'inline-block', 'color': 'black'},
                                    value = '',
                                    clearable=False,
                                    className="dropdown",
                                ),
                            ],
                        ),

                        html.Div(
                            children=[
                                html.Div(children="Extra-Comparable-Separator", className="menu-title"),
                                dcc.Dropdown(
                                    id={"type":"Extra-Comparable-filter", "index": index_id.get("index"), "second_index" : n_clicks},
                                    style={'height': '45px', 'width': '170px', 'display': 'inline-block', 'color': 'black'},
                                    value = '',
                                    clearable=False,
                                    className="dropdown",
                                ),
                            ],
                        ),
                        dcc.Store(id = {'type' : 'separator-block-output', 'index' : index_id.get("index"), "second_index" : n_clicks}, data = {}),
                    ]
                ),
        )

        return children

    @app.callback(Output({"type" : "separators-output", "index" : MATCH}, "data"),
        [
            Input({"type" : "separator-block-output", "index" : MATCH, "second_index" : ALL}, "data"),
            State({"type" : "separators-output", "index" : MATCH}, "data")
        ],prevent_initial_call=True
    )
    def combine_separator_stores(small_store,big_store):

        if small_store is None:
            raise PreventUpdate
        
        for store in small_store:

            try:
                big_store[list(store.keys())[0]] = list(store.values())[0]
            except IndexError:
                pass

        return big_store

    @app.callback(Output({"type" : "separator-block-output" , "index" : MATCH, "second_index" : MATCH}, "data"),
        [
            Input({'type' : 'separator-parameter-filter', 'index' : MATCH, "second_index" : MATCH}, 'value'),
            Input({'type' : 'Extra-Separator-column', 'index' : MATCH, "second_index" : MATCH}, 'value'),
            Input({'type' : 'Extra-Comparable-filter', 'index' : MATCH, "second_index" : MATCH}, 'value'),
            State({"type" : "separator-block-output" , "index" : MATCH, "second_index" : MATCH}, "data"),
        ],prevent_initial_call= True
    )
    def add_separator_key(parameter,separator,extra_parameter,data):

        if separator is None and parameter is None:
            raise PreventUpdate


        if separator and parameter:
            data[separator] = [parameter,extra_parameter]

        return data

    @app.callback([Output({"type" : "separator-parameter-filter", "index" : MATCH, "second_index" : MATCH}, "options"), Output({"type" : "Extra-Comparable-filter", "index" : MATCH, "second_index" : MATCH}, "options")],
        (
            Input({'type' : 'Extra-Separator-column', 'index' : MATCH, "second_index" : MATCH}, 'value'),
        ),prevent_initial_call=True
    )
    def set_extra_Parameter_values(value):
        if data is None or value == "":
            raise PreventUpdate

        Parameter_options=[ {'label': i, 'value': i} for i in np.sort(getattr(data,value).unique())]
        return Parameter_options,Parameter_options

    @app.callback([Output({'type':'YAxis-filter', 'index': MATCH}, 'options'),Output({'type' : 'Parameter-column', 'index': MATCH}, 'options'),Output({"type":"years-slider","index":MATCH},'value'),Output({"type":"years-slider","index":MATCH},'min'),Output({"type":"years-slider","index":MATCH},'max'),Output({"type":"years-slider","index":MATCH},'marks'),],
            [
                Input("submit-val","n_clicks"),
                #State('upload-data', 'contents'),
                State('upload-data', 'isCompleted'),
                State('upload-data', 'fileNames'),
                State({'type':'YAxis-filter', 'index': MATCH}, "id")
            ]
    )
    def update_default_fields(n_clicks, isCompleted, filenames, id):

        if not isCompleted:
            raise PreventUpdate

        if id.get('index') != n_clicks:
            raise PreventUpdate

        if n_clicks == 0:
            raise PreventUpdate
        print(n_clicks)

        global data

        filename = filenames[0]

        filename = 'data/'+filename
        print(filename)
        with open(filename, "r") as f:
            reader = csv.reader(f)
            headers = next(reader)

        data = pd.read_csv(filename,index_col='Date', parse_dates=['Date'])
        data.sort_index(inplace=True)
        # spark_df = spark.createDataFrame(data)
        # spark_df.show()
        #data = data.fillna(method='ffill')
        attribute_options=[ {'label': i, 'value': i} for i in headers if i != '' if i !='Date' ]
        Parameter_options=[ {'label': i, 'value': i} for i in headers if i !='Date' ]

        time_list = [pd.to_datetime(element) for element in data.index.values]
        time_list = list(dict.fromkeys(time_list))
        # timestamp_list = [element.timestamp() for element in time_list]

        min_value = unixTimeMillis(time_list[0])
        max_value = unixTimeMillis(time_list[-1])

        value = [unixTimeMillis(time_list[0]),
                         unixTimeMillis(time_list[-1])]

        frequency = int(len(time_list)/12)

        marks = getMarks(time_list[0],
                            time_list[-1],frequency,time_list)


        return attribute_options, Parameter_options, value, min_value, max_value, marks

    @app.callback([
        Output({'type' : 'YAxis-filter', 'index': MATCH}, 'value'),Output({'type' : 'Parameter-column', 'index':MATCH}, 'value')],
        [
            Input({'type' : 'YAxis-filter','index':MATCH}, 'options'),
            Input({'type' : 'Parameter-column', 'index' : MATCH}, 'options'),
        ], prevent_initial_call=True
    )
    def set_dropboxes_value(attribute_options,Parameter_options):
        return attribute_options[0]['value'],Parameter_options[0]['value']

    @app.callback(Output({'type' : 'Parameter-filter', 'index' : MATCH}, 'options'),
        [
            Input({'type' : 'Parameter-column', 'index' : MATCH}, 'value'),
        ], prevent_initial_call=True
    )
    def set_Parameter_values(value):
        if data is None or value == "":
            raise PreventUpdate

        Parameter_options=[ {'label': i, 'value': i} for i in np.sort(getattr(data,value).unique())]
        return Parameter_options

    @app.callback(Output({'type' : 'Extra-Parameter-filter', 'index' : MATCH}, 'options'),
        [
            Input({'type' : 'Parameter-column', 'index' : MATCH}, 'value'),
        ], prevent_initial_call=True
    )
    def set_Extra_Parameter_values(value):
        if data is None or value == "":
            raise PreventUpdate

        Parameter_options=[ {'label': i, 'value': i} for i in np.sort(getattr(data,value).unique())]
        return Parameter_options

    # main callback for plot generation
    @app.callback([
        Output({'type' : "price-chart", 'index' : MATCH}, "figure"),Output({'type' : "other-charts", 'index' : MATCH},"children")],
        [
            Input({'type' : "do-plots", 'index' : MATCH}, "n_clicks"),
            Input({'type' : "compare", 'index' : MATCH}, "n_clicks"),
            State({'type' : 'years-slider', 'index' : MATCH}, "value"),
            State({'type' : "YAxis-filter", 'index' : MATCH}, "value"),
            State({'type' : "Parameter-filter", 'index' : MATCH}, "value"),
            State({'type' : "Parameter-column" , 'index' : MATCH}, "value"),
            State({'type' : "Config-Checklist", 'index' : MATCH}, "value"),
            State({'type' : "Models-Checklist", 'index': MATCH}, "value"),
            State({'type' : "Anomaly-Checklist", 'index': MATCH}, "value"),
            State({'type' : "Extra-Parameter-filter", 'index' : MATCH}, "value"),
            State({'type' : 'store-output', 'index' : MATCH}, 'data'),
            State({'type' : 'separators-output', 'index' : MATCH}, 'data'),
        ], prevent_initial_call=True
    )
    def update_charts(n_clicks,compare_clicks,constraint,attribute,Parameter,parameter_column,checklist_values,model_values, anomaly_checklist, extra_parameter, model_params, extra_separator_params):

        if data is None:
            raise PreventUpdate

        filtered_data = data

        # set the filtered data
        if Parameter:
            mask = (
                (getattr(data, parameter_column) == Parameter)
                & (data.index >= unixToDatetime(constraint[0]))
                & (data.index <= unixToDatetime(constraint[1]))
            )
            if len(extra_separator_params) != 0:
                for extra_separator,extra_separated_parameter in extra_separator_params.items() :
                    small_mask = (
                (getattr(data,extra_separator) == extra_separated_parameter[0])
            )
                    mask &= small_mask
            filtered_data = data.loc[mask, :]
        else:
            mask = (
                (data.index >= unixToDatetime(constraint[0]))
                & (data.index <= unixToDatetime(constraint[1]))
            )

            filtered_data = data.loc[mask, :]
            Parameter = attribute

        # plotly layout
        layout = {"xaxis": {"title": "Time"}, "yaxis": {"title": attribute}}

        express_chart_figure = go.Figure(layout=layout)

        express_chart_figure.update_layout(hovermode='x unified')
        express_chart_figure.update_layout(paper_bgcolor="#1f2630",
                plot_bgcolor="#f2f2f2",
                font=dict(color="#2cfec1"),
                autosize=True)

        # plotly traces
        express_chart_figure.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data[attribute],
                    mode='lines',
                    name=Parameter,
                    showlegend=True))

        for value in checklist_values:
            if value != "":
                add_trace(value,express_chart_figure,filtered_data,attribute)

        children = []

        #model-checklist
        for value in model_values:
            if value != "":
                if value =='decompose' :
                    for im in ('trend','seasonal','resid','observed'):
                        show_trace_decompose(im,filtered_data,attribute,model_params['decomposition'],express_chart_figure)

                elif value == "correlation" or value == "arima" or value == "sarima":
                    plotly_figure = do_model(value,filtered_data,attribute,model_params)
                    graph = dcc.Graph(figure=plotly_figure)
                    children.append(graph)

        for value in anomaly_checklist:
            show_anomaly_detection(value,filtered_data,attribute,express_chart_figure)

        #comparable attribute
        if extra_parameter:
            mask = (
                (getattr(data, parameter_column) == extra_parameter)
                & (data.index >= unixToDatetime(constraint[0]))
                & (data.index <= unixToDatetime(constraint[1]))
            )
            if len(extra_separator_params) != 0:
                for extra_separator,extra_separated_parameter in extra_separator_params.items() :
                    small_mask = (
                (getattr(data,extra_separator) == extra_separated_parameter[1])
            )

                    mask &= small_mask

            extra_filtered_data = data.loc[mask, :]

            express_chart_figure.add_trace(go.Scatter(x=extra_filtered_data.index, y=extra_filtered_data[attribute],
                    mode='lines',
                    name=extra_parameter,
                    showlegend=True))

            if compare_clicks != 0:

                image = html.Img(id = "Comparison-over-time", src = compare(filtered_data,extra_filtered_data,attribute,Parameter,extra_parameter))
                children.append(image)

        if len(children) == 1:
            return express_chart_figure,children[0]
        else:
            return express_chart_figure,list(itertools.chain.from_iterable(children))

    return app

if __name__ == "__main__":

    app = configurations()
    app.run_server(debug=True)