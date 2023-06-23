import os
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc

import pandas as pd
import numpy as np
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

import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA as oldArima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm

import itertools
import plotly.io as pio
import plotly.graph_objects as go
from statsmodels.tsa.stattools import pacf, acf



# anomalies

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from numpy import sqrt, argsort
from sklearn.preprocessing import scale
from sklearn.metrics import silhouette_score
import pmdarima as pmd

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression


def fig_to_uri(in_fig, close_all=True, **save_args):

    out_img = BytesIO()
    in_fig.savefig(out_img, format='png', **save_args)
    if close_all:
        in_fig.clf()
        plt.close('all')
    out_img.seek(0)  # rewind file
    encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
    return "data:image/png;base64,{}".format(encoded)



def set_model_color(ax):


    ax.set_facecolor("#1f2630")
    ax.yaxis.label.set_color('#2cfec1')
    ax.xaxis.label.set_color('#2cfec1')
    ax.spines['bottom'].set_color('#2cfec1')
    ax.spines['top'].set_color('#2cfec1') 
    ax.spines['right'].set_color('#2cfec1')
    ax.spines['left'].set_color('#2cfec1')
    ax.tick_params(axis='x', colors='#2cfec1')
    ax.tick_params(axis='y', colors='#2cfec1')
    ax.title.set_color('#2cfec1')


def show_randomForest_model(content,atribute,parameters):

    # Assuming you have a DataFrame named 'df' with columns 'timestamp' and 'value'
    # Convert the time series data into a supervised learning format
    assembler = VectorAssembler(inputCols=["lag1", "lag2", "lag3"], outputCol="features")
    df_lagged = assembler.transform(df)

    # Split the data into training and test sets
    train_data, test_data = df_lagged.randomSplit([0.8, 0.2])

    # Train the Linear Regression model
    lr = LinearRegression(featuresCol="features", labelCol="value")
    model = lr.fit(train_data)

    # Make predictions on the test set
    predictions = model.transform(test_data)

    # View the predicted values
    predictions.select("timestamp", "value", "prediction").show()

def show_model(content,atribute,parameters,caller):

    train_size = int(len(content[atribute]) * 0.8)
    train_data, test_data = content[atribute][:train_size], content[atribute][train_size:]

    p = parameters['p-value']
    d = parameters['d-value']
    q = parameters['q-value']
    P = parameters['P-value']
    D = parameters['D-value']
    Q = parameters['Q-value']
    m = parameters['m-value']

    if q != None:
        if Q != None:
            model = SARIMAX(train_data, order=(p,d,q), seasonal_order=(P,D,Q,m))
        else:
            model = ARIMA(train_data, order=(p,d,q))

    model_fit = model.fit()
    predictions = model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)

    trace_original = go.Scatter(
        x=content[atribute].index,
        y=content[atribute].values,
        name='Original Data'
    )

# Create a trace for the predicted data
    trace_predicted = go.Scatter(
        x=test_data.index,
        y=predictions,
        name='Predicted Data'
    )

# Create the layout
    layout = go.Layout(
        title=f'{caller} Prediction',
        xaxis=dict(title='X-axis Label'),
        yaxis=dict(title='Y-axis Label')
    )

    # Create the figure and add the traces
    fig = go.Figure(data=[trace_original, trace_predicted], layout=layout)

    return fig


def show_trace_decompose(image,content,atribute,parameters,chart):

    decomposed = sm.tsa.seasonal_decompose(content[atribute],model = parameters['type']) # The frequncy is annual

    if image == 'seasonal':
        chart.add_trace(go.Scatter(x=content.index, y=decomposed.seasonal,
                    mode='lines',
                    name=image,
                    showlegend=True))
    if image == 'trend':
        chart.add_trace(go.Scatter(x=content.index, y=decomposed.trend,
                    mode='lines',
                    name=image,
                    showlegend=True))
    if image == 'resid':
        chart.add_trace(go.Scatter(x=content.index, y=decomposed.resid,
                    mode='lines',
                    name=image,
                    showlegend=True))
    if image == 'observed':
        chart.add_trace(go.Scatter(x=content.index, y=decomposed.observed,
                    mode='lines',
                    name=image,
                    showlegend=True))

def show_correlation(content,atribute, parameters):

    lag_value = parameters['value']
    mode = parameters['type']
    plot_pacf = False

    # fig, ax = plt.subplots()

    if mode == 'partial':
        plot_pacf = True

    corr_array = pacf(content[atribute].dropna(), alpha=0.05, nlags=lag_value) if plot_pacf else acf(content[atribute].dropna(), alpha=0.05, nlags=lag_value)
    lower_y = corr_array[1][:,0] - corr_array[0]
    upper_y = corr_array[1][:,1] - corr_array[0]

    fig = go.Figure()
    [fig.add_scatter(x=(x,x), y=(0,corr_array[0][x]), mode='lines',line_color='#3f3f3f') 
     for x in range(len(corr_array[0]))]
    fig.add_scatter(x=np.arange(len(corr_array[0])), y=corr_array[0], mode='markers', marker_color='#1f77b4',
                   marker_size=12)
    fig.add_scatter(x=np.arange(len(corr_array[0])), y=upper_y, mode='lines', line_color='rgba(255,255,255,0)')
    fig.add_scatter(x=np.arange(len(corr_array[0])), y=lower_y, mode='lines',fillcolor='rgba(32, 146, 230,0.3)',
            fill='tonexty', line_color='rgba(255,255,255,0)')
    fig.update_traces(showlegend=False)
    fig.update_xaxes(range=[-1,42])
    fig.update_yaxes(zerolinecolor='#000000')
    
    title='Partial Autocorrelation (PACF)' if plot_pacf else 'Autocorrelation (ACF)'
    fig.update_layout(title=title)
    
    return fig




def compare(filtered_data,extra_filtered_data,attribute,pam1,pam2):

    normalized_parameter = filtered_data[attribute].div(filtered_data[attribute].iloc[0]).mul(100)
    normalized_extra_parameter = extra_filtered_data[attribute].div(extra_filtered_data[attribute].iloc[0]).mul(100)

    figure = plt.figure()

    figure.set_facecolor("#1f2630")

    ax = plt.axes()
    set_model_color(ax)

    plt.plot(normalized_parameter,color = 'red')
    plt.plot(normalized_extra_parameter,color = 'blue')
    plt.legend([pam1,pam2])
    plt.title('Comparison over time')


    out_url = fig_to_uri(plt)

    return out_url


def do_model(model_value,filtered_data,atribute, model_params):

    if model_value == 'arima':
        model = show_model(filtered_data,atribute, model_params['arima'], "ARIMA")

    if model_value == 'sarima':
        model = show_model(filtered_data,atribute, model_params['sarima'], 'SARIMA')

    if model_value == 'correlation':
        model = show_correlation(filtered_data,atribute, model_params['correlation'])

    return model


def checkModelValue(value,index):

    addable = []

    if value == 'decompose':
        addable.append(dbc.Row( id = "modelBox",
                        children=[
                            html.Div(children="Choose decomposition type", className="menu-title"),
                            dcc.RadioItems(
                                id={"type" : "Decompose-Radio", "index" : index},
                                options=[
                                    {'label': 'Additive', 'value': 'addi'},
                                    {'label': 'Multiplicative', 'value': 'multi'},
                                ],
                                value = "",
                                className="radioItems",
                            ),
                        ]
                    ),
            )

    if value == 'correlation':
        addable.append(dbc.Row( id = "modelBox",
                        children=[
                            html.Div(children="Choose correlation type", className="menu-title"),
                            dcc.RadioItems(
                                id={"type" : "Correlation-Radio", "index" : index},
                                options=[
                                    {'label': 'Auto', 'value': 'auto'},
                                    {'label': 'Partial', 'value': 'partial'},
                                ],
                                value = "",
                                className="radioItems",
                            ),

                            html.Div(children="Correlation Lag", className="menu-title"),
                            dcc.Input(
                                id={"type" : "lag-option", "index" : index},
                                type="number",
                                placeholder="input type {}".format("number"),
                                className="just-inputs",
                            )
                        ]
                    ),
        )

    if value == 'arima' or value == 'sarima' :
        addable.append(html.Div( id = "modelBox",
                        style={'fontColor': 'blue'},
                        children=[
                            html.Div(children="Custom " + value.upper() + " parameters:", className="menu-title"),
                            html.Div(children="Trend autoregression order", className="menu-title"),
                            dcc.Input(
                                id={"type" : "p-" + value + "-option", "index" : index},
                                type="number",
                                placeholder="input {}".format("p"),
                                className="just-inputs",
                            ),
                            html.Div(children="Trend difference order", className="menu-title"),
                            dcc.Input(
                                id={"type" : "d-" + value + "-option", "index" : index},
                                type="number",
                                placeholder="input {}".format("d"),
                                className="just-inputs",
                            ),
                            html.Div(children="Trend moving average order.", className="menu-title"),
                            dcc.Input(
                                id={"type" : "q-" + value + "-option", "index" : index},
                                type="number",
                                placeholder="input {}".format("q"),
                                className="just-inputs",
                            ),
                        ]
                    ),
            )

    if value == 'sarima':
        addable.append(html.Div( id = "modelBox",
                        style={'fontColor': 'blue'},
                        children=[
                            html.Div(children="Specific SARIMA parameters", className="menu-title"),
                            html.Div(children="Seasonal autoregressive order", className="menu-title"),
                            dcc.Input(
                                id={"type" : "P-" + value + "-option", "index" : index},
                                type="number",
                                placeholder="input {}".format("p"),
                                className="just-inputs",
                            ),
                            html.Div(children="Seasonal difference order", className="menu-title"),
                            dcc.Input(
                                id={"type" : "D-" + value + "-option", "index" : index},
                                type="number",
                                placeholder="input {}".format("d"),
                                className="just-inputs",
                            ),
                            html.Div(children="Seasonal moving average order", className="menu-title"),
                            dcc.Input(
                                id={"type" : "Q-" + value + "-option", "index" : index},
                                type="number",
                                placeholder="input {}".format("q"),
                                className="just-inputs",
                            ),
                            html.Div(children="Nr. of time steps for a single seasonal period", className="menu-title"),
                            dcc.Input(
                                id={"type" : "m-" + value + "-option", "index" : index},
                                type="number",
                                placeholder="input {}".format("q"),
                                className="just-inputs",
                            )
                        ]
                    ),
            )

    if value == 'arima':
        addable.append(dcc.Input(id={"type" : "P-" + value + "-option", "index" : index},style={'display': 'none'}))
        addable.append(dcc.Input(id={"type" : "D-" + value + "-option", "index" : index},style={'display': 'none'}))
        addable.append(dcc.Input(id={"type" : "Q-" + value + "-option", "index" : index},style={'display': 'none'}))
        addable.append(dcc.Input(id={"type" : "m-" + value + "-option", "index" : index},style={'display': 'none'}))

    return addable



def show_KClustering(filtered_data,attribute,chart):

    range_n_clusters = [2, 3, 4, 5, 6]

    silhouette_value = 0
    actual_n_clusters_value = 1


    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters, random_state=0)
        cluster_labels = clusterer.fit_predict(filtered_data[attribute].values.reshape(-1,1))

        silhouette_avg = silhouette_score(filtered_data[attribute].values.reshape(-1,1), cluster_labels)

        if float(silhouette_avg) > silhouette_value:
            silhouette_value = float(silhouette_avg)
            actual_n_clusters_value = n_clusters

    model = KMeans(n_clusters = actual_n_clusters_value, max_iter=10, random_state=0, algorithm="elkan").fit(filtered_data[attribute].values.reshape(-1,1))

    filtered_data['anomaly']=model.predict(filtered_data[attribute].values.reshape(-1,1))

    pd.set_option("display.max_rows", None, "display.max_columns", None)

    mask = (
        (filtered_data.anomaly == actual_n_clusters_value - 1)
   )

    filtered_data = filtered_data.loc[mask, :]



    chart.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data[attribute],
                mode='markers',
                name="K-Means Anomalies",
                showlegend=True))


def show_RunForestRun(filtered_data,attribute,chart):

    model = IsolationForest(n_estimators=50, max_samples='auto', contamination=float(0.05),max_features=1.0)

    model.fit(filtered_data[attribute].values.reshape(-1, 1))

    filtered_data['scores']=model.decision_function(filtered_data[attribute].values.reshape(-1, 1))
    filtered_data['anomaly']=model.predict(filtered_data[attribute].values.reshape(-1,1))

    mask = (
        (filtered_data.anomaly == -1)
   )

    filtered_data = filtered_data.loc[mask, :]

    chart.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data[attribute],
                mode='markers',
                name="Isolation Forest Anomalies",
                showlegend=True))


def show_anomaly_detection(algorithm, filtered_data, attribute, chart):

    if algorithm == 'kmean':
        show_KClustering(filtered_data,attribute,chart)
    elif algorithm == 'forest':
        show_RunForestRun(filtered_data,attribute,chart)