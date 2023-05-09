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
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm

import itertools



# anomalies

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from numpy import sqrt, argsort
from sklearn.preprocessing import scale
from sklearn.metrics import silhouette_score
import pmdarima as pmd


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



def show_model(content,atribute,parameters,caller):


    train, test = train_test_split(content[atribute], test_size=0.05 ,shuffle=False) 
    res = (pd.Series(content.index[1:]) - pd.Series(content.index[:-1])).value_counts()

    p = parameters['p-value']
    d = parameters['d-value']
    q = parameters['q-value']
    P = parameters['P-value']
    D = parameters['D-value']
    Q = parameters['Q-value']
    m = parameters['m-value']

    if q != None:
        if Q != None:

            print("***Entering SARIMA******")
            #train_resampled = train.resample('Q').mean()
            model = sm.tsa.SARIMAX(train.values,order=(p,d,q),seasonal_order=(P,D,Q,m),trend='n', enforce_stationarity=False, enforce_invertibility=False)
            result = model.fit()
            prediction= result.forecast(len(test)*2, freq = res.index.min(), dynamic = True)
            index_of_fc = pd.date_range(content.index[-1], periods = len(test) + 1,freq = res.index.min())
            combined = test.index.union(index_of_fc)
            fitted_series = pd.Series(prediction, index=combined)

        else:

            print("***Entering ARIMA******")
            model = ARIMA(train.values, order=(p,d,q))
            result = model.fit()
            prediction= result.forecast(len(test)*2, freq = res.index.min())
            index_of_fc = pd.date_range(content.index[-1], periods = len(test) + 1,freq = res.index.min())
            combined = test.index.union(index_of_fc)
            fitted_series = pd.Series(prediction, index=combined)

    else:

        print("***Entering ARMA******")
        model = ARMA(train.values, order=(p,d))
        result = model.fit()
        prediction= result.predict(len(train) ,len(train) + len(test)*2)
        index_of_fc = pd.date_range(content.index[-1], periods = len(test) + 1,freq = res.index.min())
        combined = test.index.union(index_of_fc)
        fitted_series = pd.Series(prediction[:-1], index=combined)



    print("AAAAAAA  --> " +  str(res.index.min()))
    figure = plt.figure(figsize=(12,5))


    plt.plot(train, label='training')
    plt.plot(test, label='actual')
    plt.plot(fitted_series, color='darkgreen',label='forecast')

    plt.legend(loc='upper left', fontsize=8)
    
    plt.title(caller + ' for ' + atribute)

    figure.set_facecolor("#1f2630")

    ax = plt.axes()
    set_model_color(ax)


    out_url = fig_to_uri(figure)
    return out_url


def show_trace_decompose(image,content,atribute,parameters,chart):

    decomposed = sm.tsa.seasonal_decompose(content[atribute],model = parameters['type'],period = 360) # The frequncy is annual

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




def show_decompose(image,content,atribute,parameters):

    decomposed = sm.tsa.seasonal_decompose(content[atribute],model = parameters['type'],period = 360) # The frequncy is annual

    if image == 'seasonal':
        trendi = decomposed.seasonal.plot()
        plt.title('Seasonality of ' + atribute)
    if image == 'trend':
        trendi = decomposed.trend.plot()
        plt.title('Trend of ' + atribute)
    if image == 'resid':
        trendi = decomposed.resid.plot()
        plt.title('Residual of ' + atribute)
    if image == 'observed':
        trendi = decomposed.observed.plot()
        plt.title('Observed of ' + atribute)


    figure_trendi = trendi.figure
    figure_trendi.set_facecolor("#1f2630")

    ax = plt.axes()
    set_model_color(ax)

    out_url = fig_to_uri(figure_trendi)
    return out_url

def show_correlation(content,atribute, parameters):

    lag_value = parameters['value']
    mode = parameters['type']

    if mode == 'auto':
        model = plot_acf(content[atribute],lags=lag_value,title=atribute)
    else:
        model = plot_pacf(content[atribute],lags=lag_value,title=atribute)

    model.set_facecolor("#1f2630")
    plt.title( mode + "correlation for " + atribute)

    ax = plt.axes()
    set_model_color(ax)

    out_url = fig_to_uri(model)
    return out_url



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
    print(out_url)

    return out_url


def do_model(model_value,filtered_data,atribute, model_params):

    if model_value == 'arma':
        model = show_model(filtered_data,atribute, model_params['arma'], "ARMA")
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

    if value == 'arma'  or value == 'arima' or value == 'sarima' :
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
                        ]
                    ),
            )

    if value == 'arima' or value == 'sarima':
        addable.append(html.Div( id = "modelBox",
                        style={'fontColor': 'blue'},
                        children=[
                            html.Div(children="Specific Custom " + value.upper() + " parameters", className="menu-title"),
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

    if value == 'arma':
        addable.append(dcc.Input(id={"type" : "q-" + value + "-option", "index" : index},style={'display': 'none'}))
    if value == 'arima' or value =='arma':
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
            print(actual_n_clusters_value)

    model = KMeans(n_clusters = actual_n_clusters_value, max_iter=10, random_state=0, algorithm="elkan").fit(filtered_data[attribute].values.reshape(-1,1))

    filtered_data['anomaly']=model.predict(filtered_data[attribute].values.reshape(-1,1))

    pd.set_option("display.max_rows", None, "display.max_columns", None)
    print(filtered_data['anomaly'])
    print(type(filtered_data['anomaly']))

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