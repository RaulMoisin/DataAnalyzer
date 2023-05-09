import os
import numpy
import dash
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
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

import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.ar_model import AutoReg

import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from math import sqrt
import itertools

from helpers import *
from models import *

import re
import time

def parse_data(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:

            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return df





def add_trace(value,figure,filtered_data,attribute):


    if value == 'mean':
        filt_data = filtered_data[attribute].expanding().mean()
    elif value == 'rmean':
        filt_data = filtered_data[attribute].rolling('90D').mean()
    elif value == 'std':
        filt_data = filtered_data[attribute].expanding().std()

    figure.add_trace(go.Scatter(x=filtered_data.index, y=filt_data,
                mode='lines',
                name=value,
                showlegend=True))


# date slider
def unixTimeMillis(dt):
    ''' Convert datetime to unix timestamp '''
    return int(time.mktime(dt.timetuple()))

def unixToDatetime(unix):
    return pd.to_datetime(unix,unit='s')

def getMarks(start, end, frequency, daterange):
    result = {}
    for i, date in enumerate(daterange):
        if(i%frequency == 1):
            # Append value to dict
            result[unixTimeMillis(date)] = str(date.strftime('%Y-%m-%d'))
    return result



