# DataAnalyzer
# Installation
pip3 install -r requirements.txt <br />
python3 SimpleDashboard.py

# Summary

It's a dashboard that allows the user to import any time series related dataset and provides an interactive way of analysing that data. The app is build around something called configuration blocks. It is a piece of html code that gets appended to the main layout through the pattern matching callback decorator from the dash framework. This block contains all the tools to analyse a time series; from prediction models (arma, arima, sarima) to anomaly detection algorithms and various mathematical lines. The niche thing is that the user can create as many of these configuration blocks as possible and that they are independent of each other and so it is possible to work on multiple attributes at once.

#Usage

Once the server starts, the user should upload the csv data which MUST reside in the data folder and start creating configuration blocks.
