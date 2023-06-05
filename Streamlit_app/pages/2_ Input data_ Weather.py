from Homepage import dirpercent, dirclass, dirmeteo, dirpercent
from ast import fix_missing_locations
import pandas as pd
import random

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import geopandas as gpd
from shapely.geometry import Point, Polygon
import calplot
import folium
import streamlit as st

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

from time import time
import datetime


from streamlit_folium import st_folium
from utils import (noiseprocessing, groupdata, meteoprocessing, groupmeteo, mapoutliers, longitude, latitude,
    weather_sound_map, read_data, noisecat, school_calendar, time_of_day, min00, event_type_freq, get_coord, inbound)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", category=FutureWarning, module="Matplotlib")


random.seed(36)


st.set_page_config(layout="wide")

st.title('Weather Data')

st.markdown("The Meteo dataset has records coming from a hundred of different weather stations spread accross \
    Leuven. We will average the records across the different stations to obtain the weather per hour in Leuven. \
    There are records of weather every 10 minute. The hourly weather will be obtained by averaging the measures \
    over a given hour.")


# Another kind of preprocessing is applied to this dataset.
Base_Meteo = meteoprocessing(read_data(dirmeteo, ','))
# The meteo grouped and average across all weather stations and hours, and to allow some visualisation with the noise dataset, by day.
Meteo = groupmeteo(Base_Meteo, 'Hour')
daily_weather = groupmeteo(Base_Meteo, 'Date')

# Merging the two datasets
Sound = noiseprocessing(read_data(dirpercent, ';'))
hourly_sound = groupdata(Sound, 'Hour')
daily_sound = groupdata(Sound, 'Date')
Sound_Weather_day = pd.merge(daily_weather, daily_sound, on='Date', how='outer')

st.markdown("---")

st.markdown("The following graph shows the daily temperature in Leuven (left vertical axis), which is the average \
    temperate of all weather stations. The other line shows the value that correspond to the 25% highest decibel \
    level of that day (right vertical axis).")

# Creating the plot

fig1_weather = make_subplots(specs=[[{"secondary_y": True}]])

fig1_weather.add_trace(
    go.Scatter(
        x=Sound_Weather_day['Date'],
        y=Sound_Weather_day['avg_LC_TEMP_QCL3'],
        name='Daily temperature',
        mode='lines',
        fillcolor='red'
    ),
    secondary_y=False
)

fig1_weather.add_trace(
    go.Scatter(
        x=Sound_Weather_day['Date'],
        y=Sound_Weather_day['avg_laf25_per_hour'],
        name='Daily Noise (25th percentile)',
        marker=dict(color='rgb(0,0,139)')
    ),
    secondary_y=True
)
fig1_weather.update_layout(xaxis_title='Day')

st.plotly_chart(fig1_weather)

st.markdown("---")

st.markdown("This graph shows the daily humidty, it's the average of all stations.")

# Create figure
fig2_weather = px.line(daily_weather, x='Date', y='avg_LC_HUMIDITY', title='Average Daily Humidity in Leuven (2022)')
fig2_weather.update_yaxes(title='Relative Humidity (%)')

st.plotly_chart(fig2_weather)

st.markdown("---")

st.markdown("This graph shows the daily rain in milimeters, it's the average of all stations.")

# Create figure
fig3_weather = px.line(daily_weather, x='Date', y='avg_LC_DAILYRAIN', title='Average Daily Rain in Leuven (2022)')
fig3_weather.update_yaxes(title='Rain in mm')

st.plotly_chart(fig3_weather)

st.markdown("---")

st.markdown("This graph shows the wind speed, it's the average of all stations. The original data are in meters\
    per second. There is also the option to select kilometers per hour, which is 3.6 times as large.")

# Compute average windspeed from m/s to km/h
daily_weather['avg_LC_WINDSPEED_kmh'] = daily_weather['avg_LC_WINDSPEED'] * 3.6

# Create figure
ticker = st.selectbox(
    'Select Source',
    ['avg_LC_WINDSPEED', 'avg_LC_WINDSPEED_kmh'],
    index=0
)
fig4_weather = px.line(daily_weather, x='Date', y=ticker, title='Average Daily Wind Speed in Leuven (2022)')
fig4_weather.update_traces(marker=dict(color='#32a846'))
fig4_weather.update_layout(xaxis_title='Wind speed')

st.plotly_chart(fig4_weather)

st.markdown("---")

st.markdown("The first 100 rows of the weather data that will be used for our model is shown below.")

# Make dataframe of first 100 observations
sample_Meteo = Meteo.head(100)
st.dataframe(sample_Meteo)

st.markdown("---")

# Create session state for the dataset so it can be passed onwards to different pages of the app
st.session_state['Meteo'] = Meteo