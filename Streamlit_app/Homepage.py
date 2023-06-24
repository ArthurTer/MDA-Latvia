import pandas as pd 
import numpy as np
import os
import random
import glob
import re
import pickle
import joblib

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, QuantileTransformer
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate,GridSearchCV
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score, recall_score,precision_recall_fscore_support, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from shapely.geometry import Point, Polygon
import calplot 
import folium
import streamlit as st

import plotly.express as px 
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

from time import time 



from streamlit_folium import st_folium
from  utils import (noiseprocessing, groupdata, meteoprocessing, groupmeteo, double_lineplot, mapoutliers,longitude,latitude, 
                    weather_sound_map, read_data, noisecat, school_calendar, time_of_day,min00, event_type_freq, get_coord, inbound)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", category=FutureWarning, module="Matplotlib")

random.seed(36)


st.set_page_config(layout="wide")


#########################################
#       User input below on row 59      #
#########################################


# On the row below this set of code, between the brackets you have to set the path to the folder called 'Data'
# This folder can be found in the following path of the downloaded folder called App: App/Data 
# An example is: dir_data_folder="/Users/jonas/Documents/KU Leuven/Year 4/Semester 2/Modern Data Analytics/Assignment/App/Data/"

dir_data_folder="Data"



#########################################
#       User input above on row 59      #
#########################################

dirmeteo=os.path.join(dir_data_folder, "Meteo/*.csv")
dirclass=os.path.join(dir_data_folder, "Classified_events/*.csv")
dirpercent=os.path.join(dir_data_folder, "Percentiled_data/*.csv")
direvent=os.path.join(dir_data_folder, "Events/*.csv")


LSTM_checkpoint=os.path.join(dir_data_folder, "Models/LSTM_checkpoint.ckpt")
History_LSTM_file=os.path.join(dir_data_folder, "Models/History_LSTM.pkl")

MLP_file_sav=os.path.join(dir_data_folder, "Models/MLP_regr_model.sav")
MLP_classifier_file=os.path.join(dir_data_folder, "Models/MLP_class_model.sav")

Weather_Metadata=os.path.join(dir_data_folder, "01_Metadata_v2.csv")
Sound_weather_file=os.path.join(dir_data_folder, "Sound_weather.csv")



metadata=pd.read_csv(Weather_Metadata)

# Set title and caption
st.title('Noise in Leuven')
#st.caption('MDA-Latvia')

# Introduction text
st.markdown("For this assignment, our group was given multiple datsets from two different data sources. \
    The first source was on noise in Leuven. In the year 2022, for multiple locations at the Naamsestraat \
    in Leuven, sensors were measuring noise. The second source of data was weather data for Leuven in 2022. \
    Next to the provided data, we also collected and implemented external data. All data will be described \
    and visualised in their different subpages and below.")

st.markdown("The assignment offered the freedom to come up with your own research question. As the reason to \
    install those noise measuring sensors was to reduce the level of noise at night and offer the people of Leuven \
    a better night of sleep, our team has decided to focus on nightly noise levels for this assignment. \
    We try to predict the hours at which noise levels along the Naamsestraat are too high so that the police \
    can know beforehand that at which hours and day they might want to patrol more to prevent the noise.")

st.markdown("The model will be explained more in its own subpage. To decide on the input features, we've based \
    ourselves on academic literature on urban noise pollution. The last subpage of this app shows the results.")

st.markdown("As mentioned above, the input of our model is based on the acadamic literate on the topic. \
    we've read multiple studies and looked at other models on urban noise pollution to determine the general \
    drivers of it. While every model and city is different, this literature study did help us to point us towards \
    additional data we had to include. In general, the possible sources of noise that were mentioned in the literature\
    are:")

"""
* Traffic
* Airplanes
* Construction and (road) work
* Humans
* Weather
* Events
"""

st.markdown("From this list, airplane traffic isn't that relevant for noise in Leuven as the flightpaths for Zaventem \
    airport don't go over Leuven. Data on traffic is provided as part of the noise dataset and will be used in the model.\
    Human noise is also provided in the same dataset. More info on those can be found in the page on noise data.\
    Data on the weather is provided by the weather stations in and around Leuven. For the last two sources we had to look for dataset.\
    Both are part of the GIPOD dataset that is a collaboration of Flemish organisations to collect and organise all possible\
    happenings that could cause hindrance in the public domain. Example of such instances are events or workings on the sewage system.")

st.markdown("---")

# Show map of measuring locations
st.markdown("The following map shows the measuring stations for both the weather data and the noise data. \
    Red points show the noise locations, blue the weather stations.")

# Set up variables for map
Noise=noiseprocessing(read_data(dirpercent,';'))
Micro_loc=pd.DataFrame(data=zip(Noise['Location'].unique(),latitude,longitude),columns=['Loc','LAT','LON'])

# Create map
map=weather_sound_map(Micro_loc,metadata,width=500,height=500)
st_map=st_folium(map,width=500,height=500)

################################
st.markdown("---")

# Show two most important datasets
st.markdown('Below, one can see a snapchot of both datasets that were used as a foundation to the project.')
st.write("")

meteo=meteoprocessing(read_data(dirmeteo,','))

col1, col2, col3= st.columns([1.5,6,1.5])
with col1:
    st.write("")

with col2:
    st.write(meteo.sample(n=5))

with col3:
    st.write("")
   
st.write("")
st.markdown("Here, two things are worth noting. The temperature used is a so-called 'corrected' temperature, \
    to account for the weather's stations measurement error. On the other hand, there are multiple weather stations. \
    In general, when talking about the temperature, we will refer to the average temperature across the weather stations.")
st.write("")


st.write(Noise.sample(n=5))

st.markdown("Once again, the data deserves some explanation. In this dataset, the sound is compiled over an hour \
    and the amount of noise per percentile is given.")

dailydf=groupdata(Noise,'Date')
daily_weather=groupdata(meteo,'Date')

st.markdown("---")

