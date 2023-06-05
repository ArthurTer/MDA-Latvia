from ast import fix_missing_locations
import pandas as pd 
import calplot 
import streamlit as st
import plotly.express as px 
from plotly.subplots import make_subplots
from time import time 
from streamlit_folium import st_folium

from  utils import (noiseprocessing, groupdata, mapoutliers, read_data, min00)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", category=FutureWarning, module="Matplotlib")

from Homepage import dirpercent, dirclass


st.set_page_config(layout="wide")

st.title('Noise Data')

st.markdown("Noise in decibels:")


st.markdown("There are four different Datasets serving as input for our models. Two of them contain information \
    about noise, one being percentiled recorded noise per hour over the year 2022 and the other one containing \
    the predictions about the source of some of the noise sequences recorded over that same time period.\
    The Noise Data are recorded by hour in decibels, and percentiled by hour. Meaning, the k-th percentile represents the\
    amount of decibels below which the percentage k of the hour falls. For example, if the 25th percentile of a given\
    hour is 75 Db, that implies that the noise reached 75Db for 25% of the hour. (Loudest 25th% is at 75Db).\
    This is the dependent variable of our mode. Based on the independent variables, the aim is to predict the hours\
    at which a certain percentages of the hour a certain number of decibels is reached and thus can be seen as a \
    noisy hour.")

Sound= noiseprocessing(read_data(dirpercent,';'))
hourly_sound=groupdata(Sound,'Hour')
daily_sound=groupdata(Sound,'Date')

st.markdown("Classified noise Sequences:")

st.markdown("Some detected noise sequences were classified by some algorithm. We will use those to give us an\
    indication of traffic (as cars sometimes get detected) but also how many potentially disturbing noise sources\
    occur during a given hour. For that dataset to be merged with the one above, two things need to occur. We first\
    change all the times of the events into round hours and then compute the frequence of each event occuring in an\
    hour. We will also only keep the events that were classified with more than 50% certainty.")

# We import the Dataset 
Noise_class_perhour=noiseprocessing(read_data(dirclass,';'))

# All the hours are changed into round hours
Noise_class_perhour['Date_time']= noiseprocessing(read_data(dirclass,';'))['Date_time'].apply(lambda a: min00(a))
# Classifications with less than 50% certainty are being discarded
Noise_class_perhour=Noise_class_perhour.loc[(Noise_class_perhour['Certainty']>=50)]
# We create a column 'Count', equal to one. This will make the computation of our event classifications easier. 
Noise_class_perhour.loc[:,'count']=1

# We pivot the table based on the Date and time and Location for the  Class column and aggregate the corresponding values of the previously created "Count" column. 
# This creates a column for eaach category of the Class column indexed on time and Location, meaning we are kept with the count of each event by hour and location (matching our initial dataset size)
pivot=pd.pivot_table(Noise_class_perhour,values='count', index=['Date_time','Location'],columns='Class',aggfunc='count')
# The dataset is filled with missing values for the hours in which no events were detected, we want those to be equal to 0 instead and get rid of the double index set above. 
pivot=pivot.fillna(0).reset_index(level=['Date_time','Location'])
#pivot.describe()

# For the sake of readability and use we rename the columns
pivot.rename(columns={'Human voice - Shouting':'Shouting','Human voice - Singing':'Singing','Music non-amplified':'Music','Nature elements - Wind':'Wind','Transport road - Passenger car':'Car','Transport road - Siren':'Siren'},inplace=True)
# Create a list of the detected noise, just in case. 
detected_noise=['Shouting', 'Singing', 'Music', 'Wind', 'Car','Siren']

st.markdown("---")

st.markdown("This graph below shows how many times each noise source occured in 2022. Each bar represent six \
    consecutive days. Interesting in the 'shouting' data is to see the effect of the exam and holiday period.")

ticker = st.selectbox(
    'Select Source',
    ['Shouting', 'Singing', 'Music', 'Wind', 'Car', 'Siren'],
    index=0
)
fig1_noise = px.histogram(pivot, x=pivot['Date_time'].dt.date, y=ticker, color=ticker)
fig1_noise.update_traces(marker=dict(color='#32a846'))
fig1_noise.update_layout(xaxis_title='Period of the year')
fig1_noise.update_layout(showlegend=False)

st.plotly_chart(fig1_noise)

st.markdown("---")

st.markdown("In this graph the 25th percentile of every day is shown. Again, it's lower during the holiday periods and\
    during the weekend.")

fig2_noise = pd.concat([(pd.DataFrame(index=range(38), columns=daily_sound.columns)), daily_sound], ignore_index=True)

missing_day = pd.DataFrame({'Date': pd.date_range(start='2022-01-01', end='2022-02-07')})
fig2_noise['Date'] = fig2_noise['Date'].fillna(missing_day['Date'])
fig2_noise['Date'] = pd.to_datetime(fig2_noise['Date'], format='%Y-%m-%d %H:%M:%S')
Sound_calendar = pd.Series(fig2_noise.set_index('Date')['avg_laf25_per_hour'])
Calendarmap = calplot.calplot(Sound_calendar, dropzero='True', cmap='YlOrBr', figsize=(12, 4))

st.pyplot(Calendarmap[0].get_figure())

st.markdown("---")

st.markdown("These boxplots give more info on the different percentages-decibel values. To illustrate the different variables,\
    let's look at the one on the left. It's corresponding percentage is 0.5%. Its median value is 71.2. This means\
        that on average for all the hours of 2022, the top 0.5% loudest decibel had a value of 71.2. ")

plots=(Sound.drop(Sound.filter(regex='unit').columns,axis=1)
    .drop(['id', 'Vacation', 'Time_of_Day'], axis=1)).melt(id_vars=['Date_time', 'Date', 'Time', 'Location'])

fig3_noise = px.box(plots, x='variable', y='value', color='variable', color_discrete_sequence=px.colors.qualitative.T10)
fig3_noise.update_layout(xaxis_title='Variable', yaxis_title='Value in decibels')


st.plotly_chart(fig3_noise)

st.markdown("---")

st.markdown("This plot shows the outliers of the values that mark the top 0.5% loudest decibels of that hour. It's thus\
    the outliers of the highest scoring variable.")

fig4_noise=mapoutliers(Sound, 'laf005_per_hour')


st.plotly_chart(fig4_noise)

st.markdown("---")

st.markdown("The first 100 rows of the dataset on the decibel values per hour is shown below. This will be the dependent variable of the model.")

sample_Sound = Sound.head(100)

st.dataframe(sample_Sound)

st.markdown("---")

st.markdown("The first 100 rows of the dataset on the different types of noise for every hour that will be used in the model is shown below:")

sample_pivot = pivot.head(100)

st.dataframe(sample_pivot)

st.markdown("---")


# Create session state for the dataset so it can be passed onwards to different pages of the app
st.session_state['pivot'] = pivot
st.session_state['Sound'] = Sound
st.session_state['detected_noise'] = detected_noise



