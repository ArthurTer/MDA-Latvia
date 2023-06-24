import streamlit as st
import pandas as pd
import numpy as np 
import pickle
import os 

import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, MeanSquaredLogarithmicError

import plotly.express as px 
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from sklearn.metrics import mean_squared_error
from utils import Preprocess_to_sequence, Validation_Loss_Accuracy
from Homepage import dir_data_folder

st.set_page_config(layout="wide")

# Import the datasets from the other pages
pivot = st.session_state['pivot']
Sound = st.session_state['Sound']
detected_noise = st.session_state['detected_noise']
Meteo = st.session_state['Meteo']
Year_events = st.session_state['Year_events']

#Load the model and history:
LSTM_checkpoint=os.path.join(dir_data_folder, "Models/LSTM_checkpoint.ckpt")
History_LSTM_file=os.path.join(dir_data_folder, "Models/History_LSTM.pkl")
History=pickle.load(open(History_LSTM_file, 'rb'))
LSTM_Model = tf.keras.models.load_model(LSTM_checkpoint)

# The Dataset we merge all the others on is the percentiled noise one, every merge is based on the hour, 'Date_time' except from the first one where we want to keep the locations specific.
# We then merge the dataset containing the classifications, not all the date appears so we will fill the missing values with 0. 
# Because we assume no cars, shouting or singing were detected. Next we mearge the dataset with the events, droping the variable with the list of events.
# And, only keeping the columns with the counts of those events. Lastly we add our weather dataset. 

Sound_weather= pd.merge(Sound,pivot,on=['Date_time','Location'],how='left').fillna(0).merge((Year_events.drop('Event_type',axis=1)),on=['Date_time'],how='left').merge(Meteo,on=['Date_time'],how='left')


#To that we make some adjustements, essentially minor feature engeneering such as:

#Extracting the month of the year
Sound_weather['Month'] = Sound_weather['Date_time'].apply(lambda w: w.strftime('%B'))

#The day of the week
Sound_weather['Day_of_week'] = Sound_weather['Date_time'].apply(lambda w: w.strftime('%A'))

#And the hour of the day 
Sound_weather['Hour']=Sound_weather['Date_time'].dt.hour.astype(str)

#One specific date and time appears twice, we therefore delete the duplicates 
Sound_weather=Sound_weather.drop_duplicates(['Date_time','Location'],keep='last')

#Sort the dataframe for upcoming manipulations
Sound_weather=Sound_weather.sort_values(['Location','Date_time'])

#And create an index from one to 50320, sorted by Location and Date_time
Sound_weather.reset_index(drop=True, inplace=True)




st.title("The Model")
st.subheader("Long-Short-Term-Memory neural networks")

st.markdown("The output of the model is the decibel value that belongs to the a certain percentages of the hour.\
            ")
st.markdown("To predicted the noise level of a given hour using a sequence of the previous hour we will make use of the Long-Short-Term-Memory neural networks architecture.\
            Unlike simpler neural network architectures, which use a feedforward method, where inputs are fed directly to the outputs via a series of weights, LSTM is based on \
            feedback connections. Often used in speech or handwriting recognition, this architecture can process not only data points but entire sequences of data, \
            which in the case of multi-varaible forcasting, can become a powerful asset.")

st.markdown("The key to LSTM is that it uses different connections that assess events further back in the time sequence from those closer to the time of predictions, creating the so-called \"memory\". \
            In broad terms, the architecture allow weights to be assigned based on how relevant a past instance is. \
            In this case, one could have kept the actual noise value of an hour as an input variable for the next one.\
            However, it is assumed that the microphones on Naamestraat won't stay there forever, and the goal of our model is to predict noise level without having the noise recorded. \
            For the time being, the count of classified recorded events was kept in the model." )

st.markdown("Our model has as input these three datasets:")
"""
* Noise classifications
* Weather data
* Gipod hindrance counts
"""
st.markdown("We merge the three datasets on the hour of each day. We also create some variables to indicate the\
    day, week and month of the year. As well as variables indicating if a given hour is a holiday, during the night, evening or day.\
    The noise variable selected was the 25th percentile laf per hour. The amount of noise that was exceeded for 15 minutes in a given hour.")

sample_Sound_weather = Sound_weather.head(100)
st.dataframe(sample_Sound_weather)

COLUMNS_TO_KEEP=['Date_time', 'Location', 'Vacation', 'Time_of_Day',
       'Shouting', 'Singing', 'Music', 'Wind', 'Car', 'Siren', 'Evenement',
       'Werk', 'Grondwerk', 'avg_LC_HUMIDITY', 'avg_LC_RAD', 'avg_LC_RAININ',
       'avg_LC_DAILYRAIN', 'avg_LC_WINDDIR', 'avg_LC_WINDSPEED',
       'avg_LC_TEMP_QCL3', 'Month', 'Day_of_week', 'Hour']

st.markdown("From the initial dataset above we will keep the following variables: " )
st.write(*COLUMNS_TO_KEEP)


X_seqtrain, X_seqtest, y_seqtrain, y_seqtest,X_seqval,y_seqval=Preprocess_to_sequence(Sound_weather,COLUMNS_TO_KEEP,'laf25_per_hour',window=24)

set_dict={}

list_ysets=[y_seqtrain,y_seqtest,y_seqval]
keys=['train','test','val']

for i, set_array in enumerate(list_ysets):
    key=keys[i]
    set_dict[key] = set_array[:,[0,1]]

y_seqtrain,y_seqtest,y_seqval=np.delete(y_seqtrain, [0,1], axis=1).astype('float32'),np.delete(y_seqtest, [0,1], axis=1).astype('float32'),np.delete(y_seqval, [0,1], axis=1).astype('float32')

SHAPE=(X_seqtrain.shape)


st.markdown(" Since we need to create a sequential dataframe, some reshaping and processing is necessary. \
            First of all, we need to set our sequence window, which is the amount of hours we will feed to our model.\
            If the window is set to 12 then each sequence will contain the data of the 12 previous hours. \
            We will choose a window of 24 hours, arbitrarily, it might be that the window affects the predictions of the model. \
            Every sequence of 24 hours, containing with the data shown above, will be used to predict the 25th one. ")




st.markdown('We have created t pairs of training, testing and validation sets with our input and output variables. Those respectively account for 65, 20 and 15 percent of the dataset.\
As a result of our processing we have also created a large number of new dimensions. Since all the categorical variables were dummy encoded, we now have 73 variables in total.\
The final product of our reshaping is a training set with the following shape: {} '.format(SHAPE))


st.markdown("32579 being the amount of 24 hour sequences, 24 being equal to our window, the number of periods we look back into. And each one of those periods contianing 73 measurements, or observations")

st.markdown("Using tensorflow and the sequential model, we will use our training and test set to calibrate our model. Our model has the following architecture: ")

LSTM_Model.summary(print_fn=lambda x: st.text(x))

st.markdown("We trained our model on our training dataset and evaluated it using our test set.\
            We selected the best model based on our loss function\'s metric being  mean squared logarithmic error.\
            MSLE was chosen as a loss function. The reason behind such a choice is because our output variable varies over a wide range of variables \
            and is prone to outliers, i.e, high level of noise. LMSE emphasises the model to reduce errors on samples with larger differences between true and predicted values.\
            Those are the cases that will arise when predicted a high value of noise, which we are mainly concerned about.")


Training=Validation_Loss_Accuracy(History,100,1200,400)
st.plotly_chart(Training)


st.markdown('The first few epochs were left out of the graph for the sake of readability but this is the model behaviour throughout training. In the end, the best MSLE ')
print('In the end, the best MSLE achieved on the test set was: '+ str(np.min(History.history['val_mse']))+' and was reached at the '+ str(np.argmin(History.history['val_mse'])+1)+'th epoch.')



PREDICTIONS=(LSTM_Model.predict(X_seqval).flatten())
RESIDUALS=y_seqval.flatten()-PREDICTIONS
LOCATION_VAL=(set_dict['val'][:,0]).flatten()
DATE_TIME_VAL=pd.to_datetime(((set_dict['val'][:,1])).flatten())

val_set_score=mean_squared_error(y_seqval.flatten(),PREDICTIONS)

st.markdown("The score on our validation test on the other hand is: "+ str(val_set_score) + ", which is promising.\
             But let's dive deeper into how our model is performing.")


LSTM_result=pd.DataFrame(data={'Predictions':PREDICTIONS,'True_values':(y_seqval.flatten()),'Location':LOCATION_VAL,'Date_time':DATE_TIME_VAL,'Residuals':RESIDUALS})


st.session_state['LSTM_result'] = LSTM_result