
import streamlit as st
import pandas as pd
import numpy as np 
import pickle

import plotly.express as px 
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from utils import residuals_by
LSTM_result=st.session_state['LSTM_result']

st.title("Results")

st.markdown("During our preprocessing we made sure to keep aside the date and locations of our predicted values to be able to analyse\
            if there were any correlations between the residuals and those variables. This is the dataset with all predicted values, true values, corresponding residuals, locations and time.\
            The validation dataset contains" + str(len(LSTM_result)) + " observations")

st.dataframe(LSTM_result.head(n=5))
st.markdown("First we look at the relationship between the predicted and the true values. We plot them against each other, a perfect prediction would result in a straight line.")
fig2= px.scatter(LSTM_result,x='True_values',y='Predictions',labels={'x':'True_values','y':'Predictions'},color='Residuals', color_continuous_scale=px.colors.diverging.Picnic,height=500,width=1500)
fig2.add_shape(
    type="line", line=dict(dash='dash'),
    x0=LSTM_result['Predictions'].min(), y0=LSTM_result['Predictions'].min(),
    x1=LSTM_result['Predictions'].max(), y1=LSTM_result['Predictions'].max(),
    layer='above'
)

st.plotly_chart(fig2)

st.markdown("Aside from a few observations, our model seems to predict higher values quite accurately. Points below the straight line are under-predicted while the others are over-predicted.\
            The model seems to under-predict quite a bit despite the MSLE loss function. That being said, let's see if the residual behave uniformaly with respect to the true values")

fig3= px.scatter(LSTM_result,x='True_values',y='Residuals',labels={'x':'True values','y':'Residuals'},color=LSTM_result['Residuals'], color_continuous_scale=px.colors.diverging.RdYlBu[::-1],height=500,width=1500)
st.plotly_chart(fig3)


st.markdown('There seems to be some departure from the 0 line when true values are increasing, at least for some points, our model struggles to predict high values accurately.\
            Maybe there is some kind of pattern to the residuals, it might be that some time periods are correlated to our residuals.')

st.markdown('Over the course of the year, the residuals averaged accross every location and per day seem to be rather constant. The trend doesn\'t seem to follow any pattern \
            and is mostly influenced by "outliers". On a sidenote, those seem to occur mainly around the end of exams or holidays.')

fig4=residuals_by(LSTM_result,'Date')
fig4.update_layout(height=500,width=1500)
st.plotly_chart(fig4)


st.markdown('The relationship between residuals and our data might be more localised than that. It might be that some days are way louder than others, as some locations may be. \
            To assess that we will look at the densities of averaged residuals over days and locations')

col1, col2= st.columns([3,3])
with col1:
    fig6=residuals_by(LSTM_result,'Density_Day')
    st.plotly_chart(fig6)

with col2:
    fig7=residuals_by(LSTM_result,'Density_Location')
    st.plotly_chart(fig7)


st.markdown('As one can see, the average residual is very close to 0, once again, consistently under 0 which confirms that our model under-predicts. The model seems to be more consistent when it comes to predicting the the level of sound \
            throughout the week than accross all locations. This might be due to the fact that every hour of the week appears the same amount of time while some locations have more observations than others. Since we kept all locations\' \
            sequences independent from one another, the model is more likely to train on the locations which appear the most in the overall dataset.')
            

st.subheader('Conclusion')

st.markdown('While not being perfect our LSTM based model offers very satisfactory results and valued properties. Our model makes use of all the information available and predicts not only the noise during the night but also during the day. Also, the model is dynamic and easy to maintain.\
            Deployment is also easily done as it only requires information available before the time of prediction. The metric chosen was the maximum noise level exceeded for 15 minutes\
            , but setting up the model for any other metric would be simple, only the weights would have to be retrained. ')
st.markdown('The model could also be refined. It could be beneficial to try out different time windows and add some features related to the precise days of exams and or holidays.\
            Training the model for individual locations could also be an alternative to get more accurate predictions.\
            The noise level of the previous hour is not an input variable in our model because we assumed it would be difficult to compute the percentiled noise of an hour while using it as an imput to predict the noise level of the next one.\
            Nothing stops us to add more lags though, and predict the noise in two or more hours. An alternative would be to use the peak sound of the previous hour, which would, most likely be available by the time of prediction.')
st.markdown('Last but not least, the model offers a lot of versatility as it could easily be converted in a classification problem if that would be desired, the pre-processing and model\'s architecture would vary slightly,\
             the weights would have to be re-trained, but the framework would remain the same. If the model was to be used, we believe it would be a great tool to monitor and prevent disturbing level of noises in Naamsestraat before they occur.')