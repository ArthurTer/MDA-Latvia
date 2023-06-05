import streamlit as st
import pandas as pd
import os
import plotly.express as px 

from Homepage import direvent
from utils import get_coord, inbound, datetime, event_type_freq, read_data


st.set_page_config(layout="wide")


st.title("Gipod Data")

st.markdown("After identifing the drivers of urban noise, we started to look for data that could capture events and \
    workings that took place in the city of Leuven. The GIPOD (Generiek Informatieplatform Openbaar Domein) dataset\
    contains those and more. GIPOD is an initiative of Informatie Vlaanderen to create a platform that collects\
    and lists all possible hindrances in the public domain in Flenders. Construction companies and utility companies\
    for example are obliged to add their works to the platform. In this way, different authorities and organisations can \
    have an easy overview of all hindrances.")

st.markdown("The portal is available for everyone to see the current and planned instances. However, two problems make \
    that unsuitable for our analysis. Firstly, as our noise data are on 2022, we need to have the GIPOD data of 2022.\
    Only organisations with special rights have access to these historical data. \
    Secondly, the portal that can be used by everyone only works on a case by case basis, meaning that info can be obtained on only\
    one instance at a time. We sent an email to the organisation behind GIPOD to ask if it's possible to obtain the \
    complete dataset of instances in 2022 in Leuven. They agreed and provided us with the anonimised data (that is,\
    without the name of the organisation or company). They send three different files for different periods of 2022.\
    Those files are merged and we dropped duplicates (based on their ID). Some events also don't mention a clear starting data, we remove \
    those as well.")

st.markdown("---")

Events=read_data(direvent,',')

# #Rename several columns
Events.rename(columns={'Einde':'End','Soort':'Sort','Categorie':'Category','Locatie':'Location','Id':'id'},inplace=True)

#We remove those events that start on different periods. And we  convert the start and end date columns into pandas datetime foramt 
Events=Events[Events['Start']!='meerdere periodes']
for i in ['Start','End']:
    Events[str(i)]=pd.to_datetime(Events[str(i)],format='%d-%m-%Y %H:%M')

#We drop duplicate events
Events=Events.sort_values('End',ascending=True).drop_duplicates(subset='id',keep='last')

st.markdown("The combined dataset looks like this, note that column names have been translated into English:")

st.write(Events.sample(n=5))

st.markdown("---")

st.markdown("We change the original data in several ways. First, as this dataset is on the entire city of Leuven, we only keep those that\
    are close to Naamsestraat. This dataset uses the Belgian Lambert 72 system to specify the location. We only keep those that fall within\
    a defined rectangle covering Naamstestraat, the points are:")

"""
* Top left [173230.000000, 174395.00000]
* Bottom left [173230.000000, 173230.0000]
* Top right [173500.00000, 174395.00000]
* Bottom right [173500.000000, 173230.0000]
"""

st.markdown("The dataset gives info on the sort of hindrance, the three values are: event, work, groundwork.\
    The effect of each on noise at night is ambiguous. Events is the most clear, as it's expected that events \
    lead to a higher noise level on the streets. The other two will have mostly increase noise levels during the day,\
    as that's when the actual work is done. However, they might also have a secondary effect at night. A container\
    that is placed at one side of the road makes that cars need to break and accelerate more often, which causes noise.\
    On the other hand, if it's a bigger project that blocks the whole road, there might be fewer people and less traffic.")

st.markdown("As our model uses hourly data, we create indicator variables that count the number of events, works and groundworks\
    happening at that certain hour. We don't specify it more based on the Type column,  since it has a lot of missing values.")

st.markdown("---")

#We will keep the events that solely happen in Naamsestraat
#Strictly Naamsestraat
Top_left=[173230.000000,174395.00000]
Bottom_left=[173230.000000,173230.0000]
Top_right=[173500.00000,174395.00000]
Bottom_right=[173500.000000,173230.0000]

R_limit=Bottom_right[0]
Down_limit=Bottom_right[1]
L_limit=Top_left[0]
Up_limit=Top_left[1]

#if the coordonates happen to be in the rectangle specified above we keep the event. Since the coordinate column contains strings this need some manipulation. 
Events['Coordinates']=Events['Location'].apply(lambda x: get_coord(x))


#We create a dummy varaible, equal to one if the veent is within the area around naamsestraat
Events['In_limit']=Events['Coordinates'].apply(lambda x: inbound(R_limit,Down_limit,L_limit,Up_limit,x))

Events=Events[Events['In_limit']==1].drop(['Location','In_limit','Category'],axis=1)
events_sort=Events['Sort'].unique()
events_type=Events['Type'].unique()

# Events[Events.duplicated(['id'],keep=False)].sort_values('id')

#We create a dataset with hours throughout the year 2022.
start_date = datetime.datetime(2022, 1, 1)
end_date = datetime.datetime(2022, 12, 31, 23, 59)
date_range = pd.date_range(start=start_date, end=end_date, freq='H')
Year_events=pd.DataFrame()
Year_events['Date_time']=date_range

# Year_events['Events_info']= ''
Year_events['Event_type']=''

#If a given hour lays between the start and end date of an event, such event will be added in a list which will fill the empty value of the Year_events Dataframe. 
for index, row in Year_events.iterrows(): 
    current_hour=row['Date_time']
    matching_events= Events[(current_hour>=(Events['Start'])) & (current_hour <= (Events['End']))]
    #event_info_list = matching_events['Type'].tolist()
    event_type_list=matching_events['Sort'].tolist()
    # Year_events.at[index, 'Events_info'] = event_info_list
    Year_events.at[index, 'Event_type'] = event_type_list

for sort in events_sort:
    Year_events[f'{sort}']=Year_events['Event_type'].apply(lambda x: event_type_freq(f'{sort}',x))

st.markdown("This graph shows the input values for the three different types of events. A value of 1 is assigned\
    to the specific hour per event around Naamsestraat that is in the dataset. For example, if an event is happening from \
    9am till midnight of the same day, a value of 1 given to the 15 hours from 9am till midnight. The total value \
    of the groundwork variable is the number of groundworks happening for every hour. The same holds for the other two \
    variables: events and works.")

ticker = st.selectbox(
    'Select Source',
    ['Evenement', 'Werk', 'Grondwerk'],
    index=0
)
fig1_gipod = px.histogram(Year_events, x=Year_events['Date_time'].dt.date, y=ticker, color=ticker)
fig1_gipod.update_traces(marker=dict(color='#32a846'))
fig1_gipod.update_layout(xaxis_title='Period of the year')
fig1_gipod.update_layout(showlegend=False)

st.plotly_chart(fig1_gipod)

st.markdown("---")

st.markdown("The dataset for hindrance thus looks like this and will be passed on to the model. This example \
    dataframe shows the first 100 rows of the dataset that will be used in our model.")

sample_Year_events = Year_events.head(100)

st.dataframe(sample_Year_events)

st.markdown("---")

# Create session state for the dataset so it can be passed onwards to different pages of the app
st.session_state['Year_events'] = Year_events