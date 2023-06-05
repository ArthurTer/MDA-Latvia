
import pandas as pd 
import numpy as np
import os 
import random
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import geopandas as gpd
from shapely.geometry import Point, Polygon
import calplot 
import folium
import streamlit as st
import plotly.express as px 
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import re
from tensorflow import keras 

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import  confusion_matrix

###################################################################################################################################
#                                                                                                                                 #
# Here we will store all function and their descriptions.                                                                         #
#   This file is an attempt to make our jupyter notebook cleaner by storing repetitive and cumbersome code into function.         #
#   It will also allow us to explain the purpose of the functions without making our Notebook unreadable.                         #
#                                                                                                                                 #
###################################################################################################################################


##      PREPROCESSING FUNCTIONS    ##

#1  read_data

#   read_data reads the different csv datasets given and appends them into one single datasets.
#   The functions loops into a directory and combines all the csv files by returning a pandas dataframe.

def read_data(files_dir:str,separator:str=',')-> pd.DataFrame:
    folder=glob.glob(files_dir)                                                         #Declare the folder path
    dataset_list=[]                                                                     #Initiate list to store the single dataframes

    for file in folder:                                                                 #Loop through said folder
        df=pd.read_csv(file,sep=separator)                                              #Read csv into pandas dataframe
        dataset_list.append(df)                                                         #Store dataframe in a list
    out_dataset=pd.concat(dataset_list,ignore_index=True,join='outer')                  #Concatenate all datasets into one

    return out_dataset                                                                 


###################################################################################################################################################################


#2  noisepreprocessing
    
''' As always when face with a dataset, the first stage of the analysis was to clean it. Since that process is rather unwieldy it is done within a function. 
      Both the datasets with the percentiled noise per hour and the ones containing the classified events can fit the function'''

def noiseprocessing(dataset:pd.DataFrame)->pd.DataFrame:
    """Processing the nosie data: 
    Rearranging the columns, and create a dataframe with new columns. 

    Args:
        dataset (pd.DataFrame): Name of the dataset that will be preprocessed.

    Returns:
        pd.DataFrame: Returns a pandas Dataframe without superflous and with more readable
        columns than the initial one.
        
    """
    processed_data=dataset.copy()                                                                              #Initiate the new dataframe#       

    processed_data[['result_timestamp','id']]=processed_data.result_timestamp.str.split('.',expand=True)   #The column containing the Time stamp is followed by an id#
                                                                                                           #We separate them in two different columns#
    if processed_data['id'].unique()[0]=='000':                                                            #In one of the datasets, all id's are 000#
        processed_data.drop('id',axis=1)                                                                   #If that is the case, we get  rid of the column#
        #processed_data['result_timestamp']=(processed_data['result_timestamp'].str.replace('.000','')) 
    else: 
        pass                                                                                               #Otherwise, we keep the column (for classified events)#
    
    processed_data['result_timestamp']=pd.to_datetime(
          processed_data['result_timestamp'],format='%d/%m/%Y %H:%M:%S')                                   #We format the result_timestamp into panda's datetime format#
    
    processed_data['Time']=processed_data['result_timestamp'].dt.time                                      #We retrieve the Time from the above column#
    processed_data['Date']=processed_data['result_timestamp'].dt.date                                      #As well as the Date: year,month and day#

    processed_data['description']=processed_data['description'].str.replace('-',':')                       #The description column of the initial dataset #
    processed_data['Location']=processed_data.description.str.split(':').str[1]                            #We store the Location from the description#
    processed_data=processed_data.drop(processed_data.filter(regex='unit').columns,axis=1)                 #We drop all the decibel unit columns, all units are in dB(A)#

    if 'Unnamed: 1' in processed_data.columns:                                                             #If the classified dataset is passed through the function:#

        processed_data=processed_data.drop(                                                                #The columns below are discarded#
              columns=processed_data.columns.intersection(
              ['description','#object_id', 'sep=','Unnamed: 1']))
    
        processed_data.rename(                                                                             #And those are renamed#
              columns={'noise_event_laeq_primary_detected_class':'Class',
            'noise_event_laeq_primary_detected_certainty':'Certainty',
            'noise_event_laeq_model_id':'Model_id'},inplace=True)

    else:                                                                                                  #Otherwise, we delete the two columns below#
           processed_data=processed_data.drop(
              columns=processed_data.columns.intersection(['description','#object_id']))
        

    
    processed_data['Vacation']=processed_data['result_timestamp'].apply(school_calendar)                   #We create a column 'vacation', using our built in function school calendar, see below#
    processed_data['Time_of_Day']=processed_data['result_timestamp'].dt.hour.apply(time_of_day)            #We create a column 'Time_of_Day', using our built in function school calendar, see below#
    processed_data.rename(columns={'result_timestamp':'Date_time'},inplace=True)                           #To make the future merge with other datasets the column is renamed into Date_time#
    processed_data=processed_data.dropna()                                                                 #We drop any missing values from the datasets#

    return processed_data

###################################################################################################################################################################


#3 groupdata

''' For the sake of visualisation, to analyse how the noise is distributed per day, or hours, we create a function that averages the selected percentiled noise 
over the time period chosen and across and locations. The function takes in as parameters a dataset and a column(over which to group by) and returns a new dataset. 
Grouping by Date would return a 365 rows long dataset, by time a 8760 long dataset. (Provided that every day or hour of the year has an observation, which is not the case)'''

def groupdata(origin_dataset:pd.DataFrame,group_by_column:str)->pd.DataFrame:
    """Groups the Noise dataset according to the specified column.

   _________________________________________________

    Args:
        origin_dataset (pd.DataFrame): 
        Noise Dataset.

        group_by_column (str): 'Date' | 'Hour'.
        Column that data will be grouped on. 

   _________________________________________________

    Returns:
        pd.DataFrame: Returns a dataframe grouped on the options mentionned above
    """
                                                                                     #We initiate a list where we will later append the columns of the grouped values
    
    newdataframe=pd.DataFrame()                                                                          #Initiate an empty dataframe where we will store our grouped data            

    if (group_by_column=='Date'):                                                                        #If we want to group it by hour, accross locations:

        origin_dataset[group_by_column]=origin_dataset['Date_time'].dt.date                              #We make sure to have a column upon which to group
        
        for col in (origin_dataset.filter(regex='laf').columns):                                         #For every noise measure column in the dataset 
            col_name= f"avg_{col}"                                                                       # A new column name is created
            # measures.append(f'avg_{col}')                                                                
            newdataframe[col_name]=(origin_dataset.groupby([group_by_column])[col]).mean()               #The column is added to our newdataframe, as the mean per day accross location of said percentile
            
        
    elif (group_by_column=='Hour'):                                                                      #If 

        
        newdataframe['Date_time']=origin_dataset['Date_time']
        
        for col in (origin_dataset.filter(regex='laf').columns): 
            col_name= f"avg_{col}"
            # measures.append(f'avg_{col}')
            
            newdataframe[col_name]=(origin_dataset.groupby(['Date_time'])[col]).mean() 
            
        newdataframe=newdataframe.drop_duplicates(subset=['Date_time']).set_index('Date_time')
            
    else:
        print('Grouping by ' + group_by_column + ' is not possible. Try again with Date, or Hour.')

    newdataframe=newdataframe.reset_index()
    return newdataframe

###################################################################################################################################################################


#4 Meteo-preprocessing

def meteoprocessing(origin_dataset:pd.DataFrame)->pd.DataFrame:
    """Processes the Weahther dataset to ease the readability and further use. 
        Averages weather per hour and accross all weather stations. 
    
    _________________________________________________

    Args:
        origin_dataset (pd.DataFrame): Original weather dataset

   _________________________________________________

    Returns:
        pd.DataFrame: Returns a Dataframe per hour accross all weather stations
    """
    origin_dataset['Date_time']=pd.to_datetime(origin_dataset['DATEUTC'],format='%Y-%m-%d %H:%M:%S')
    origin_dataset=origin_dataset[origin_dataset['Year']<2023]
    colist=[]
    remove=[]
    
    for col in origin_dataset.columns: 
        colist.append(col)

    remove={'DATEUTC','Date','ID','date','Year','Month','Day','Hour','Minute','LC_DWPTEMP','LC_TEMP_QCL0', 'LC_TEMP_QCL1', 'LC_TEMP_QCL2','LC_n','LC_RAD60'}
    colist=[e for e in colist if e in remove]
    origin_dataset=origin_dataset.drop(colist,axis=1)
    return origin_dataset

###################################################################################################################################################################


#5 Group meteo

def groupmeteo(origin_dataset:pd.DataFrame,group_by_column:str)->pd.DataFrame:
      
    """Groups the Meteo dataset according to the specified column.

_________________________________________________

Args:
    origin_dataset (pd.DataFrame): 
    Meteo Dataset.

    group_by_column (str): 'Date' | 'Hour'.
    Column that data will be grouped on. 

_________________________________________________

Returns:
    pd.DataFrame: Returns a dataframe grouped on the options mentionned above
"""
      
    measures=[]
    newdataframe=pd.DataFrame()
    if (group_by_column=='Date'):
       
        origin_dataset[group_by_column]=origin_dataset['Date_time'].dt.date
        
        
        for col in (origin_dataset.filter(regex='LC').columns): 
            col_name= f"avg_{col}"
            measures.append(f'avg_{col}')
            newdataframe[col_name]=(origin_dataset.groupby([group_by_column])[col]).mean()
            
        
    
    elif (group_by_column=='Hour'):
        
        origin_dataset['Date_time']=origin_dataset['Date_time'].apply(lambda a: min00(a))
        
        for col in (origin_dataset.filter(regex='LC').columns): 
            col_name= f"avg_{col}"
            measures.append(f'avg_{col}')
            newdataframe[col_name]=(origin_dataset.groupby(['Date_time'])[col]).mean()
            
        
        
        

    newdataframe=newdataframe.reset_index()  
    return newdataframe


###################################################################################################################################################################


#6   Process Dataframe to sequence

def Preprocess_to_sequence(dataframe:pd.DataFrame,columns_to_keep:list,variable_of_interest:str,window=24):
    Label=dataframe[[variable_of_interest,'Date_time']]
    (columns_to_keep.append(variable_of_interest))
    COLUMNS_TO_DROP=dataframe.columns.difference(columns_to_keep)
    Sequence_sound=dataframe.drop(COLUMNS_TO_DROP,axis=1)
   
    Sequence_sound['Hour']=Sequence_sound['Hour'].astype(str)

    
    numeric_cols=Sequence_sound[Sequence_sound.filter(regex=r'^(?!laf25_per_hour)').columns].select_dtypes(include=['float','int']).columns
    categorical_cols=Sequence_sound[Sequence_sound.filter(regex=r'^(?!Date_time)').columns].select_dtypes(include=['object']).columns

    print('With those columns as numeric variables')
    print(*numeric_cols)
    print('With those columns as categorical variables')
    print(*categorical_cols)

    Sequence_sound=Sequence_sound.drop(['laf25_per_hour','Date_time'],axis=1)

    Sequence_sound[numeric_cols]=(StandardScaler().fit_transform(Sequence_sound[numeric_cols])).astype('float32')

    dummifier=make_column_transformer((OneHotEncoder(dtype='float32',handle_unknown='infrequent_if_exist'),categorical_cols),remainder='passthrough')
    dummified=dummifier.fit_transform(Sequence_sound)
    
    feature_names = []
    for col_idx, col_name in enumerate(categorical_cols):
        categories = dummifier.transformers_[0][1].categories_[col_idx]
        feature_names.extend([f'{col_name}_{category}' for category in categories])



    feature_names.extend(Sequence_sound.columns.difference(categorical_cols))



    Sequence_sound=pd.DataFrame(dummified,columns=feature_names)


    Sequence_sound['Location']=dataframe['Location']
    Sequence_sound[['Date_time','Noise']]=Label[['Date_time','laf25_per_hour']]


    locations=Sequence_sound['Location'].unique()
    dictloc={}
    for loc in locations:
        dictloc[loc]=Sequence_sound[Sequence_sound['Location']==loc].copy()

    nhours=24
    X=[]
    y=[]

    for loc in locations:
   
        df_loc=dictloc[loc].to_numpy()
        loc_x=[]
        loc_y=[]
        for i in range(len(df_loc)-nhours): 
            row= [a for a in df_loc[i:i+nhours]]
            loc_x.append(row)
            label=(df_loc[i+nhours])
            loc_y.append(label[72:75])
        
        X.append(np.array(loc_x))
        y.append(np.array(loc_y))

    sequence_set=np.concatenate(np.array(X,dtype=object))
    Class_label=np.concatenate(np.array(y,dtype=object))

    sequence_set = np.delete(sequence_set, [72,73], axis=2)

    sequence_set=sequence_set.astype('float32')

    X_seqtrain, X_seqtest, y_seqtrain, y_seqtest= train_test_split(sequence_set,Class_label, test_size=0.35)
    X_seqtest, X_seqval, y_seqtest,y_seqval= train_test_split(X_seqtest,y_seqtest, test_size=0.15)

    return X_seqtrain, X_seqtest, y_seqtrain, y_seqtest,X_seqval,y_seqval

###################################################################################################################################################################


#7  True, False, Positive, Negative
def True_False_positive(target_list,predicitions):
    confusion=confusion_matrix(target_list,predicitions)
    FP = confusion.sum(axis=0) - np.diag(confusion)  
    FN = confusion.sum(axis=1) - np.diag(confusion)
    TP = np.diag(confusion)
    TN = confusion.sum() - (FP + FN + TP)
    return FP,FN,TP,TN





##      LAMBDA FUNCTIONS    ##


def noisecat(x): 
    if x<50: 
        return 'Soft'
    elif 50<=x<70:
        return 'Moderate'
    elif 70<=x<90:
        return 'Loud'
    else :
        return 'Very_Loud'
    
    ###################################################################################################################################################################


def school_calendar(x):
    if (pd.Timestamp('2022-01-14')<=x<=pd.Timestamp('2022-02-04') or 
        pd.Timestamp('2022-06-13')<=x<=pd.Timestamp('2022-07-02') or 
        pd.Timestamp('2022-08-22')<=x<=pd.Timestamp('2022-09-02')):
        return 'Exams'
    elif (pd.Timestamp('2022-02-14')<=x<=pd.Timestamp('2022-04-02') or 
          pd.Timestamp('2022-04-19')<=x<=pd.Timestamp('2022-05-26')):
        return "Class"
    else :
        return "Holidays"
    
    ###################################################################################################################################################################


def time_of_day(x): 
    if 0<=x<7 or x>=22: 
        return 'Night'
    elif 7<=x<19:
        return 'Day'
    else: 
        return 'Evening'

###################################################################################################################################################################


def min00(time):
    """Rounds the hour to the lower one by setting the minutes and seconds to 0

    Args:
        time datetime64[ns]: datetime instance. 

    Returns:
       datetime64[ns]: Returns rounded hour
    """
    return time.replace(second=00).replace(minute=00)

###################################################################################################################################################################


def get_coord(Location:pd.Series)->list:
    """Extracts the Lambert system coordinates stored in dataset under following format : 
    POINT (coordinate coordinate) 

_________________________________________________

    Args:
        Location (pd.Series): pandas Series, column of a dataset contain the Lambert system coordinate

_________________________________________________

    Returns:
        list: List containing the coordinates under following format: [coordinate,coordinate]
    """
    coords = re.findall('\d+', Location)
    if len(coords)==4:
        L_coord = float(str(coords[0])+'.'+str(coords[1]))
        R_coord = float(str(coords[2])+'.'+str(coords[3]))
        Coordinates = [L_coord,R_coord]
        return Coordinates
    else:
        
        L_coord=float(((Location.replace('(','').replace(')','')).split(' ',2))[1])
        R_coord=float(((Location.replace('(','').replace(')','')).split(' ',2))[2])
        Coordinates = [L_coord,R_coord]
        return Coordinates

###################################################################################################################################################################


def inbound(R_limit:float,Down_limit:float,L_limit:float,Up_limit:float,x:float)->int: 
    """Function checking if a point is within the limits of the spatial box specified in the arguments. 
 ________________________________________________
       
    Args:
        R_limit (float): The eastern limit of our box.
        Down_limit (float): The southern limit of our box.
        L_limit (float): The western limit of our box.
        Up_limit (float): The norhten limit of our box.
        x (float): Our coordinate stored as: [horizontal,vertical] or [latitude,longitude] if GCS is in place.
________________________________________________

    Returns:
        int: 1 if the point lays within the limit of the box, 0 otherwise.
    """
    if L_limit<float(x[0])<R_limit and Down_limit <float(x[1])<Up_limit :
        return 1
    else: 
        return 0

###################################################################################################################################################################


def event_type_freq(event_sort:str,x:list)->int: 
    """Counts the amount of event of a given type from a list of event. 
    
________________________________________________

    Args:
        event_sort (str): Type of event :Evenement | Werk | Grondwerk
        x (list): list of said events 
________________________________________________

    Returns:
        int: Returns the number of times that event appears in the list.
    """
    frequency={}
    for event in x: 
        if event in frequency:
            frequency[event]+=1
        else: 
            
            frequency[event]=1

    return frequency.get(event_sort)



##      PLOTS FUNCTIONS    ##


# List of longitude coordinates
longitude = [4.700566158760322,4.700487797342588, 4.700309259236343, 4.700015670473659, 4.6999715665362345, 4.699937600375688, 4.700919219878161,  4.7001912819217395]

# List of latitude coordinates
latitude = [50.877123890665274, 50.87646324085526, 50.875843609282796, 50.87450396969933, 50.874149821796564, 50.87382591784915, 50.878761158868, 50.875237268772]


def weather_sound_map (df_noise:pd.DataFrame(),df_weather:pd.DataFrame(),width:int,height:int):
    # limit_weather=(df_weather.LON.min(),df_weather.LON.max(),df_weather.LAT.min(),df_weather.LAT.max())
    # limit_sound=(df_noise.LON.min(),df_noise.LON.max(),df_noise.LAT.min(),df_noise.LAT.max())
    """Folium map plotting where weather stations and microphones are located. 

_________________________________________________

    Args:
        df_noise (pd.DataFrame): Dataframe containing the coordinates of the microphones 
        df_weather (pd.DataFrame): Dataframe containing the coordinates of the weather stations 
        width (int): width of map
        height (int): height of map

_________________________________________________

    Returns:
        _type_: Returns a folium map where the red dots represent location of microphones and the blue dots represent the weather stations' locations.
    """
    map=folium.Map(location=[50.87577038284798, 4.7007329375351385],zoom_start=13,scrollWheelZoom=False,width=width,height=height)
    list_mic=df_noise[['Loc','LAT','LON']].values.tolist()
    list_meteo=df_weather[['ID','LAT','LON']].values.tolist()
    naamsestraat= df_noise[['LAT','LON']].sort_values(by=['LAT']).values.tolist()
    for i in list_mic:
        folium.CircleMarker(radius=3,fill=True,fill_color='red',location=[i[1],i[2]],popup=i[0],color='red').add_to(map)
    for i in list_meteo:
        folium.CircleMarker(radius=3,fill=True,fill_color='blue',location=[i[1],i[2]],popup=i[0],color='blue').add_to(map)
    return map

###################################################################################################################################################################

def double_lineplot(dataset:pd.DataFrame,var1:str,var2:str,time_axis:str,xlabel:str,y1_label:str,y2_label:str):
    """Generates a double axed smoothed out lined plot over a timed axis. 

_________________________________________________

    Args:
        dataset (pd.DataFrame): Dataset from which to extract the columns displayed 
        var1 (str): First variable to be plotted on the left y-axis
        var2 (str): Second variable to be plotted on the right y-axis
        time_axis (str): x-axis time variable 
        xlabel (str): x-axis label
        y1_label (str): left axis label
        y2_label (str): right axis label
    """
    fig, ax1= plt.subplots(figsize=(10,6))
    ax2= ax1.twinx()

    # Apply a rolling mean with a window size of 7 days to the sound data
    dataset['Smoothed Sound'] = dataset[var1].rolling(window=7).mean()

    # Apply a rolling mean with a window size of 7 days to the temperature data
    dataset['Smoothed Temperature'] = dataset[var2].rolling(window=7).mean()

    sns.lineplot(data=dataset, x=time_axis,y='Smoothed Sound',ax=ax1,color='red')
    sns.lineplot(data=dataset, x=time_axis,y='Smoothed Temperature',ax=ax2,color='blue')

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(y1_label,color='red')
    ax2.set_ylabel(y2_label,color='blue')

    months = mdates.MonthLocator()
    month_fmt = mdates.DateFormatter('%b')
    ax1.xaxis.set_major_locator(months)
    ax1.xaxis.set_major_formatter(month_fmt)

    plt.show()

###################################################################################################################################################################


def mapoutliers (dataset:pd.DataFrame,y_col:str,x_col:str='Date_time'):
    """Generates a plot of the selected column's outliers over time
 ________________________________________________
    Args:
        dataset (pd.DataFrame): Dataset to be used.
        y_col (str): Column from which outliers will be plotted
        x_col (str, optional): Time column to be used as x-axis. Defaults to 'Date_time'.
  ________________________________________________
    
    """
    Q1=(dataset[f'{y_col}'].quantile(0.25))
    Q3=(dataset[f'{y_col}'].quantile(0.75))
    IQR=Q3-Q1
    upper_lim=Q3+1.5*IQR
    print((dataset[f'{y_col}']>upper_lim).sum())
    dataset['outliers']= (dataset[f'{y_col}']>upper_lim)
    outliersplot=dataset.loc[dataset['outliers']==True,dataset.columns]
    outliersscat=px.scatter(outliersplot,x='Date_time',y=f'{y_col}',color='Location',hover_data=['Time','Date'],color_discrete_sequence=px.colors.qualitative.T10)
    outliersscat.update_xaxes(
    tickformat="%B")
    outliersscat.update_layout(xaxis_title='Day', yaxis_title='Value in decibels')
     
    return outliersscat

###################################################################################################################################################################


def plot_predictions(dataframe:pd.DataFrame,n_obs=100): 
    
    df=dataframe.sample(n=n_obs).sort_values('Date_time')

    scatter_true=go.Scatter(
    x=df['Date_time'],
    y=df['True_values'],
    mode='markers',
    marker_color='#636EFA',
    name='True_values'
    )

    scatter_pred=go.Scatter(
        x=df['Date_time'],
        y=df['Predictions'],
        mode='markers',
        marker_color='#EF553B',
        name='Predictions'
    )


    layout = go.Layout(
    title='True values against Predictions over time',
    xaxis=dict(title='Date'),
    yaxis=dict(title='15 minute noise level')
    )   

    predi_plot = go.Figure(data=[scatter_true, scatter_pred], layout=layout)
    return predi_plot

###################################################################################################################################################################


def residuals_by(df:pd.DataFrame,group_by_column:str):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        group_by_column (str, optional): _description_. 'Date'|'Day'|'Density_Day'|'Hour'|'Location'|'Density_Location'
    """
    measures=[]
    newdf=pd.DataFrame()

    if group_by_column=='Date':
        

        df[group_by_column]=pd.to_datetime(df['Date_time'])
        for col in df.select_dtypes(include=['float','int']).columns:
            col_name= f"avg_{col}"
            measures.append(f'avg_{col}')
            newdf[col_name]=(df.groupby([group_by_column])[col]).mean()
        
        newdf=newdf.reset_index()

        fig3= px.scatter(newdf,x='Date',y=(newdf['avg_Residuals']),trendline='rolling',
                         color=newdf['avg_Residuals'],trendline_options=dict(window=50,),trendline_color_override=px.colors.qualitative.Set1[5],color_continuous_scale=px.colors.diverging.Picnic)


    elif group_by_column=='Day':
        df[group_by_column]=df['Date_time'].apply(lambda x: x.strftime('%A'))
        df['sort']=df['Date_time'].dt.strftime('%w').astype(int)
        df=df.sort_values('sort')
        fig3=px.histogram(df,x=group_by_column,y=df['Residuals'],color=group_by_column,color_discrete_sequence=px.colors.qualitative.Prism, histfunc='avg')


    elif group_by_column=='Density_Day':
        
        df[group_by_column]=df['Date_time'].apply(lambda x: x.strftime('%A'))
        day=df[group_by_column].unique()

        pivoted=df.pivot(columns=group_by_column,values='Residuals')

        hist_data=[pivoted[column].dropna().tolist() for column in pivoted[day]]

        colours = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880']
       
        fig3=ff.create_distplot(hist_data, group_labels=day, show_hist=False,bin_size=10, colors=colours)
        fig3.update_layout(xaxis=dict(range=[-6, 6]))


    elif group_by_column=='Hour':
        df[group_by_column]=(df['Date_time'].apply(lambda x: x.strftime('%H'))).astype(str)
        fig3=px.histogram(df,x=group_by_column,y=df['Residuals'],color=group_by_column,color_discrete_sequence=px.colors.qualitative.Prism, histfunc='avg')
    
    elif group_by_column=='Location':
        fig3=px.histogram(df,x='Location',y=df['Residuals'],color='Location',color_discrete_sequence=px.colors.qualitative.Prism, histfunc='avg')

    elif group_by_column=='Density_Location':
        
        loca=df['Location'].unique()

        pivoted=df.pivot(columns='Location',values='Residuals')

        hist_data=[pivoted[column].dropna().tolist() for column in pivoted[loca]]
       

      
        colours = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880']
       
        fig3=ff.create_distplot(hist_data, group_labels=loca, show_hist=False,bin_size=10, colors=colours)
        fig3.update_layout(xaxis=dict(range=[-6, 6]))
    return fig3

###################################################################################################################################################################


def Validation_Loss_Accuracy(Dictionary:keras.callbacks.History,epochs:int,width:int=1200,height:int=400):
    train_loss = Dictionary.history['loss']
    val_loss = Dictionary.history['val_loss']
    train_acc = Dictionary.history['mse']
    val_acc = Dictionary.history['val_mse']

    epochs = list(range(1, (len(Dictionary.history['val_mse'])+1)))
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Training and Validation Loss, MSLE", "Training and Validation Accuracy,MSE"))

    fig.add_trace(go.Scatter(x=epochs, y=train_loss, mode='lines', name='Training Loss', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=val_loss, mode='lines', name='Validation Loss', line=dict(color='red')), row=1, col=1)

    fig.add_trace(go.Scatter(x=epochs, y=train_acc, mode='lines', name='Training Accuracy, MSE', line=dict(color='blue')), row=1, col=2)
    fig.add_trace(go.Scatter(x=epochs, y=val_acc, mode='lines', name='Validation Accuracy, MSE', line=dict(color='red')), row=1, col=2)

    fig.update_xaxes(title_text='Epochs', row=1, col=1)
    fig.update_xaxes(title_text='Epochs', row=1, col=2)
    fig.update_yaxes(title_text='Loss', row=1, col=1)
    fig.update_yaxes(title_text='Accuracy', row=1, col=2)

    fig.update_layout(width=width, height=height,showlegend=True, legend=dict(x=0.5, y=1.2), margin=dict(t=50, b=50))

    return fig

###################################################################################################################################################################


def plot_class_proba(df:pd.DataFrame,target_list,prediction_list,probability_list):
    class_df=pd.DataFrame()
    class_df['True_class']=target_list
    class_df['Predictions']=prediction_list
    class_df['Proba']=probability_list[:,1]

    mergeddf= class_df.merge(df, left_index=True, right_index=True,how='inner')

    mergeddf['Outcome'] = np.where(mergeddf['True_class'] == mergeddf['Predictions'], 'Correct', 'Wrong')
    Class_fig=px.scatter(mergeddf,x='laf10_per_hour',y='Proba',color='Outcome',color_discrete_sequence=px.colors.qualitative.T10,hover_data=['Nuisance','Location','Date_time'])
    Class_fig.show()