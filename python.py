# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 18:29:12 2019

@author: Richard Ademefun
"""
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')
#Find Out if there's any missing data

#Doing somethimg about those pesky Nans
#Orientation
""" If Nan is present in Orientation i will replace it
 with the average orientation """
Orient =  dataset['Orientation']
MeanOrient = np.nanmean(Orient, axis=0)
Orientation = Orient.fillna(MeanOrient)

#Dir
""" If Nan is present in Dir i will replace it
 with the average Dir """
Dir =  dataset['Dir']
MeanDir = np.nanmean(Dir, axis=0)
Direction = Dir.fillna(MeanDir)

#Humidity
Humid =  dataset['Humidity']
MeanHumid = np.nanmean(Humid, axis=0)
Humidity = Humid.fillna(MeanHumid)

##WindSpeed
#WindS =  dataset['WindSpeed']
#MeanWindS = np.nanmean(WindS, axis=0)
#WindSpeed = WindS.fillna(MeanWindS)

#Temperature
Temp =  dataset['Temperature']
MeanTemp = np.nanmean(Temp, axis=0)
Temperature = Temp.fillna(MeanTemp)


#Field Position, OffenseFormation,StadiumType, Turf, GameWeather, WindDirection

""" Replace categorical data with string blank"""
newdf3 = dataset[['FieldPosition','OffenseFormation','StadiumType','Turf',
                        'GameWeather','WindDirection']]

Nan_replacement = 'blank'
newdf3.update(newdf3.fillna(Nan_replacement))

#Combine the new dataframes
CategoricalData_3 = pd.concat([newdf3, Direction,Orientation], axis=1)

FullDataset = dataset.drop(['FieldPosition','OffenseFormation','StadiumType','Turf',
                        'GameWeather','Temperature','Humidity',
                        'WindSpeed','WindDirection','Dir','Orientation'], axis=1)

NanFreeDataSet = pd.concat([newdf3, Direction,Orientation,FullDataset], axis=1)

X = NanFreeDataSet.drop(['GameId','PlayId','NflId','Yards'], axis=1)
y = NanFreeDataSet['Yards']

#Separate the categorical data and numerical data
CategoricalData = X[['Team','DisplayName','JerseyNumber','Season','YardLine',
                         'Quarter','PossessionTeam','Down','Distance',
                         'FieldPosition',
                         'NflIdRusher','OffenseFormation','OffensePersonnel',
                         'PlayDirection','PlayerHeight','PlayerWeight',
                         'PlayerBirthDate','PlayerCollegeName','Position',
                         'HomeTeamAbbr','VisitorTeamAbbr','Week','Stadium',
                         'Location','StadiumType','Turf','GameWeather',
                         'WindDirection','Dir','Orientation','DefensePersonnel']]

NumericalData = X.drop(['Team','DisplayName','JerseyNumber','Season','YardLine',
                         'Quarter','PossessionTeam','Down','Distance',
                         'FieldPosition',
                         'NflIdRusher','OffenseFormation','OffensePersonnel',
                         'PlayDirection','PlayerHeight','PlayerWeight',
                         'PlayerBirthDate','PlayerCollegeName','Position',
                         'HomeTeamAbbr','VisitorTeamAbbr','Week','Stadium',
                         'Location','StadiumType','Turf','GameWeather',
                         'WindDirection','Dir','Orientation','DefensePersonnel'], axis=1)

#Calculating the Age of the players(closest age)
''' Changing date of birth to age '''
from datetime import date
from datetime import datetime

DateOfBirth = dataset['PlayerBirthDate']

def calculate_age(born):
    today = date.today()
    return round(today.year - born.year)

Age = [0]*len(DateOfBirth)
for i in range(0, len(DateOfBirth)):
    dt_object1 = datetime.strptime(DateOfBirth[i], "%m/%d/%Y")
    Age[i] = calculate_age(dt_object1)

#Calculating time between handof
#pip install python-dateutil

HandOf = dataset['TimeHandoff']
TimeSnap = dataset['TimeSnap']

from dateutil import parser
import datetime

throw_time = [0]*len(HandOf)
for i in range(0,len(HandOf)):
    tt = parser.parse(HandOf[i]) - parser.parse(TimeSnap[i])
    throw_time[i] = tt.seconds

#Converting time on Game clock
def get_sec(time_str):
    """Get Seconds from time."""
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)

GameC = dataset['GameClock']
GameClock = [0]*len(GameC)
for i in range(0,len(HandOf)):
    GameClock[i] = get_sec(GameC[i])

#Changing the height into just inchs
Height = dataset['PlayerHeight']
H_inches = [0]*len(Height)
for i in range(0,len(Height)):    

    H_inches[i] = int(Height[i].split('-'
            )[0]) *12 + int(''.join(Height[i].split('-')[1:]))

#Adding and dropping the Altered variable back into the dataframe
""" Numerical Data"""
NumericalData_1 = NumericalData.drop(['TimeHandoff','TimeSnap',
                    'GameClock'], axis=1)   
newdf1=pd.DataFrame(zip(throw_time,
                        GameClock),columns=['ThrowTime','GameClock'])

NumericalData_2 = pd.concat([NumericalData_1, newdf1], axis=1)

""" Categorical Data"""
CategoricalData_1 = CategoricalData.drop(['PlayerHeight',
                                          'PlayerBirthDate'], axis=1)       
newdf2=pd.DataFrame(zip(H_inches,
                        Age),columns=['PlayerHeight','PlayerBirthDate'])
CategoricalData_2 = pd.concat([CategoricalData_1, newdf2], axis=1)

from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
d = defaultdict(LabelEncoder)
from sklearn.preprocessing import LabelEncoder
fit = CategoricalData_2.apply(lambda x: d[x.name].fit_transform(x))

newCatdf = (pd.get_dummies(CategoricalData_2, drop_first=True)).values
newCatdf1 = newCatdf.iloc[:,0:100].values
#Scalling the numerical values
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
NumericalData_3 = scaler.fit_transform(NumericalData_2)

#combining the numerical and Categorical
CatVar = newCatdf.to_numpy()





















