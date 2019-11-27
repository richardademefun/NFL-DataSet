
        
        
"""
Created on Tue Nov 19 18:29:12 2019

@author: Richard Ademefun
"""
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
#
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


# Importing the dataset
#dataset = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv')
dataset = pd.read_csv('train.csv',low_memory=False)

#Find Out if there's any missing data

#Doing somethimg about those pesky Nans
#Orientation
""" If Nan is present in Orientation i will replace it
 with 0 category """
dataset.loc[(dataset.Orientation >= 0) & (dataset.Orientation <= 45), 'Orientation'] = 1
dataset.loc[(dataset.Orientation > 315) & (dataset.Orientation <= 360), 'Orientation'] = 1
dataset.loc[(dataset.Orientation > 225) & (dataset.Orientation <= 315), 'Orientation'] = 4
dataset.loc[(dataset.Orientation > 135) & (dataset.Orientation <= 225), 'Orientation'] = 3
dataset.loc[(dataset.Orientation > 45) & (dataset.Orientation <= 135), 'Orientation'] = 2
dataset['Orientation'].fillna(0, inplace=True) # if colum is nan give value of 0
# Now orientation is in 5 categories

#Dir
""" If Nan is present in Dir i will replace it
 with 0 """

dataset.loc[(dataset.Dir >= 0) & (dataset.Dir <= 45), 'Dir'] = 1
dataset.loc[(dataset.Dir > 315) & (dataset.Dir <= 360), 'Dir'] = 1
dataset.loc[(dataset.Dir > 225) & (dataset.Dir <= 315), 'Dir'] = 4
dataset.loc[(dataset.Dir > 135) & (dataset.Dir <= 225), 'Dir'] = 3
dataset.loc[(dataset.Dir > 45) & (dataset.Dir <= 135), 'Dir'] = 2
dataset['Dir'].fillna(0, inplace=True) # if colum is nan give value of 0

#Humidity
dataset['Humidity']
MeanHum = np.nanmean(dataset['Humidity'], axis=0)
dataset['Humidity'].fillna(MeanHum, inplace=True) # if colum is nan give value of 0


# Weather
dataset['GameWeather'].str.lower()
dataset['GameWeather'].apply(lambda x: "indoor" if not pd.isna(x) and "indoor" in x else x)
dataset['GameWeather'].apply(lambda x: x.replace('coudy', 'cloudy').replace('clouidy', 'cloudy').replace('party', 'partly') if not pd.isna(x) else x)
dataset['GameWeather'].apply(lambda x: x.replace('clear and sunny', 'sunny and clear') if not pd.isna(x) else x)
dataset['GameWeather'].apply(lambda x: x.replace('skies', '').replace("mostly", "").strip() if not pd.isna(x) else x)

#Temperature
dataset['Temperature']
MeanTemp = np.nanmean(dataset['Temperature'], axis=0)
dataset['Temperature'].fillna(MeanTemp, inplace=True) # if colum is nan give value of 0

#Defenders in the box
dataset['DefendersInTheBox']
MeanDef = np.nanmean(dataset['DefendersInTheBox'], axis=0)
dataset['DefendersInTheBox'].fillna(MeanDef, inplace=True) # if colum is nan give value of 0

#WindSpeed
dataset['WindSpeed'] = pd.to_numeric(dataset.WindSpeed, errors='coerce')
MeanWind = dataset['WindSpeed'].mean()
dataset['WindSpeed'].fillna(MeanWind, inplace=True) # if colum is nan give value of 0#Field Position, OffenseFormation,StadiumType, Turf, GameWeather, WindDirection

#Wind Direction
dataset['WindDirection'].fillna('unknown', inplace=True) # if colum is nan give value of 0#Field Position, OffenseFormation,StadiumType, Turf, GameWeather, WindDirection

#Game Weather
dataset['GameWeather'].fillna('unknown', inplace=True) # if colum is nan give value of 0#Field Position, OffenseFormation,StadiumType, Turf, GameWeather, WindDirection

#Stadium type
dataset['StadiumType'].fillna('unknown', inplace=True) # if colum is nan give value of 0#Field Position, OffenseFormation,StadiumType, Turf, GameWeather, WindDirection

#Field Position
dataset['FieldPosition'].fillna('unknown', inplace=True) # if colum is nan give value of 0#Field Position, OffenseFormation,StadiumType, Turf, GameWeather, WindDirection

#Offense Formation
dataset['OffenseFormation'].fillna('unknown', inplace=True) # if colum is nan give value of 0#Field Position, OffenseFormation,StadiumType, Turf, GameWeather, WindDirection

X = dataset.drop(['GameId','PlayId','NflId'], axis=1)

#Separate the categorical data and numerical data
CategoricalData = X[['Team','DisplayName','JerseyNumber','Season',
                         'Quarter','PossessionTeam','Down','FieldPosition',
                         'NflIdRusher','OffenseFormation','OffensePersonnel',
                         'PlayDirection','PlayerCollegeName','Position',
                         'HomeTeamAbbr','VisitorTeamAbbr','Stadium','Down',
                         'Location','StadiumType','Turf','GameWeather',
                         'WindDirection','Dir','Orientation',
                         'DefensePersonnel']]

NumericalData = X.drop(['Team','DisplayName','JerseyNumber','Season',
                         'Quarter','PossessionTeam','Down','FieldPosition',
                         'NflIdRusher','OffenseFormation','OffensePersonnel',
                         'PlayDirection','PlayerCollegeName','Position',
                         'HomeTeamAbbr','VisitorTeamAbbr','Stadium','Down',
                         'Location','StadiumType','Turf','GameWeather',
                         'WindDirection','Dir','Orientation',
                         'DefensePersonnel'], axis=1)

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
                    'GameClock','PlayerHeight','PlayerBirthDate'], axis=1)   
newdf1=pd.DataFrame(zip(throw_time,
                        GameClock,H_inches,Age),columns=['ThrowTime',
    'GameClock','PlayerHeight','PlayerBirthDate'])

NumericalData_2 = pd.concat([NumericalData_1, newdf1], axis=1)

""" Encoding Categorical Data"""
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
d = defaultdict(LabelEncoder)
from sklearn.preprocessing import LabelEncoder
CategoricalData = (CategoricalData.apply(
        lambda x: d[x.name].fit_transform(x)))
CategoricalData = CategoricalData.to_numpy()
#Scalling the numerical values

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
y = NumericalData_2['Yards']
NumericalData_22 = NumericalData_2.drop(['Yards'],axis = 1)
NumericalData_3 = scaler.fit_transform(NumericalData_22)
#combining the numerical and Categorical
FullData = np.concatenate((CategoricalData,NumericalData_3),axis=1)

#rain test split
import numpy as np
from sklearn.model_selection import train_test_split

X = FullData
y = y.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

# Function to check model accuracy
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

# Visualising the Random Forest Regression results (higher resolution)

plt.scatter(y_test, y_pred, color = 'red')
plt.plot(y_pred, regressor.predict(y_pred), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


thresholds = np.arange(70)

def heavyside(actual):
    return thresholds >= actual

def is_cdf_valid(case):
    if case[0] < 0 or case[0] > 1:
        return False
    for i in range(1, len(case)):
        if case[i] > 1 or case[i] < case[i-1]:
            return False
    return True

def calc_crps(predictions, actuals):
    #some vector algebra for speed
    obscdf = np.array([heavyside(i) for i in actuals])
    crps = np.mean(np.mean((predictions - obscdf) ** 2))
    return crps

def CRPS(predictions, actuals):
    
    check = True
    '''
    for p in predictions :
        if is_cdf_valid(p) == False : 
            print 'something wrong with your prediction'
            check = False
            break
    '''
    if check == True : 
         return calc_crps(predictions, actuals)
         
###example of usage
predictions = np.ones((X_test.shape[0],70))
print(CRPS(predictions, y_test))












