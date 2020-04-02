# -*- coding: utf-8 -*-
"""
Created on Tue May  7 02:23:06 2019

@author: Alfred Zane Rajan
"""

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from datetime import datetime

import matplotlib.pyplot as plt
from pylab import title,subplot

#print(datetime.datetime.now())


dataset = pd.read_csv('GLD_data.csv', header=0, parse_dates=True, index_col='date')
#test = dataset[-7:]
dataset = dataset.iloc[:,0:4].astype('float32')

scaler = MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(dataset)

# reshaping
look_back = 10

X, Y = [], []
for i in range(len(dataset)-look_back-1):
    X.append(dataset[i:(i+look_back),:])
    Y.append(dataset[i+look_back,:])
    
X = np.array(X)
Y = np.array(Y)

#%%
# LSTM model
model = Sequential()
model.add(Dense(100, input_shape=(look_back,4), activation = 'relu'))
#model.add(LSTM(20, input_shape=(look_back,5), return_sequences = True))
#model.add(Dropout(0.2))
model.add(LSTM(100))
#model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dense(4))
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X, Y, nb_epoch=20, batch_size = 8, verbose=1)

#%%
pred = model.predict(X)
   
# scale back 
pred = scaler.inverse_transform(pred)
#Y = Y.reshape(-1,1)
Y = scaler.inverse_transform(Y)

#%%    
#plotting
y_plot = np.empty_like(dataset)
y_plot[:,:] = np.nan
y_plot = pred

cols = ["Open", "High", "Low", "Close"]

     
f = plt.figure(figsize = (20,10)) 
for i in range(4):   
    subplot(2,2,i+1)
    title(cols[i])
    plt.plot(Y[:,i], color='b', lw=2.0, label='Truth')
    plt.plot(y_plot[:,i], color='g', lw=2.0, label='Predict')
    plt.legend(loc=3)

#last j
#j =7
#y_plot = np.empty_like(dataset[:j])
#y_plot[:,:] = np.nan
#y_plot = pred[:j]
#
#
#f = plt.figure(figsize = (20,10)) 
#for i in range(4):   
#    subplot(2,2,i+1)
#    title(cols[i])
#    plt.plot(Y[:j, i], color='b', lw=2.0, label='Truth')
#    plt.plot(y_plot[:,i], color='g', lw=2.0, label='Predic')
#    plt.legend(loc=3)

#%%%

#today's prediction
test = pd.read_csv('GLD.csv', header=0, parse_dates=True, index_col='Date')
test = test.iloc[-look_back:, 0:4]
test = test.astype('float32')
test = scaler.transform(test)
test = np.reshape( test, (1, look_back, 4))

today = model.predict(test)
today = scaler.inverse_transform(today)
print(today[0])
#print(Y[-1][0:4])

#%%

#test
test = pd.read_csv('GLD.csv', header=0, parse_dates=True, index_col='Date')
test = test.iloc[:, 0:4].astype('float32')
test = scaler.transform(test)

testX, testY = [], []
for i in range(len(test)-look_back-1):
    testX.append(test[i:(i+look_back),:])
    testY.append(test[i+look_back,:])
testX = np.array(testX)
testY = np.array(testY)

test_pred = model.predict(testX)
test_pred = scaler.inverse_transform(test_pred)
testY = scaler.inverse_transform(testY)

y_plot = np.empty_like(test)
y_plot[:,:] = np.nan
y_plot = test_pred

cols = ["Open", "High", "Low", "Close"]

f = plt.figure(figsize = (20,10)) 
for i in range(4):   
    subplot(2,2,i+1)
    title(cols[i])
    plt.plot(testY[:,i], color='b', lw=2.0, label='Truth')
    plt.plot(y_plot[:,i], color='g', lw=2.0, label='Predict')
    plt.legend(loc=3)