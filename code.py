# -*- coding: utf-8 -*-
"""
Created on Wed May  8 13:58:36 2019

@author: Alfred Zane Rajan
"""

dataset = pd.read_csv('GLD_data.csv', header=0, parse_dates=True, index_col='date')
#test = dataset[-7:]
dataset = dataset.iloc[:,0:4].astype('float32')

scaler = MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(dataset)

# reshaping
look_back = 7

X, Y = [], []
for i in range(len(dataset)-look_back-1):
    X.append(dataset[i:(i+look_back),:])
    Y.append(dataset[i+look_back,:])
    
X = np.array(X)
Y = np.array(Y)

#%%
# LSTM model
model = Sequential()
model.add(LSTM(100, input_shape=(look_back,4)))
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