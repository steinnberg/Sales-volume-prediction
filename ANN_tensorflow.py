# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 15:33:35 2021

@author: Kered
"""

#importing formated datas
import pandas as pd
df = pd.read_csv('/new_df2.csv')
df.head()


#Preparing data for the model
X=df.drop(['dates','dates.1','keyword','sales'], axis=1)
y = df['sales']
#checking
X.shape
y.shape

#Rescaling datas
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
y=y.values.reshape(-1,1)
y_scaled = scaler.fit_transform(y)

#verification of the right shape
X_scaled.shape
y_scaled.shape

#Training the model
from sklearn.model_selection import train_test_split
#train test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.66, random_state=1)

#Building the predictive model
import tensorflow.keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
#all connected layers
#40 initial neurones
model.add(Dense(40, input_dim=10, activation = 'relu'))
#activiation function
model.add(Dense(40, activation = 'relu'))
model.add(Dense(1, activation = 'linear'))


model.compile(optimizer='adam', loss= 'mean_squared_error')

#Evaluation of the model
epochs_hist =  model.fit(X_train, y_train, epochs=100, batch_size=75 ,verbose=1, validation_split=0.2)
epochs_hist.history.keys()

#Ploting the comparaison
import matplotlib.pyplot as plt
plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title('Model loss progress during training with 40 neurones model ')
plt.xlabel('epoch number')
plt.ylabel('Training and Validation loss')
plt.legend(['Training Loss', 'Validation Loss'])




