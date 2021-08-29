# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 14:56:52 2021

@author: Kered
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('max_columns', 100) # more than 10 columns

#Reading data from saved file
df_fixed = pd.read_csv('new_df2.csv')


dataHawk_prophet = df_fixed.resample('d').size().reset_index()
#Renaming columns
dataHawk_prophet.columns=['Date', 'Sales_Count']


dataHawk_prophet_df_final = dataHawk_prophet.rename(columns={'Date':'ds', 'Sales_Count': 'y'})

dataHawk_prophet_df_final.head()
#exporting data to csv form
dataHawk_prophet_df_final.to_csv("new_df.csv")


#Simple moving average
dataHawk_prophet_df_final['SMA'] = dataHawk_prophet_df_final.iloc[:,1].rolling(window=10).mean()
dataHawk_prophet_df_final['diff'] = dataHawk_prophet_df_final['y'] - dataHawk_prophet_df_final['SMA']
dataHawk_prophet_df_final[['y','SMA']].plot()
plt.figure()
dataHawk_prophet_df_final['diff'].hist()
plt.title('The distribution of diff')


#Printing the data 
dataHawk_prophet_df_final['upper'] = dataHawk_prophet_df_final['SMA'] + 2000
dataHawk_prophet_df_final['lower'] = dataHawk_prophet_df_final['SMA'] - 2000
dataHawk_prophet_df_final[10:20]

#Ploting the interval of prediction
def plot_it():
    plt.plot(dataHawk_prophet_df_final['y'],'go',markersize=2,label='Actual')
    plt.fill_between(
       np.arange(dataHawk_prophet_df_final.shape[0]), dataHawk_prophet_df_final['lower'], dataHawk_prophet_df_final['upper'], alpha=0.5, color="r",
       label="Predicted interval")
    plt.xlabel("Ordered samples.")
    plt.ylabel("Values and prediction intervals.")
    plt.show()
    
plot_it()


#Exponential Smoothing
from statsmodels.tsa.api import SimpleExpSmoothing

EMAfit = SimpleExpSmoothing(dataHawk_prophet_df_final['y']).fit(smoothing_level=0.2,optimized=False)
EMA = EMAfit.forecast(3).rename(r'$\alpha=0.2$')
dataHawk_prophet_df_final['EMA'] = EMAfit.predict(start = 0)
dataHawk_prophet_df_final['diff'] = dataHawk_prophet_df_final['y'] - dataHawk_prophet_df_final['EMA']
dataHawk_prophet_df_final[['y','EMA']].plot()



#Seasonal trend decomposition 

import statsmodels.api as sm
dataHawk_prophet_df_final = dataHawk_prophet_df_final.reset_index(drop='index') #inplace=True)
dataHawk_prophet_df_final.index = pd.to_datetime(dataHawk_prophet_df_final['ds'])
result = sm.tsa.seasonal_decompose(dataHawk_prophet_df_final['y'], model='additive')
result.trend[1:200].plot()

#Ploting the result of the decomposition
result.seasonal[1:100].plot()









