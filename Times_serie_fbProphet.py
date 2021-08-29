# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 15:23:13 2021

@author: Kered
"""

import pandas as pd
from fbprophet import Prophet


df = pd.read_csv('/new_df.csv')
#df.head()

#tracking seasonality
m = Prophet(daily_seasonality = True)
m.fit(df)

#1 month prediction
future = m.make_future_dataframe(periods=30)
future.tail()

#making prediction
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()

#ploting
fig1 = m.plot(forecast)
#ploting more component
fig2 = m.plot_components(forecast)

#1 week prediction

future_7d = m.make_future_dataframe(periods=7)
future_7d.tail()
forecast = m.predict(future_7d)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()
#ploting 
fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)

