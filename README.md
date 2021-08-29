# Sales-volume-prediction
Problem statement
I am a saler and i want to know the amount of money the customer is willing to pay for the right price given th following attributes:
0) Number of sales of the product
1) The price
2) the title of the product
3) the ratings of the price 
4) Nulber of images 
5) Description score
6) Imges score


The main code in the repository is named:
1)sales_volume.py
In this code there is three steps for the study
1) Cleaning datas 
2) Visualisation of the most important features 
3) Builiding the model

The second code named periodic_stusy.py of consist of treating data as mainly a time series.
Approaches like simple moving average (SMA), exponential smoothing  (ES) are implemented according to the study presented  in this link: https://medium.com/dataman-in-ai/anomaly-detection-for-time-series-a87f8bc8d22e

The third code of the study named:  Times_serie_fbProphet.py try to implment a classical approach by using the Prophet. 

The Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data. Prophet is robust to missing data and shifts in the trend, and typically handles outliers well 

The forth code implement an approach using artificial neural network built with Keras named : ANN_tensorflow.py
The simple model include only one hidden layer fully connected and containing 40 neurones to start.
