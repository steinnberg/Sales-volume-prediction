# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 15:59:55 2021

@author: Kered
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('max_columns', 100) # more than 10 columns

#Cleaning data
df= pd.read_csv('dataset_to_predict.csv')
#df.head()
#df.shape()
#df.info()

#checking na in each column
df.isna().sum()

df_fixed = df.interpolate(inplace=False)
#checking na in each column
df_fixed.isna().sum()
#droping na values from dataset
df_fixed.dropna(inplace =True)

#Important feature correlation
sns.heatmap(df.corr(), vmin=-1, cmap='YlGn')

#Statistical insight to the data
df_fixed.describe().T.round(2)

#Data visualization for the sales volume column
sns.histplot(df_fixed['sales'])

#Normalization of the values of the sale's column
df_fixed.sales.value_counts(normalize=False, sort=True, ascending=False, bins=None, dropna=True)

#Visualization of the most sold product in the dataset
order_data = df_fixed['keyword'].value_counts().iloc[:100].index
#figure
from matplotlib.pyplot import figure
figure(figsize=(15, 15), dpi=80)
sns.countplot(y='keyword', data= df_fixed, order=order_data )

#visualization by ordering sales
order_sales = df_fixed['sales'].value_counts().iloc[:100].index
figure(figsize=(15, 15), dpi=80)
sns.countplot(y='sales', data= df_fixed, order=order_sales )

#visualization of the nomber of sales by day
figure(figsize=(10, 9), dpi=80)
plt.plot(df_fixed.resample('d').size())
plt.title('Number of sales per days')
plt.xlabel('Month')
plt.ylabel('Number of sales')

#7 days rolling average
data_columns = ['sales']
data_monthly_max = df_fixed[data_columns].resample('W').max() # W stands for weekly data_monthly_max
data_7d_rol = df_fixed[data_columns].rolling(window = 7, center = True).mean()
data_7d_rol.head()

fig, ax = plt.subplots(figsize = (11,15),dpi=80)


# plotting 7-day rolling data
ax.plot(data_7d_rol['sales'], linewidth=2, label='7-d Rolling Mean')

# Nicefication of plot
ax.legend()
ax.set_xlabel('Month')
ax.set_ylabel('Sales ')
ax.set_title('Trends in Sales')

#Observation of autocorrelation of datas
# Autocorrelation is a technique for analyzing
# seasonality. It plots the correlation of the
# time series with itself at a different time lag

plt.figure(figsize=(11,4), dpi= 80)
pd.plotting.autocorrelation_plot(df_fixed.loc['2021-01-04': '2021-01-10', 'sales'])


#Time serie regression prediction with scikit-learn
df_copy= df_fixed.copy()
df_copy.head()

# to explicitly convert the date column to type DATETIME
df_copy['Date'] = pd.to_datetime(df_copy['dates'])
data = df_copy.set_index('Date')
data.head()

#Data normalization

X=data.drop(['keyword','sales'], axis=1)
y = data['sales']
X.shape
y.shape


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
y=y.values.reshape(-1,1)
y_scaled = scaler.fit_transform(y)

#verification
y_scaled.shape


#Code adapted from:
#https://towardsdatascience.com/time-series-modeling-using-scikit-pandas-and-numpy-682e3b8db8d1

#To print all performance metrics relevant to 
#a regression task (such as MAE and R-square), we will be defining the regression_results function.


import sklearn.metrics as metrics
def regression_results(y_true, y_pred):
    # Regression metrics
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)
    print('explained_variance: ', round(explained_variance,4))    
    print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))


#(T-1) model 
#a simplistic model, one that predicts today’s
#consumption value based on yesterday’s consumption 
#value and; difference between yesterday and the day 
#before yesterday’s consumption value.

# creating new dataframe from consumption column
data_consumption = data[['sales']]
# inserting new column with yesterday's consumption values
data_consumption.loc[:,'Yesterday'] = data_consumption.loc[:,'sales'].shift()
# inserting another column with difference between yesterday and day before yesterday's consumption values.
data_consumption.loc[:,'Yesterday_Diff'] = data_consumption.loc[:,'Yesterday'].diff()
# dropping NAs
data_consumption = data_consumption.dropna()

#printing values
print(data_consumption.loc['2021-02-01':])

#Defining training and test sets
X_train = data_consumption.loc['2021-01-01':'2021-01-24'].drop(['sales'], axis = 1)
y_train = data_consumption.loc['2021-01-01':'2021-01-24', 'sales']
X_test = data_consumption.loc['2021-02-01':].drop(['sales'], axis = 1)
y_test = data_consumption.loc['2021-02-01':, 'sales']
#Scaling columns of data 
scaler_data = MinMaxScaler()
X_train_scaled= scaler.fit_transform(X_train)
y=y_train.values.reshape(-1,1)
y_train_scaled = scaler.fit_transform(y)



#Testing Different alogorithm for regression over sales
#importing functions
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score


# Spot Check Algorithms
models = []
models.append(('LR', LinearRegression()))
models.append(('NN', MLPRegressor(solver = 'lbfgs')))  #neural network
models.append(('KNN', KNeighborsRegressor())) 
models.append(('RF', RandomForestRegressor(n_estimators = 10))) # Ensemble method - collection of many decision trees

# Evaluate each model in turn
results = []
names = []
for name, model in models:
    # TimeSeries Cross validation
 tscv = TimeSeriesSplit(n_splits=10)
    
 cv_results = cross_val_score(model, X_train_scaled, y_train_scaled.ravel(), cv=tscv, scoring='r2')
 results.append(cv_results)
 names.append(name)
 print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
    
# Compare Algorithms
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()

#Up until now, we have been using values at
#(t-1)th day to predict values on t date. Now,
#let us also use values from (t-2)days to predict sales:


##(t-2)days model
# creating copy of original dataframe
data_consumption_2o = data_consumption.copy()
# inserting column with yesterday-1 values
data_consumption_2o['Yesterday-1'] = data_consumption_2o['Yesterday'].shift()
# inserting column with difference in yesterday-1 and yesterday-2 values.
data_consumption_2o['Yesterday-1_Diff'] = data_consumption_2o['Yesterday-1'].diff()
# dropping NAs
data_consumption_2o = data_consumption_2o.dropna()

#Reseting test train sets
X_train_2o = data_consumption_2o.loc['2021-01-04':'2021-02-14'].drop(['sales'], axis = 1)
y_train_2o = data_consumption_2o.loc['2021-01-04':'2021-02-14', 'sales']
X_test = data_consumption_2o.loc['2021-02-14':].drop(['sales'], axis = 1)
y_test = data_consumption_2o.loc['2021-02-14':, 'sales']


#Scaling
X = data_consumption_2o.drop(['sales'], axis=1)
y = data_consumption_2o['sales']

scaler_data_2o = MinMaxScaler()
X = scaler_data_2o.fit_transform(X)
y = y.values.reshape(-1,1)
y = scaler.fit_transform(y)


#NN Machine learning perceptron algo
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

X, y = make_regression(n_samples=1000, random_state=1)

#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.66,
                                                    random_state=1)

regr = MLPRegressor(random_state=1, max_iter=5000).fit(X_train, y_train)

regr.predict(X_test[:2])
#score of the regression
regr.score(X_test, y_test)

#Simple model : Linear regression

from sklearn.linear_model import LinearRegression
#scaling
X = data_consumption_2o.drop(['sales'], axis=1)
y = data_consumption_2o['sales']

scaler_data_2o = MinMaxScaler()
X = scaler_data_2o.fit_transform(X)
y = y.values.reshape(-1,1)
y = scaler.fit_transform(y)

#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.66,
                                                    random_state=1)


reg = LinearRegression().fit(X, y)
reg.score(X, y)


reg.coef_
reg.intercept_
reg.predict(X_test)






