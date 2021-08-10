# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 19:21:44 2020

@author: AnaGabz
"""
from sklearn.datasets import load_boston
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sn

boston = load_boston()
X = boston.data
Y = boston.target
print(boston.DESCR)

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2)

#Exploracion de datos
t = np.arange(0.0, train_Y.shape[0], 1)

fig, ax = plt.subplots()
ax.plot(t, train_Y)

ax.set(xlabel='Observation', ylabel='Price',
       title='Housing Prices')
ax.grid()
plt.show()

#regresion
regressor = linear_model.LinearRegression()
regressor.fit(train_X, train_Y)

#fit y visualizacion
test_predictions = regressor.predict(test_X)


t = np.arange(0.0, test_Y.shape[0], 1)

fig, ax = plt.subplots()
ax.plot(t, test_Y)

ax.set(xlabel='Observation', ylabel='Price',
       title='Housing Prices')
ax.plot(t, test_predictions)
ax.grid()
plt.show()

#metricas
r2 = r2_score(test_predictions, test_Y)
print ('The R2 score of the model is {:.3f}'.format(r2))

#coeficientes
np.round(regressor.coef_, 1)
np.array(boston.feature_names)[np.argsort(np.abs(regressor.coef_))[::-1]]