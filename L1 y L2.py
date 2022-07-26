# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 15:04:58 2020

@author: AnaGabz
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('Advertising.csv')
data.head()

data.drop(['Unnamed: 0'],axis=1, inplace=True)
data.head()

def scatter_plot(feature, target):
    plt.figure(figsize=(16,8))
    plt.scatter(
        data[feature],
        data[target],
        c='black'
        )
    plt.xlabel("Money spent on {} ads ($)".format(feature))
    plt.ylabel("Sales ($k)")
    plt.show()
    
scatter_plot('TV','sales')
scatter_plot('radio','sales')
scatter_plot('newspaper','sales')

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

Xs = data.drop(['sales'], axis=1)
y = data['sales'].values.reshape(-1,1)

lin_reg = LinearRegression()

#mean square errors
MSEs = cross_val_score(lin_reg, Xs, y, scoring='neg_mean_squared_error', cv=5)
mean_MSE = np.mean(MSEs)
print(mean_MSE)

#Ridge Reg
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

alpha = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]

ridge = Ridge()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}

ridge_regressor = GridSearchCV(ridge, parameters,scoring='neg_mean_squared_error', cv=5)

ridge_regressor.fit(Xs, y)


ridge_regressor.best_params_
ridge_regressor.best_score_

#Lasso
from sklearn.linear_model import Lasso

lasso = Lasso()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}

lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv = 5)

lasso_regressor.fit(Xs, y)

lasso_regressor.best_params_
lasso_regressor.best_score_

#cual es el mejor score (el menor)

#ElasticNet
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV

alphas = np.logspace(-4, -0.5, 30)

lm_elastic= ElasticNet()
lm_elastic.fit(Xs,y)
lm_elasticcv= ElasticNetCV(alphas=alphas, random_state=0, max_iter=100)
lm_elasticcv.fit(Xs,y)



MSEs = cross_val_score(lm_elastic, Xs, y, scoring='neg_mean_squared_error', cv=5)
mean_MSE = np.mean(MSEs)
print(mean_MSE)

print(lm_elastic.coef_)

plt.figure(figsize=(15,10))
ft_importances_lm_elastic=pd.Series(lm_elastic.coef_, index=Xs.columns)
ft_importances_lm_elastic.plot(kind='barh')
plt.show()

plt.figure(figsize=(15,10))
ft_importances_lm_elasticcv=pd.Series(lm_elasticcv.coef_, index=Xs.columns)
ft_importances_lm_elasticcv.plot(kind='barh')
plt.show()