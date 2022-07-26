# -*- coding: utf-8 -*-
"""
Created on Thu May 21 18:01:16 2020

@author: DELL
"""

from statsmodels.formula.api import ols # for R-like formulas
import pandas as pd # use pandas dataframes with formulas
import statsmodels.api as sm

#data
from sklearn.datasets import load_boston
boston_data = load_boston()

boston_df = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
boston_df['target'] = boston_data.target

boston_df.head()
boston_df.describe()

#formula.api.ols hace intercepto automaticamente
#modelo1
estimator = ols(formula='target ~ RM + NOX', data=boston_df)
trained_estimator = estimator.fit()

#coeficientes
trained_estimator.params
#summary
trained_estimator.summary()
#modelo2
import numpy as np
estimator = ols(formula='target ~ np.log(TAX) + DIS', data=boston_df)
trained_estimator = estimator.fit()

#coeficientes
trained_estimator.params
#summary
trained_estimator.summary()

from pandas_profiling import ProfileReport
prof = ProfileReport(boston_df)
prof.to_file(output_file='boston_df.html')

#para todas las variables
# create a formula with all the features
formula = 'target ~ ' + ' + '.join(boston_data.feature_names)

estimator = ols(formula=formula, data=boston_df)
trained_estimator = estimator.fit()

#coeficientes
trained_estimator.params # estimated coefficients

#summary
trained_estimator.summary()
#statsmodel.api; Espera que uno cree una columna de 1s antes de correrlo
estimator = sm.OLS(endog=boston_data.target , exog=boston_data.data)
trained_estimator = estimator.fit()

#coeficientes, endogena=y; exogena=x
trained_estimator.params

#summary
trained_estimator.summary()

#agregando 1s
boston_data_with_ones = sm.add_constant(boston_data.data)
print(boston_data_with_ones[:4])

#plot, (372 alto residuo vs 380 alto leverage (quitar))
from statsmodels.graphics.regressionplots import plot_leverage_resid2

fig = plot_leverage_resid2(trained_estimator)