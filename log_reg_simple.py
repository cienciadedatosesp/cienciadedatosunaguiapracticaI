# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 19:22:50 2020

@author: AnaGabz
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data[:,:2] #Taking only two features
y = iris.target

#Data Mining
print("Printing first 10 features set")
print(X[:10,:])
print("--------------------------")
print("Printing first 10 targets")
print(y[:10])
print(iris.DESCR)

#Cantidad por clase
unique, counts = np.unique(y, return_counts=True)
dict(zip(unique, counts))

#Scatter
plt.figure(2, figsize=(8, 6))
plt.clf()
scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k')
legend1 = plt.legend(*scatter.legend_elements(),loc = "upper right", title="Classes")
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()

#Eliminando Clase '2'
new_y = y[:100]
new_X = X[:100]

#Scatter2
plt.figure(2, figsize=(8, 6))
plt.clf()
scatter = plt.scatter(new_X[:, 0], new_X[:, 1], c=new_y,cmap=plt.cm.Set1, edgecolor='k')
legend1 = plt.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()

#train test split
X_train,X_test,Y_train,Y_test=train_test_split(new_X,new_y,test_size=0.1,random_state=0)
print(X_train.shape,X_test.shape)
print(Y_train.shape,Y_test.shape)

#Logistic Regression
logreg = LogisticRegression(solver='lbfgs')
logreg.fit(X_train,Y_train)

#Scores, coeficientes, intercepto
print(logreg.score(X_train,Y_train))
print (logreg.intercept_, logreg.coef_)

#Predicciones
predict_y = logreg.predict(X_test)

#Accuracy
print ('Accuracy from sk-learn: {0}'.format(logreg.score(X_test,Y_test)))