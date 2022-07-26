# -*- coding: utf-8 -*-
"""
Created on Fri May 15 22:08:23 2020

@author: DELL
"""
#ejemplo de implementación de SVM
from sklearn import svm
X = [[0, 0], [1, 1]]
y = [0, 1]
clf = svm.SVC(gamma='scale')
clf.fit(X, y)

clf.predict([[2., 2.]])

# get support vectors
print(clf.support_vectors_)
# get indices of support vectors
print(clf.support_)
print(clf.n_support_)

#ejemplo 2 de SVM

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn import svm

iris = load_iris()

X = iris.data[:, :2]
y = iris.target

svc = svm.SVC(gamma='scale')
svc.fit(X, y)

def plotSVC(title):
  # create a mesh to plot in
  x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
  h = (x_max / x_min)/100
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
  
  plt.subplot(1, 1, 1)
  Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)
  
  plt.contourf(xx, yy, Z, cmap=plt.cm.Greys, alpha=0.8)
  plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.spring)
  plt.xlabel('Sepal length')
  plt.ylabel('Sepal width')
  plt.xlim(xx.min(), xx.max())
  plt.title(title)
  plt.show()
  
plotSVC(svc)

from sklearn.svm import SVC

kernels = ['linear', 'rbf', 'poly']
for kernel in kernels:
  svc = SVC(kernel=kernel).fit(X, y)
  plotSVC('kernel=' + str(kernel))
  
cs = [0.1, 1, 10, 100, 1000]
for c in cs:
    svc = SVC(kernel='rbf', C=c).fit(X, y)
    plotSVC('C=' + str(c))
    
#wrapper
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC

model = OneVsOneClassifier(SVC())
model.fit(X, y)
plotSVC(model)