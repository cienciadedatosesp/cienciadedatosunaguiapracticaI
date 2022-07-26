# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 15:52:58 2021

@author: AnaGabz
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

#Data
iris = load_iris()
print(iris.DESCR)

X = iris.data 
y = iris.target
names = iris.target_names

X = iris.data 

import pandas as pd
df = pd.DataFrame(X, columns=iris.feature_names)


#Split
train_X, test_X, train_Y, test_Y = train_test_split(X, y,test_size=0.2,random_state=0)

#grafic

from matplotlib import pyplot as  plt

fig, ax = plt.subplots()
ax.scatter(iris.data[:, 2], iris.data[:, 3], c=iris.target, label=iris.target)
ax.set(xlabel='Petal Length', ylabel='Petal Width',
       title='Iris Flowers')

ax.grid()
plt.show()


from sklearn.preprocessing import StandardScaler
# Standardizing the features
x = StandardScaler().fit_transform(X)


from sklearn.decomposition import PCA
import pandas as pd
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])



y=pd.DataFrame(y, dtype='object')

finalDf = pd.concat([principalDf, y], axis=1 )
                     
                     
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf[:,2] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

pca.explained_variance_ratio_



#Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver='lbfgs')


#Split
train_X, test_X, train_Y, test_Y = train_test_split(principalDf, y,test_size=0.2,random_state=0)


logreg.fit(train_X,train_Y)

#Scores, coeficientes, intercepto
print(logreg.score(X_train,Y_train))
print (logreg.intercept_, logreg.coef_)

#Predicciones
predict_y = logreg.predict(X_test)