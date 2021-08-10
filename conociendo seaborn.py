# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 17:55:53 2021

@author: AnaGabz
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

iris = sns.load_dataset('iris')
iris
iris.describe()



from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#cargando datos y viendo que contienen
iris = datasets.load_iris()
print(iris['DESCR'])

#scatterplot es matplotlib
plt.scatter(iris.sepal_length, iris.sepal_width)

#scatterplot en seaborn

sns.set()
plt.scatter(iris.sepal_length, iris.sepal_width)

#agregamos set_style. Probar con  darkgrid, dark, white, and ticks

sns.set_style('darkgrid')
plt.scatter(iris.sepal_length, iris.sepal_width)
plt.show()


#grafico de sepal length y width. puntos dependen de length. No me sale bien
#set_context() probar con ‘paper’, ‘notebook’ and ‘poster’
sns.set_style('darkgrid')
sns.set_context('talk', font_scale=1.1)
plt.figure(figsize=(8, 6))
sns.scatterplot(iris.sepal_length, iris.sepal_width, size="petal_length",data=iris, sizes=(20, 500))
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Sepal Length vs Sepal Width")
plt.show()


#agregamos especie al grafico

sns.set_context('talk', font_scale=1.1)
plt.figure(figsize=(8, 6))
sns.scatterplot(iris.sepal_length, iris.sepal_width, 
                size="petal_length", data=iris,
               sizes=(20, 500), hue="species", 
                alpha=0.6, palette="deep")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Sepal Length vs Sepal Width")
plt.legend(bbox_to_anchor = (1.01, 1), borderaxespad=0)
plt.show()

#relplots. line y scatter
#sin banda, ci = None
sns.relplot(iris.sepal_length, iris.sepal_width, 
                data=iris, kind='line', hue='species')
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Sepal Length vs Sepal Width")
plt.show()

#histograma
#sin curva agg. kde = False
plt.figure(figsize=(8, 6))
sns.distplot(iris.sepal_length)
plt.show()

#histograma 2
plt.figure(figsize=(8, 6))
sns.distplot(iris.sepal_length, vertical=True, kde=False, color='red')
plt.show()

#usamos otro dataset para explorar mas los histogramas
tips = sns.load_dataset("tips")
tips.head()

#histogramas por genero. no sale
g = sns.distplot(
    data=tips, x="total_bill", col="day", row="sex",
    binwidth=3, height=3, facet_kws=dict(margin_titles=True))
g.fig.set_size_inches(18, 10)
g.set_axis_labels("Total Bill", "Frequency")


g = sns.displot(
    tips, x="total_bill", col="day", row="sex",
    binwidth=3, height=3, facet_kws=dict(margin_titles=True))
g.fig.set_size_inches(18, 10)
g.set_axis_labels("Total Bill", "Frequency")

#relplot
sns.set_context('paper', font_scale=1.8)
sns.relplot('total_bill', 'tip', data=tips, hue="time", col='day', col_wrap=2)


#barplot
plt.figure(figsize=(8, 6))
sns.barplot(x='size', y= 'total_bill', hue='time', 
            palette = 'GnBu',
            data=tips, ci='sd',
           capsize=0.05,
           saturation=5,
           errcolor='lightblue',
           errwidth=2)
plt.xlabel("Size")
plt.ylabel("Total Bill")
plt.title("Total Bill Per Day of Week")
plt.show()

#countplot
plt.figure(figsize=(8, 6))
sns.countplot(x='day', data=tips)
plt.xlabel("Day")
plt.title("Total Bill Per Day of Week")
plt.show()

#agreagr variable usando hue
plt.figure(figsize=(8, 6))
sns.countplot(x = 'day', hue='time', 
            palette = 'GnBu',
            data=tips)
plt.xlabel("Day")
plt.title("Tip Per Day of Week")
plt.show()

#swarmplot (como box)
plt.figure(figsize=(8, 6))
sns.set_style('whitegrid')
sns.swarmplot(x='size', y='total_bill', data=tips)
plt.xlabel("Size")
plt.ylabel("Total Bill")
plt.title("Total bill per size of the table")
plt.show()

#agregar genero
plt.figure(figsize=(10, 6))
sns.set_style('whitegrid')
sns.set(font_scale=1.5)
sns.swarmplot(x='size', y='total_bill', data=tips, hue="sex")
plt.xlabel("Size")
plt.ylabel("Total Bill")
plt.legend(title="Sex", fontsize=14)
plt.show()

#separar
plt.figure(figsize=(10, 6))
sns.set_style('whitegrid')
sns.set(font_scale=1.5)
sns.swarmplot(x='size', y='total_bill', data=tips, hue="sex", split=True)
plt.xlabel("Size")
plt.ylabel("Total Bill")
plt.legend(title="Time", fontsize=14)
plt.show()

#swarms separados

g = sns.factorplot(x='size', y="tip",
              data=tips, hue="time",
              col="day", kind="swarm",
              col_wrap=2, size=4)
g.fig.set_size_inches(10, 10)
g.set_axis_labels("Size", "Tip")
plt.show()

#pointplot. media y lineas de confianza
plt.figure(figsize=(8, 6))
sns.pointplot(x="day", y="tip", data=tips)
plt.xlabel("Day")
plt.ylabel("Tip")
plt.title("Tip Per Day of Week")
plt.show()

#agregando hue
plt.figure(figsize=(8, 6))
sns.pointplot(x="day", y="tip", hue="sex", data=tips, palette="Accent")
plt.xlabel("Day")
plt.ylabel("Tip")
plt.title("Tip Per Day of Week by Gender")
plt.show()

#scatter con regresion

plt.figure(figsize=(8, 6))
sns.set_style('whitegrid')
sns.regplot(x='total_bill', y='tip', data=tips)
plt.xlabel("Total Bill")
plt.ylabel("Tip")
plt.show()

#jointplot (dos graficos en uno)

sns.set_style('dark')
g = sns.jointplot(x='total_bill', y='tip', hue='time', data=tips)
g.fig.set_size_inches(8, 8)
g.set_axis_labels("Total Bill", "Tip")
plt.show()

#otrojoint
sns.set_style('darkgrid')
g = sns.jointplot(x='total_bill', y='tip', data=tips, kind='reg')
g.fig.set_size_inches(8, 8)
g.set_axis_labels("Total Bill", "Tip")
plt.show()

#jointplot3. hexplot
sns.set_style('dark')
g = sns.jointplot(x='total_bill', y='tip', data=tips, kind='hex')
g.fig.set_size_inches(8, 8)
g.set_axis_labels("Total Bill", "Tip")
plt.show()

#jitterplot. parecido al swarm y agrega regresion
plt.figure(figsize=(8, 6))
sns.set_style('whitegrid')
sns.regplot(x='size', y='total_bill', data=tips, x_jitter=0.2)
plt.xlabel("Size")
plt.ylabel("Total Bill")
plt.show()

#lmplot.combinacion de regplot y facet grid
sns.set(font_scale=1.5)
sns.lmplot(x='total_bill', y='tip', data = tips, 
           hue='time')
plt.gcf().set_size_inches(12, 8)
plt.ylabel("Total Bill")
plt.xlabel("Tip")
plt.show()

#lmplot2
g = sns.lmplot(x='total_bill', y='tip', col="day", hue = "day", 
          data=tips, col_wrap=2, height=4)
g.fig.set_size_inches(11, 11)
g.set_axis_labels("Total Bill", "Tip")
plt.show()

#boxplot
sns.set(font_scale = 1.5)
sns.boxplot(x='size', y='total_bill', data=tips)
plt.gcf().set_size_inches(12, 8)
plt.xlabel("Size")
plt.ylabel("Total Bill")

#violin(distribucion)
plt.figure(figsize=(10, 7))
sns.violinplot(x='day', y='total_bill', hue="smoker",
              data=tips, palette="muted")
plt.xlabel("Day")
plt.ylabel("Total Bill")
plt.title("Total Bill per Day of the Week")
plt.show()

#violin2
plt.figure(figsize=(10, 7))
sns.violinplot(x='day', y='total_bill', hue="smoker",
              data=tips, palette="muted", split=True)
plt.xlabel("Day")
plt.ylabel("Total Bill")
plt.title("Total Bill per Day of the Week")
plt.show()

#swarm con violin
plt.figure(figsize=(10, 6))
sns.violinplot(x='day', y='total_bill', inner=None,
              data=tips, palette="muted")
sns.swarmplot(x='day', y='total_bill',
              data=tips, color="k", alpha=0.9)
plt.ylabel("Total Bill")
plt.xlabel("Day")
plt.title("Total Bill per Day")
plt.show()

#mapa de calor(correlacion)
sns.heatmap(tips[["total_bill", "tip"]].corr(), annot=True, 
            linewidths=0.9, linecolor="gray")
plt.show()

#correlacion en iris
plt.figure(figsize=(8, 6))
sns.heatmap(iris.corr(), annot=True, linewidths=0.5, cmap='Blues')
plt.show()

#facetgrid
g = sns.FacetGrid(tips, col="time")
g.map(sns.scatterplot, "total_bill", "tip")
g.fig.set_size_inches(12, 8)
g.set_axis_labels("Total Bill", "Tip")
plt.show()

#facetgrid2
g = sns.FacetGrid(tips, col="time", row="sex")
g.map(sns.scatterplot, "total_bill", "tip")
g.fig.set_size_inches(12, 12)
g.set_axis_labels("Total Bill", "Tip")
plt.show()

#pairplot
df = sns.load_dataset('iris')
sns.set_style('ticks')
sns.pairplot(df, hue="species", diag_kind='kde', kind='scatter', palette='husl')
plt.show()

#otros ir a lectura