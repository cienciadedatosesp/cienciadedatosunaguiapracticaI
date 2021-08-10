# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from matplotlib import pyplot as plt

periodo = [1950, 1960, 1970, 1980, 1990, 2000, 2010]
pib = [300.2, 543.3, 1075.9, 2862.5, 5979.6, 10289.7, 14958.3]

#crear grafico de linea, periodo en x y pib en y

plt.plot(periodo, pib, color='red', marker='o', linestyle='solid')

#titulo
plt.title("PIB Nominal")

