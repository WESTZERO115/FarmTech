# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 01:45:20 2021

@author: 8305-01
"""

import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_excel('C:\\Users\\8305-01\\source\\source1\\data1.xlsx', sheet_name=0, usecols=(0,5))

x = df1.loc[:,'거래일자']
y = df1.loc[:,'마늘(kg)']

plt.plot(x,y,'-', c = 'red', label = 'variable x')