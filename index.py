# -*- coding: utf-8 -*-
"""
Created on Tue May 16 23:40:18 2023

@author: Alunc
"""
import plotly.graph_objects as go

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler





base_census = pd.read_csv('./database/census.csv')
base_credit = pd.read_csv('./database/credit_data.csv')


## baseCredit = base_credit.drop(base_credit[base_credit['age']<0].index)

ageMean = base_credit['age'][base_credit['age']>0].mean()
base_credit.loc[base_credit['age']<0, 'age'] = ageMean



base_credit['age'].fillna(base_credit['age'].mean(), inplace=True)



xCredit = base_credit.iloc[:, 1:4].values

yCredit = base_credit.iloc[:, 4].values

print(xCredit[:,0].min(), xCredit[:,0].max())
print(xCredit[:,1].min(), xCredit[:,1].max())
print(xCredit[:,2].min(), xCredit[:,2].max())

##normalização
scalerCredit = StandardScaler()
xCredit = scalerCredit.fit_transform(xCredit)

print(xCredit[:,0].min(), xCredit[:,0].max())
print(xCredit[:,1].min(), xCredit[:,1].max())
print(xCredit[:,2].min(), xCredit[:,2].max())


c=base_credit.iloc[:,4]
color=[]
for i in c:
    if i == 1:
        color.append('red')
    else:
        color.append('green')



yName = 'income'
xName = 'age'


x= base_credit[xName]
y =  base_credit[yName]
label = 'age'

fig, ax = plt.subplots()

ax.scatter(x, y, c=color,alpha=0.5)
ax.set_xlabel(xName)
ax.set_ylabel(yName)

plt.show()


        

df = pd.DataFrame(base_credit, columns=['income','loan','age',])
pd.plotting.scatter_matrix(df, alpha=0.2,diagonal='kde', c=color, s=50)
















