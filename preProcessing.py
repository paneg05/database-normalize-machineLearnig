# -*- coding: utf-8 -*-
"""
Created on Tue May 16 23:40:18 2023

@author: Alunc
"""

import plotly.express as px
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pickle

baseCensus = pd.read_csv('./database/census.csv')
base_credit = pd.read_csv('./database/credit_data.csv')

## base de dados censo




xCensus = baseCensus.iloc[:, 0:14].values

yCensus = baseCensus.iloc[:, 14].values

##transformação em numero inteiro simples
labelEncoderWorkclass = LabelEncoder()
labelEncoderEducation = LabelEncoder()
labelEncoderMaritial = LabelEncoder()
labelEncoderOcupation = LabelEncoder()
labelEncoderRelationShip = LabelEncoder()
labelEncoderRace = LabelEncoder()
labelEncoderSex = LabelEncoder()
labelEncoderCountry = LabelEncoder()

xCensus[:,1] = labelEncoderWorkclass.fit_transform(xCensus[:,1])
xCensus[:,3] = labelEncoderEducation.fit_transform(xCensus[:,3])
xCensus[:,5] = labelEncoderMaritial.fit_transform(xCensus[:,5])
xCensus[:,6] = labelEncoderOcupation.fit_transform(xCensus[:,6])
xCensus[:,7] = labelEncoderRelationShip.fit_transform(xCensus[:,7])
xCensus[:,8] = labelEncoderRace.fit_transform(xCensus[:,8])
xCensus[:,9] = labelEncoderSex.fit_transform(xCensus[:,9])
xCensus[:,13] = labelEncoderCountry.fit_transform(xCensus[:,13])

##transformação em binario
oneHotEncoderCensus = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [1,3,5,6,7,8,9,13])], remainder='passthrough')
xCensus = oneHotEncoderCensus.fit_transform(xCensus).toarray()


##padronização
scalerCensus = StandardScaler()
xCensus = scalerCensus.fit_transform(xCensus)


##definicao das bases de dados de treinamento e de teste
xCensusTreinamento, xCensusTeste, yCensusTreinamento, yCensusTeste= train_test_split(xCensus, yCensus, test_size=0.15, random_state=0)

with open('census.pkl', mode='wb') as f:
    pickle.dump([
        xCensusTreinamento,
        xCensusTeste,
        yCensusTreinamento,
        yCensusTeste
        ], f)

##print(xCensus)

##print(yCensus)
##print(np.unique(baseCensus['income'], return_counts=True))



fig, ax1 = plt.subplots()
sns.countplot(x=baseCensus['income'], ax=ax1)


fig, ax2 = plt.subplots()
ax2.hist(x=baseCensus['age'])


fig, ax3 = plt.subplots()
ax3.hist(x=baseCensus['education-num'])

fig, ax4 = plt.subplots()
ax4.hist(x=baseCensus['hour-per-week'])

grafico = px.treemap(baseCensus, path=['workclass', 'age'])
grafico.show()

grafico = px.parallel_categories(baseCensus,dimensions=['occupation', 'relationship'])
grafico.show()



## base de dados bancarios


## baseCredit = base_credit.drop(base_credit[base_credit['age']<0].index)

ageMean = base_credit['age'][base_credit['age']>0].mean()
base_credit.loc[base_credit['age']<0, 'age'] = ageMean



base_credit['age'].fillna(base_credit['age'].mean(), inplace=True)



xCredit = base_credit.iloc[:, 1:4].values

yCredit = base_credit.iloc[:, 4].values


##normalização
scalerCredit = StandardScaler()
xCredit = scalerCredit.fit_transform(xCredit)


xCreditTreinamento, xCreditTeste, yCreditTreinamento, yCreditTeste = train_test_split(xCredit, yCredit, test_size=0.25, random_state=0)


with open('credit.pkl', mode='wb') as f:
    pickle.dump([xCreditTreinamento, xCreditTreinamento, yCreditTreinamento, yCreditTeste], f)




##print(xCreditTreinamento)
##print(yCreditTreinamento)


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
