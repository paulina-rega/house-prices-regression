#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 22:16:37 2022

@author: paulinarega
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')



df_train = pd.read_csv('train.csv')

df_analysis = pd.read_csv('data_analysis_by_importance.csv', sep=';')

df_analysis.rename(columns={'Unnamed: 0': 'Variable'}, inplace = True)



#Code used to plot variables with expected low inpact on SalePrice:
    # expectation_high = df_analysis.loc[df_analysis[
    #     'Expectation'] == 'High']['Variable'].to_list()
    
    # for var in expectation_high:
    #     df_train.plot(x=var, y='SalePrice', kind='scatter')
    

# Code used to plot variables with expected low inpact on SalePrice:
    # expectation_medium = df_analysis.loc[df_analysis[
    #     'Expectation'] == 'Medium']['Variable'].to_list()
    
    # for var in expectation_medium:
    #     df_train.plot(x=var, y='SalePrice', kind='scatter')
    
    
# code used to plot variablkes with expected low inpact on SalePrice:
    # expectation_low = df_analysis.loc[df_analysis[
    # 'Expectation'] == 'Low']['Variable'].to_list()
    
    # for var in expectation_low:
    # df_train.plot(x=var, y='SalePrice')
    
    
# From plots, there is relavance in:
    # MSZoning
    # OverallQual
    # GrLivArea
    # YearBuild
    # ExterCond



# Plot MSZoning/SalePrice
var = 'MSZoning'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)

# Plot OverallQual/SalePrice
var = 'OverallQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);

# Plot GrLivarea/SalePrice
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

# Plot YearBuild/SalePrice
var = 'YearBuilt'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);

# Plot ExterCond/SalePrice
var = 'ExterCond'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
    


print(df_train['SalePrice'].describe())

sns.distplot(df_train['SalePrice'])

 
#skewness and kurtosis
print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())




#correlation matrix
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);



#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(
    cm, cbar=True, annot=True, square=True, fmt='.2f',
    annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()



# from correlation matrix: 
    #scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size = 2.5)
plt.show();

