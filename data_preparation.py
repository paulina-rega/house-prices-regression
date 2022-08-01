import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import missingno as msno
#import warnings
#warnings.filterwarnings('ignore')



df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')


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


sns.heatmap(df_train.corr())


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
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 
        'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size = 2.5)
plt.show();




#missing data:
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(
    ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(20))

#missing value for test set
total_test = df_test.isnull().sum().sort_values(ascending=False)
percent_test = (df_test.isnull().sum()/df_test.isnull().count()).sort_values(
    ascending=False)
missing_data_test = pd.concat([total_test, percent_test], axis=1, keys=['Total', 'Percent'])
print(missing_data_test.head(30))



#dealing with missing data:
df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
#checking that there's no missing data missing:
print(df_train.isnull().sum().max()) 





#standardizing data
saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis]);
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)





#bivariate analysis saleprice/grlivarea
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));



#deleting points
df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)


#bivariate analysis saleprice/grlivarea
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))




#histogram and normal probability plot
sns.distplot(df_train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)

#applying log transformation
df_train['SalePrice'] = np.log(df_train['SalePrice'])


#transformed histogram and normal probability plot
sns.distplot(df_train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)



#histogram and normal probability plot
sns.distplot(df_train['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)


#data transformation
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
df_test['GrLivArea'] = np.log(df_test['GrLivArea'])



#transformed histogram and normal probability plot
sns.distplot(df_train['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)





#histogram and normal probability plot
sns.distplot(df_train['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)



#create column for new variable (one is enough because it's a binary categorical feature)
#if area>0 it gets 1, for area==0 it gets 0
df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
df_train['HasBsmt'] = 0 
df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1

df_test['HasBsmt'] = pd.Series(len(df_test['TotalBsmtSF']), index=df_test.index)
df_test['HasBsmt'] = 0 
df_test.loc[df_test['TotalBsmtSF']>0,'HasBsmt'] = 1


#transform data
df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])
df_test.loc[df_test['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_test['TotalBsmtSF'])

#histogram and normal probability plot
sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)



#scatter plot
plt.scatter(df_train['GrLivArea'], df_train['SalePrice']);


#scatter plot
plt.scatter(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], df_train[df_train['TotalBsmtSF']>0]['SalePrice']);



# Preprocessing df_test:
    
#dropping columns:
# col_to_leave = df_train.columns.tolist()
# col_to_leave.remove('SalePrice')
# df_test = df_test[col_to_leave]








#choosing top 10 parameters for model
important_pred = corrmat.nlargest(5, 'SalePrice')['SalePrice'].index.tolist()
important_pred_train = important_pred.copy()
important_pred.remove('SalePrice')
df_train = df_train[important_pred_train]
df_test = df_test[important_pred]


msno.matrix(df_test)
msno.matrix(df_train)



#Garage NaN value set to 0
df_test['GarageCars'] = df_test['GarageCars'].fillna(0)
df_test['GarageArea'] = df_test['GarageCars'].fillna(0)

# TotaLBsmntSF NaNs to 0:
df_test['TotalBsmtSF'] = df_test['TotalBsmtSF'].fillna(0)


#convert categorical variable into dummy
#df_train = pd.get_dummies(df_train)


df_train.to_csv('data_train_processed.csv', index=False)
df_test.to_csv('data_test_processed.csv', index=False)



