import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# reading data:
df_train = pd.read_csv('data_train_processed.csv')
df_test = pd.read_csv('data_test_processed.csv')

# splitting data into train/test 
train, test = train_test_split(df_train, test_size=0.3)

X_train = train.drop('SalePrice', axis=1)
X_test = test.drop('SalePrice', axis=1)
y_train = train.pop('SalePrice')
y_test = test.pop('SalePrice')

# creating linear regresssor
regression = LinearRegression()
regression.fit(X_train, y_train)

print("root of mean sqared error (all features X): ",np.sqrt(np.mean((
        y_test - regression.predict(X_test))**2)))

# predicting from test data   
X_kaggle = df_test
y_kaggle = regression.predict(X_kaggle)

# getting rid of log transformation
y_kaggle = pd.Series(np.exp(y_kaggle))

# getting data's IDs from test file
i = pd.read_csv('test.csv')
ids = pd.Series(i['Id'])

df_kaggle = pd.DataFrame({'Id': ids, 'SalePrice': y_kaggle})

# exporting predictions
df_kaggle.to_csv('prediction.csv', index=False)
