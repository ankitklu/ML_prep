# pip3 install scikit-learn
# pip3 install numpy
# pip3 install pandas
# pip3 install seaborn
# pip3 install matplotlib

#house pricing dataset
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing



df = fetch_california_housing()
type(df)
pd.DataFrame(df.data)

dataset = pd.DataFrame(df.data)
dataset.columns = df.feature_names
dataset.head()

dataset["Price"] = df.target
dataset.head()

# Dividing Dataset into dependant and independant features
X = dataset.iloc[:,:-1] # Independant Features
Y = dataset.iloc[:,-1] # Dependant features

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error

# Linear regression estimator
lin_reg = LinearRegression()

# Cross-validated MSE (note the correct scoring name)
cv_mse = cross_val_score(lin_reg, X, Y, scoring='neg_mean_squared_error', cv=5)
print('Cross-validated MSE (mean):', -cv_mse.mean())

# Train/test split, fit, and predict
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
lin_reg.fit(X_train, y_train)

# Use .predict() on the test set
preds = lin_reg.predict(X_test)
print('First 10 predictions:', preds[:10])
print('First 10 actuals:', y_test.values[:10])

test_mse = mean_squared_error(y_test, preds)
print('Test set MSE:', test_mse)

# Example: predict for a single sample (use a DataFrame slice to preserve feature order)
single_sample = X_test.iloc[0:1]
print('Single-sample prediction:', lin_reg.predict(single_sample))

