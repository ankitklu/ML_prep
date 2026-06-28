from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import pandas as pd

# Load a sample regression dataset
housing = fetch_california_housing()
dataset = pd.DataFrame(housing.data, columns=housing.feature_names)
dataset['Price'] = housing.target

X = dataset.iloc[:, :-1]
Y = dataset.iloc[:, -1]

ridge = Ridge()
params = {'alpha': [1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20]}

ridge_regressor = GridSearchCV(ridge, params, scoring='neg_mean_squared_error', cv=5) 
ridge_regressor.fit(X, Y)

print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)


## ridge_regressor = GridSearchCV(ridge, params, scoring='neg_mean_squared_error', cv=5) 
##                               model_name, alpha values, scoring method, cv-> Cross validation.

