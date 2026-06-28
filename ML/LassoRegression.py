from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error

lasso = Lasso()

# Load a sample regression dataset
housing = fetch_california_housing()
dataset = pd.DataFrame(housing.data, columns=housing.feature_names)
dataset['Price'] = housing.target

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

params = {'alpha': [1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20]}

# Train and evaluate on the training split
lasso_regressor = GridSearchCV(lasso, params, scoring='neg_mean_squared_error', cv=5)
lasso_regressor.fit(X_train, y_train)
print('Train-split best alpha:', lasso_regressor.best_params_)
print('Train-split best CV score (neg MSE):', lasso_regressor.best_score_)
train_model = lasso_regressor.best_estimator_
y_pred_train = train_model.predict(X_test)
print('Train-split Test set MSE:', mean_squared_error(y_test, y_pred_train))

# Train on the full dataset and test on X_test/y_test
full_lasso = Lasso()
full_lasso_regressor = GridSearchCV(full_lasso, params, scoring='neg_mean_squared_error', cv=5)
full_lasso_regressor.fit(X, y)
print('\nFull-data best alpha:', full_lasso_regressor.best_params_)
print('Full-data best CV score (neg MSE):', full_lasso_regressor.best_score_)
full_model = full_lasso_regressor.best_estimator_
y_pred_full = full_model.predict(X_test)
print('Full-data Test set MSE:', mean_squared_error(y_test, y_pred_full))

# Compare actual and predicted values for the test set
comparison = pd.DataFrame({
    'Actual': y_test,
    'Predicted_train_split': y_pred_train,
    'Predicted_full_data': y_pred_full,
})
print('\nActual vs Predicted values (first 10 rows):')
print(comparison.head(10))

