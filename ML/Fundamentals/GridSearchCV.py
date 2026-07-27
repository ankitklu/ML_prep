import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

df = sns.load_dataset('iris')
print(df.head())

X = df.drop('species', axis=1)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_knn = KNeighborsClassifier(n_neighbors = 5)

model_knn.fit(X_train, y_train)

print(model_knn.score(X_test, y_test))

grid_search = GridSearchCV(estimator=model_knn, param_grid={'n_neighbors': [1, 3, 5, 7, 9]}, cv=5)

grid_search.fit(X_train, y_train)

print("Best parameters: ", grid_search.cv_results_)

print(pd.DataFrame(grid_search.cv_results_)[['param_n_neighbors', 'mean_test_score', 'std_test_score']])

