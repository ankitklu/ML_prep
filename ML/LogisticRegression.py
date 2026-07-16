from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split

# Load dataset
cancer_data = load_breast_cancer()
X = pd.DataFrame(cancer_data['data'], columns=cancer_data['feature_names'])

y = pd.DataFrame(cancer_data['target'], columns=['Target'])

X_train, y_train, X_test, y_test = train_test_split(X,y, test_size = 0.33, random_state = 42)

print(X.head())
print(y)
print(y['Target'].value_counts())

model1 = LogisticRegression()

params = [{'C': [1,5,10]}, {'max_iter' : [100,150]}]

model1= LogisticRegression(C=100, max_iter=100)

model = GridSearchCV(model1, param_grid=params,scoring='f1', cv=5)
model.fit(X_train, y_train)


# 0. Goal of this code

# The dataset contains information about breast cancer tumors.

# Each row is one patient's tumor.

# Input features (X):

# Radius
# Texture
# Perimeter
# Area
# Smoothness
# ...
# (30 numerical features)

# Output (Y):

# 0 → Malignant (Cancerous)
# 1 → Benign (Non-cancerous)

# The goal is:

# Patient's tumor details
#           ↓
#  Logistic Regression
#           ↓
# Predict:
# Cancer or No Cancer
# 1.
# from sklearn.linear_model import LogisticRegression
# What?

# Imports the Logistic Regression algorithm.

# Why?

# Because this is a classification problem, not regression.

# The output is

# 0

# or

# 1

# not

# 5.2

# 10.4

# 99.7
# Interview Question

# Q: Why can't we use Linear Regression here?

# Answer:

# Because Linear Regression predicts continuous values like 1.8 or -0.5, whereas classification requires discrete class labels (0/1).

# 2.
# from sklearn.datasets import load_breast_cancer

# Loads a built-in dataset.

# No need to download anything.

# 3.
# import pandas as pd

# Used for DataFrames.

# Makes the data easier to read.

# 4.
# from sklearn.model_selection import train_test_split

# This is one of the most important functions in ML.

# Why do we split data?

# Suppose you have 1000 examples.

# If you train and test on the same data:

# Training Accuracy = 100%

# Does that mean the model is good?

# No.

# Maybe it simply memorized the data.

# Instead:

# 1000 examples

# ↓

# Training Data

# ↓

# Testing Data

# Now the model is tested on unseen data.

# This tells us whether it generalizes.

# 5.
# cancer_data = load_breast_cancer()

# Loads the dataset.

# Internally it contains:

# data

# target

# feature_names

# description
# 6.
# X = pd.DataFrame(
#     cancer_data['data'],
#     columns=cancer_data['feature_names']
# )

# Creates the input features.

# Example:

# Radius	Texture	Perimeter	Area
# 17.99	10.38	122.8	1001

# There are 30 features.

# 7.
# y = pd.DataFrame(
#     cancer_data['target'],
#     columns=['Target']
# )

# Creates the target column.

# Example:

# Target
# 0
# 1
# 1
# 0

# Meaning:

# 0 = Malignant

# 1 = Benign
# 8.

# Your code says:

# X_train, y_train, X_test, y_test = train_test_split(...)

# This is incorrect.

# The correct order is:

# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.33,
#     random_state=42
# )
# Why?

# train_test_split() always returns:

# X_train

# X_test

# y_train

# y_test

# If you change the order:

# X_train

# y_train

# X_test

# y_test

# then

# y_train

# actually contains X_test!

# Everything becomes mismatched.

# This will either produce errors or train the model incorrectly.

# 9.
# test_size=0.33

# Means

# 33% → Testing

# 67% → Training

# If dataset has

# 600 rows

# Then

# Training = 402

# Testing = 198
# 10.
# random_state=42

# Very important.

# Suppose you shuffle cards.

# Without a fixed seed:

# Every run

# ↓

# Different split

# With

# random_state=42

# the split is always the same.

# This makes your results reproducible.

# Interview Question
# Why use random_state?

# Answer:

# To make the train-test split reproducible so that the same data is selected every time the code runs.

# 11.
# print(X.head())

# Prints first five rows.

# Useful for checking data.

# 12.
# print(y)

# Prints all targets.

# 13.
# print(y['Target'].value_counts())

# Counts each class.

# Example:

# 1    357

# 0    212

# Meaning:

# 357 benign

# 212 malignant

# Why is this useful?

# To check

# Class Balance

# If you see:

# 990

# 10

# the dataset is highly imbalanced.

# Then accuracy becomes misleading.

# 14.
# model1 = LogisticRegression()

# Creates the model.

# Nothing has been trained yet.

# 15.
# params = [
# {'C':[1,5,10]},
# {'max_iter':[100,150]}
# ]

# These are hyperparameters.

# What is C?

# Very important interview question.

# Most people memorize this incorrectly.

# C is the inverse of regularization strength.

# Large C

# ↓

# Less Regularization
# Small C

# ↓

# More Regularization

# Remember:

# Large alpha

# ↓

# More regularization

# for Ridge.

# But

# Large C

# ↓

# Less regularization

# for Logistic Regression.

# They move in opposite directions.

# What is max_iter?

# Maximum number of iterations allowed for the optimization algorithm to converge.

# If the model hasn't converged within 100 iterations, increasing max_iter gives it more time.

# 16.
# model1 = LogisticRegression(
#     C=100,
#     max_iter=100
# )

# Here you overwrite the previous model.

# Now

# Regularization is weak

# Maximum iterations = 100
# 17.
# model = GridSearchCV(
#     model1,
#     param_grid=params,
#     scoring='f1',
#     cv=5
# )

# Now GridSearchCV tries different hyperparameters.

# But there is a problem

# You passed

# C=100

# inside the model.

# Then you ask GridSearchCV to test

# C

# 1

# 5

# 10

# GridSearchCV will override the model's C with the values from param_grid, so C=100 won't actually be used in the search. If you wanted to test 100 as well, include it in the grid:

# params = {
#     'C': [1, 5, 10, 100],
#     'max_iter': [100, 150]
# }
# 18.
# scoring='f1'

# This tells GridSearchCV:

# Choose the model with the highest

# F1 Score
# Why not Accuracy?

# Imagine

# 990 Healthy

# 10 Cancer

# Model predicts

# Healthy

# Healthy

# Healthy

# Healthy

# Accuracy:

# 990 / 1000

# =

# 99%

# Looks amazing.

# But

# It never detects cancer.

# Bad model.

# F1 Score balances Precision and Recall, making it more suitable for many classification problems, especially when classes are imbalanced.

# 19.
# cv=5

# Performs 5-fold Cross Validation.

# Every combination is tested five times.

# Average F1 Score is calculated.

# Best model wins.

# 20.
# model.fit(X_train, y_train)

# This is where actual learning happens.

# Internally:

# Try C=1

# ↓

# Train

# ↓

# 5-fold CV

# ↓

# Average F1

# ↓

# Try C=5

# ↓

# Repeat

# ↓

# Choose Best
# There are two more important issues in your code
# 1. Missing import

# You use GridSearchCV, but you didn't import it.

# You need:

# from sklearn.model_selection import GridSearchCV
# 2. y shape

# You created:

# y = pd.DataFrame(...)

# Many scikit-learn estimators expect a 1-dimensional target.

# Better:

# y = cancer_data.target

# or

# y = y['Target']

# This avoids warnings and follows common practice.

# Complete Flow
# Load Breast Cancer Dataset

# ↓

# Create X

# ↓

# Create y

# ↓

# Split into Train/Test

# ↓

# Create Logistic Regression

# ↓

# Give parameter grid

# ↓

# GridSearchCV

# ↓

# 5-Fold Cross Validation

# ↓

# Find Best C

# ↓

# Find Best max_iter

# ↓

# Train Final Model
# What should you learn next?

# You're now entering Classification. I recommend this order:

# Step 1 (Very Important)
# Logistic Regression Mathematics
# Why it's called "regression" even though it performs classification
# Sigmoid function
# Log-odds (intuition)
# Decision boundary

# https://chatgpt.com/c/6a21e01c-9220-83a4-8ba7-9f4ef2cde630