import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC


import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings('ignore')

df= sns.load_dataset('titanic')

# print(df.head())


df.drop(['deck','embark_town','alive','class','who','adult_male'], axis=1, inplace=True)

df['age'] = df['age'].fillna(df['age'].mean())
df.dropna(subset=['embarked'], inplace=True)

# print(df.info())

le = LabelEncoder() # it assign unique number to each category in a column


df['sex'] = le.fit_transform(df['sex'])
df['embarked'] = le.fit_transform(df['embarked']) # S=2, C=0, Q=1

df = df.astype(int)

# print(df.head())

X = df.drop('survived', axis=1)
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

scaler = StandardScaler() # it assign unique number to each category in a column

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.fit_transform(X_test)

model_SVM = SVC(kernel='rbf')

model_SVM.fit(X_train_scaled , y_train)

y_pred_svm = model_SVM.predict(X_test_scaled)


print(accuracy_score(y_test, y_pred_svm))

# print(confusion_matrix(y_test, y_pred))

# print(classification_report(y_test, y_pred))