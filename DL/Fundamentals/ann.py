import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import tensorflow as tf

from tensorflow.keras.models import Sequential # Lets you build a neural network in a linear fashion

from tensorflow.keras.layers import Dense # connects each neuron to othe rneurons of the different layer

from tensorflow.keras.layers import Dropout #Randomly drops ome neurons during training to prevent overfitting

from tensorflow.keras.utils import to_categorical # Converts class labels into one-hot encoded format

df = pd.read_csv("Iris.csv")

print("Dataset loaded successfully")

X = df.drop(columns = ['Species', 'Id'])
y = df['Species']

encoder = LabelEncoder()
y_int = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_int, test_size=0.2, random_state=42, stratify=y_int) 

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

per = Perceptron(max_iter=1000, random_state=42)
per.fit( X_train_scaled, y_train)

y_pred_percep = per.predict(X_test_scaled)
print(accuracy_score(y_test, y_pred_percep))

y_train_categorical = to_categorical(y_train, num_classes=3)
y_test_cat = to_categorical(y_test, num_classes=3)

model = Sequential([
    Dense(16, input_dim=4, activation='relu'),
    Dense(8, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    X_train_scaled, y_train_categorical,
    epochs = 100, batch_size = 8,
    validation_split = 0.2, verbose = 1
)

loss, acc = model.evaluate(X_test_scaled, y_test_cat, verbose=1)
print(acc)





