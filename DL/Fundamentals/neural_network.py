import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import pandas as pd

df = pd.DataFrame({
    "soil_moisture": [0.10, 0.15, 0.20, 0.25, 0.40, 0.60, 0.35, 0.18,
                       0.45, 0.05, 0.80, 0.27, 0.55, 0.70, 0.12, 0.30],
    "temperature_c": [34, 30, 26, 22, 28, 30, 19, 22,
                       35, 24, 33, 33, 21, 25, 20, 29],
    "sunlight_hours": [9, 8, 7, 4, 8, 10, 3, 10,
                        12, 5, 9, 11, 2, 6, 1, 9],
    "needs_water": [1, 1, 1, 0, 0, 0, 0, 1,
                     0, 1, 0, 1, 0, 0, 1, 1]
})

# print(df.head())

X = df[['soil_moisture', 'temperature_c', 'sunlight_hours']]
y = df['needs_water']

X_min = X.min()
X_max = X.max()
X_scaled = (X - X_min) / (X_max - X_min)

# print(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

model = keras.Sequential([
    layers.Input(shape = (X_train.shape[1],)),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

# history = model.fit(
#     X_train.values, y_train.values,
#     validation_data = (X_test.values, y_test.values), ## validation_data is not used directly for learning, only for monitoring performance during training
#     epochs = 100, batch_size=4, verbose=1
# )

opt = optimizers.SGD(learning_rate=0.01, momentum=0.9)

model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    X_train.values, y_train.values,
    validation_data = (X_test.values, y_test.values), ## validation_data is not used directly for learning, only for monitoring performance during training
    epochs = 100, batch_size=4, verbose=1
)
