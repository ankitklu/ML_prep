import string
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Dense

# ---------------------------------------------------------------------------
# Load + clean data
# ---------------------------------------------------------------------------
df = pd.read_csv("qoute_dataset.csv")
print(df.head())
print(df.shape)

quotes = df["quote"]
quotes = quotes.str.lower()

translator = str.maketrans("", "", string.punctuation)
quotes = quotes.apply(lambda x: x.translate(translator))
print(quotes.head())

# ---------------------------------------------------------------------------
# Tokenize
# ---------------------------------------------------------------------------
vocab_size = 10000

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(quotes)

word_index = tokenizer.word_index
print(len(word_index))
print(list(word_index.items())[:10])

sequence = tokenizer.texts_to_sequences(quotes)
for i in range(3):
    print(quotes.iloc[i])
for i in range(3):
    print(sequence[i])

# ---------------------------------------------------------------------------
# Build next-word prediction pairs
# ---------------------------------------------------------------------------
X = []
y = []

for seq in sequence:
    for i in range(1, len(seq)):
        input_seq = seq[:i]
        output_seq = seq[i]
        X.append(input_seq)
        y.append(output_seq)

print(len(X))
print(len(y))

max_len = max(len(x) for x in X)
print(max_len)

X_padded = pad_sequences(X, maxlen=max_len, padding="pre")
y = np.array(y)
print(X_padded.shape)
print(y.shape)

y_one_hot = to_categorical(y, num_classes=vocab_size)
print(y_one_hot.shape)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
embedding_dim = 50
rnn_units = 128

rnn_model = Sequential()
rnn_model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim))
rnn_model.add(SimpleRNN(units=rnn_units))
rnn_model.add(Dense(units=vocab_size, activation="softmax"))
rnn_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
rnn_model.build(input_shape=(None, max_len))
rnn_model.summary()

lstm_model = Sequential()
lstm_model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim))
lstm_model.add(LSTM(units=rnn_units))
lstm_model.add(Dense(units=vocab_size, activation="softmax"))
lstm_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
lstm_model.build(input_shape=(None, max_len))
lstm_model.summary()

# ---------------------------------------------------------------------------
# Train + persist the LSTM model
# ---------------------------------------------------------------------------
lstm_model.fit(X_padded, y_one_hot, epochs=5, batch_size=64, verbose=1)
lstm_model.save("lstm_model.h5")

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("max_len.pkl", "wb") as f:
    pickle.dump(max_len, f)

# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
index_to_word = {index: word for word, index in word_index.items()}


def predictor(model, tokenizer, text, max_len):
    text = text.lower()
    seq = tokenizer.texts_to_sequences([text])[0]
    seq = pad_sequences([seq], maxlen=max_len, padding="pre")

    pred = model.predict(seq, verbose=0)
    pred_index = np.argmax(pred)
    return index_to_word.get(pred_index, "")


def generate_text(model, tokenizer, seed_text, max_len, n_words):
    for _ in range(n_words):
        next_word = predictor(model, tokenizer, seed_text, max_len)
        if next_word == "":
            break
        seed_text += " " + next_word
    return seed_text


seed_text = "what are you"
next_word = predictor(lstm_model, tokenizer, seed_text, max_len)
print(next_word)

seed = "are you a "
generated = generate_text(lstm_model, tokenizer, seed, max_len, 10)
print(generated)
