import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, SimpleRNN, Dense
from tensorflow.keras.layers import SimpleRNN as SRNN

sentences = [
 "I love this product",
 "This movie made me smile",
 "Service was friendly and quick",
 "Today felt bright and happy",
 "This is the best day",
 "Absolutely fantastic experience",
 "I enjoyed every single moment",
 "Great job, well done",
 "The food tasted delicious",
 "Totally recommend to everyone",
 "Very satisfied with results",
 "This worked better than expected",
 "Amazing quality and value",
 "Such a pleasant surprise",
 "I feel positive about this",
 "I hate this product",
 "This movie bored me",
 "Service was rude and slow",
 "Today was cold and lonely",
 "This is the worst day",
 "Terrible experience overall",
 "I regret buying this",
 "Very disappointed with results",
 "The food tasted awful",
 "Do not recommend this",
 "It broke after one use",
 "Not worth the money",
 "Utterly frustrating and annoying",
 "I feel negative about this",
 "Such a waste of time",
]
labels = [1]*15 + [0]*15
labels = np.array(labels)

vocab_size = 2000
tok = Tokenizer(num_words= vocab_size, oov_token = '<OOV>')
tok.fit_on_texts(sentences)
seqs = tok.texts_to_sequences(sentences)
maxlen= max(len(s) for s in seqs)

X = pad_sequences(seqs, maxlen= maxlen,  padding='post')
y= labels

print("Running well")

embed_dim = 16
rnn_units = 8

inp = Input(shape=(maxlen,), dtype = 'int32', name = 'input')
x = Embedding(input_dim = vocab_size, output_dim=embed_dim, mask_zero=True, name='embed')(inp)
rnn = SimpleRNN(units=rnn_units, name = 'simple_rnn')

x_last = rnn(x)
out = Dense(1,activation='sigmoid', name='out')(x_last)
model = Model(inputs= inp, outputs = out)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())

# Model: "functional"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                  ┃ Output Shape              ┃         Param # ┃ Connected to               ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
# │ input (InputLayer)            │ (None, 5)                 │               0 │ -                          │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ embed (Embedding)             │ (None, 5, 16)             │          32,000 │ input[0][0]                │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ not_equal (NotEqual)          │ (None, 5)                 │               0 │ input[0][0]                │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ simple_rnn (SimpleRNN)        │ (None, 8)                 │             200 │ embed[0][0],               │
# │                               │                           │                 │ not_equal[0][0]            │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ out (Dense)                   │ (None, 1)                 │               9 │ simple_rnn[0][0]           │
# └───────────────────────────────┴───────────────────────────┴─────────────────┴────────────────────────────┘
#  Total params: 32,209 (125.82 KB)
#  Trainable params: 32,209 (125.82 KB)
#  Non-trainable params: 0 (0.00 B)
# None

model.fit(X, y, epochs=30, batch_size=8, verbose=1)

intermediate_model = Model(inputs=model.inputs, outputs=[model.get_layer('embed').output, model.get_layer('simple_rnn').output])


seq_inp = Input(shape=(maxlen,), dtype='int32')
seq_emb = model.get_layer('embed')(seq_inp)  # reuse trained embedding

# Create RNN with return_sequences=True
rnn_seq = SRNN(units=rnn_units, return_sequences=True, name='rnn_seq')

# DO NOT CALL build() manually
seq_hidden = rnn_seq(seq_emb)  # builds automatically

# Copy trained RNN weights
try:
    trained_weights = model.get_layer('simple_rnn').get_weights()
    rnn_seq.set_weights(trained_weights)
    print("Copied RNN weights into sequence-inspection RNN.")
except Exception as e:
    print("Could not copy weights automatically:", e)

inspect_model = Model(inputs=seq_inp, outputs=seq_hidden)

# Inspect
idx = 0
example_seq = X[idx:idx+1]  # shape (1, maxlen)
hidden_seq = inspect_model.predict(example_seq)

print("Sentence:", sentences[idx])
print("Token ids:", example_seq)
print("Hidden states per timestep shape:", hidden_seq.shape)
print("Hidden states (timesteps x units):")
print(np.round(hidden_seq[0], 3))