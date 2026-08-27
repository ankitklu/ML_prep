import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, SimpleRNN, Dense

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


