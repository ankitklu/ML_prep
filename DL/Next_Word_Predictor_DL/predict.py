import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("lstm_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("max_len.pkl", "rb") as f:
    max_len = pickle.load(f)

index_to_word = {index: word for word, index in tokenizer.word_index.items()}


def predictor(text):
    text = text.lower()
    seq = tokenizer.texts_to_sequences([text])[0]
    seq = pad_sequences([seq], maxlen=max_len, padding="pre")

    pred = model.predict(seq, verbose=0)
    pred_index = pred.argmax()
    return index_to_word.get(pred_index, "")


def generate_text(seed_text, n_words):
    for _ in range(n_words):
        next_word = predictor(seed_text)
        if next_word == "":
            break
        seed_text += " " + next_word
    return seed_text


if __name__ == "__main__":
    seed = "are you a"
    print(generate_text(seed, 10))
