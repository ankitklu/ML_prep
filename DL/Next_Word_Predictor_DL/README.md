# Next Word Predictor (LSTM)

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Files
- `codefile.py` — full pipeline: loads `qoute_dataset.csv`, tokenizes, trains an RNN + LSTM model, and saves `lstm_model.h5`, `tokenizer.pkl`, `max_len.pkl`.
- `predict.py` — loads the already-trained artifacts and generates text without retraining.
- `qoute_dataset.csv` — training data (quotes).
- `lstm_model.h5`, `tokenizer.pkl`, `max_len.pkl` — pretrained artifacts.

## Run
Quick inference with the pretrained model:
```bash
python predict.py
```

Retrain from scratch (overwrites the saved artifacts):
```bash
python codefile.py
```
