import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.models.lstm_model import LSTMModel


DATA_PATH = Path("data/processed/feature_engineered_faang.csv")
MODEL_PATH = Path("models/lstm_model.pt")
SEQ_LEN = 30


def explain_lstm_temporal(ticker="AAPL"):
    print("🔍 Explaining LSTM temporal importance...")

    df = pd.read_csv(DATA_PATH).dropna()

    features = [
        c for c in df.columns
        if c not in ["Date", "Ticker", "Next_Day_Close", "Market_Regime"]
    ]

    data = df[df["Ticker"] == ticker].tail(SEQ_LEN)
    X = data[features].values.astype(np.float32)

    model = LSTMModel(len(features))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    base_pred = model(torch.tensor(X).unsqueeze(0)).item()

    importances = []

    for t in range(SEQ_LEN):
        X_perturbed = X.copy()
        X_perturbed[t] = 0  # remove timestep info

        pred = model(torch.tensor(X_perturbed).unsqueeze(0)).item()
        importances.append(abs(base_pred - pred))

    plt.plot(range(-SEQ_LEN, 0), importances)
    plt.xlabel("Days Before Prediction")
    plt.ylabel("Impact on Prediction")
    plt.title(f"LSTM Temporal Importance — {ticker}")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    explain_lstm_temporal("AAPL")
