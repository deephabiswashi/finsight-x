import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


DATA_PATH = Path("data/processed/feature_engineered_faang.csv")
MODEL_PATH = Path("models/lstm_model.pt")

SEQ_LEN = 30
EPOCHS = 100
BATCH_SIZE = 64


class LSTMModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])


def create_sequences(df, features):
    X, y = [], []

    for _, group in df.groupby("Ticker"):
        data = group[features].values
        target = group["Next_Day_Close"].values

        for i in range(len(data) - SEQ_LEN):
            X.append(data[i:i+SEQ_LEN])
            y.append(target[i+SEQ_LEN])

    return np.array(X), np.array(y)


def train_lstm():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    df = pd.read_csv(DATA_PATH)
    df = df.dropna()

    features = [
        c for c in df.columns
        if c not in ["Date", "Ticker", "Next_Day_Close", "Market_Regime"]
    ]

    X, y = create_sequences(df, features)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = LSTMModel(len(features)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)

    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        preds = model(X_train).squeeze()
        loss = criterion(preds, y_train)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ LSTM model saved → {MODEL_PATH}")


if __name__ == "__main__":
    train_lstm()
