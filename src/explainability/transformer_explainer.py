import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


DATA_PATH = Path("data/processed/feature_engineered_faang.csv")
SEQ_LEN = 30


def explain_transformer_context(ticker="AAPL"):
    print("🔍 Explaining Transformer context (TimesFM)...")

    df = pd.read_csv(DATA_PATH)
    data = df[df["Ticker"] == ticker].tail(SEQ_LEN)

    plt.plot(data["Date"], data["Close"], label="Close Price")
    plt.plot(data["Date"], data["SMA_21"], label="SMA_21")
    plt.plot(data["Date"], data["EMA_12"], label="EMA_12")

    plt.xticks(rotation=45)
    plt.title(f"TimesFM Context Window — {ticker}")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    explain_transformer_context("AAPL")
