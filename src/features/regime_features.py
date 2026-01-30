import pandas as pd
import numpy as np
from pathlib import Path


DATA_PATH = Path("data/processed/feature_engineered_faang.csv")


def add_market_regime(df: pd.DataFrame) -> pd.DataFrame:
    conditions = [
        (df["Close"] > df["SMA_21"]) & (df["Trend_Strength"] > 0.01),
        (df["Close"] < df["SMA_21"]) & (df["Trend_Strength"] > 0.01),
    ]

    choices = ["Bull", "Bear"]

    df["Market_Regime"] = np.select(conditions, choices, default="Sideways")

    return df


def main():
    print("🔹 Loading data...")
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])

    print("🔹 Adding market regime labels...")
    df = add_market_regime(df)

    df.to_csv(DATA_PATH, index=False)
    print("✅ Market regime feature added")


if __name__ == "__main__":
    main()
