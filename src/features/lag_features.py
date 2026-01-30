import pandas as pd
from pathlib import Path


PROCESSED_DATA_PATH = Path("data/processed/processed_faang_stock_prices.csv")
OUTPUT_PATH = Path("data/processed/feature_engineered_faang.csv")


LAGS = [1, 3, 7]


def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["Ticker", "Date"])

    for lag in LAGS:
        df[f"Close_lag_{lag}"] = (
            df.groupby("Ticker")["Close"].shift(lag)
        )

        df[f"Return_lag_{lag}"] = (
            df.groupby("Ticker")["Daily_Return"].shift(lag)
        )

    return df


def main():
    print("🔹 Loading processed data...")
    df = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["Date"])

    print("🔹 Creating lag features...")
    df = create_lag_features(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"✅ Lag features added → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
