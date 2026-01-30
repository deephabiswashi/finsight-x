import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path


RAW_DATA_PATH = Path("data/raw/faang_stock_prices.csv")
PROCESSED_DATA_PATH = Path("data/processed/processed_faang_stock_prices.csv")


NUMERIC_FEATURES = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA_7", "SMA_21", "EMA_12", "EMA_26",
    "RSI_14", "MACD", "MACD_Signal",
    "Bollinger_Upper", "Bollinger_Lower",
    "Daily_Return", "Volatility_7d"
]


def load_data():
    df = pd.read_csv(RAW_DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def sort_and_forward_fill(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort by Ticker + Date and forward-fill missing values per ticker.
    """
    df = df.sort_values(["Ticker", "Date"])

    df[NUMERIC_FEATURES] = (
        df.groupby("Ticker")[NUMERIC_FEATURES]
        .apply(lambda x: x.ffill())
        .reset_index(level=0, drop=True)
    )

    return df


def rolling_zscore(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Apply rolling z-score normalization per ticker.
    """
    for col in NUMERIC_FEATURES:
        rolling_mean = (
            df.groupby("Ticker")[col]
            .transform(lambda x: x.rolling(window).mean())
        )
        rolling_std = (
            df.groupby("Ticker")[col]
            .transform(lambda x: x.rolling(window).std())
        )

        df[f"{col}_rollnorm"] = (df[col] - rolling_mean) / (rolling_std + 1e-8)

    return df


def scale_per_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize features per ticker to avoid cross-stock leakage.
    """
    scaled_frames = []

    for ticker, group in df.groupby("Ticker"):
        scaler = StandardScaler()
        group_scaled = group.copy()

        group_scaled[NUMERIC_FEATURES] = scaler.fit_transform(
            group[NUMERIC_FEATURES]
        )

        scaled_frames.append(group_scaled)

    return pd.concat(scaled_frames).sort_values(["Ticker", "Date"])


def remove_data_leakage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure Next_Day_Close is not part of feature matrix.
    """
    if "Next_Day_Close" in df.columns:
        df = df.dropna(subset=["Next_Day_Close"])

    return df


def main():
    print("🔹 Loading raw data...")
    df = load_data()

    print("🔹 Sorting and forward-filling missing values...")
    df = sort_and_forward_fill(df)

    print("🔹 Removing data leakage...")
    df = remove_data_leakage(df)

    print("🔹 Applying rolling window normalization...")
    df = rolling_zscore(df)

    print("🔹 Scaling features per ticker...")
    df = scale_per_ticker(df)

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"✅ Preprocessed data saved to: {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    main()
