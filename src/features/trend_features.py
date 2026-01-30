import pandas as pd
from pathlib import Path


DATA_PATH = Path("data/processed/feature_engineered_faang.csv")


def add_trend_strength(df: pd.DataFrame) -> pd.DataFrame:
    df["Trend_Strength"] = (
        (df["SMA_7"] - df["SMA_21"]).abs() / (df["Close"] + 1e-8)
    )

    return df


def main():
    print("🔹 Loading feature-engineered data...")
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])

    print("🔹 Adding trend strength feature...")
    df = add_trend_strength(df)

    df.to_csv(DATA_PATH, index=False)
    print("✅ Trend strength feature added")


if __name__ == "__main__":
    main()
