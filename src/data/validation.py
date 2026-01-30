import pandas as pd
from statsmodels.tsa.stattools import adfuller
from pathlib import Path


PROCESSED_DATA_PATH = Path("data/processed/processed_faang_stock_prices.csv")


def adf_test(series, ticker):
    """
    Perform Augmented Dickey-Fuller test.
    """
    result = adfuller(series.dropna(), autolag="AIC")

    return {
        "Ticker": ticker,
        "ADF Statistic": result[0],
        "p-value": result[1],
        "Stationary": result[1] < 0.05
    }


def run_stationarity_tests(df: pd.DataFrame):
    results = []

    for ticker, group in df.groupby("Ticker"):
        res = adf_test(group["Close"], ticker)
        results.append(res)

    return pd.DataFrame(results)


def main():
    print("🔹 Loading processed data...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    print("🔹 Running stationarity tests (ADF)...")
    results_df = run_stationarity_tests(df)

    print("\n📊 Stationarity Results:")
    print(results_df)

    non_stationary = results_df[~results_df["Stationary"]]
    if not non_stationary.empty:
        print("\n⚠️ Non-stationary tickers detected:")
        print(non_stationary["Ticker"].tolist())
    else:
        print("\n✅ All series are stationary.")


if __name__ == "__main__":
    main()
