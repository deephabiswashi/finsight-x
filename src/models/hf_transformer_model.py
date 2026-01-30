import numpy as np
import pandas as pd
from pathlib import Path
import torch
import timesfm

DATA_PATH = Path("data/processed/feature_engineered_faang.csv")
SEQ_LEN = 30
HORIZON = 1


def load_timesfm():
    print("🚀 Loading TimesFM 2.5 (200M, PyTorch backend)...")

    torch.set_float32_matmul_precision("high")

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )

    model.compile(
        timesfm.ForecastConfig(
            max_context=1024,
            max_horizon=256,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
    )

    return model


def prepare_inputs(df):
    series = []
    tickers = []

    for ticker, group in df.groupby("Ticker"):
        g = group.tail(SEQ_LEN)
        if len(g) == SEQ_LEN:
            series.append(g["Close"].values.astype(np.float32))
            tickers.append(ticker)

    return tickers, series


def transformer_predict():
    df = pd.read_csv(DATA_PATH).dropna()

    model = load_timesfm()
    tickers, inputs = prepare_inputs(df)

    print("📈 Running TimesFM inference...")

    point_forecast, quantile_forecast = model.forecast(
        horizon=HORIZON,
        inputs=inputs
    )

    predictions = {
        ticker: float(pred[0])
        for ticker, pred in zip(tickers, point_forecast)
    }

    print("\n📊 TimesFM Predictions:")
    for k, v in predictions.items():
        print(f"{k}: {v:.2f}")

    return predictions


if __name__ == "__main__":
    transformer_predict()
