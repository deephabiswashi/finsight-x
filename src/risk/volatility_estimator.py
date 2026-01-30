import pandas as pd
import numpy as np


def normalize_volatility(vol, vmin, vmax):
    return (vol - vmin) / (vmax - vmin + 1e-8)


def compute_risk_metrics(df, predictions):
    """
    df: feature_engineered_faang.csv
    predictions: output from ensemble_predict()
    """

    vol_min = df["Volatility_7d"].min()
    vol_max = df["Volatility_7d"].max()

    risk_output = {}

    for ticker, pred_data in predictions.items():
        latest_row = df[df["Ticker"] == ticker].sort_values("Date").iloc[-1]

        vol = latest_row["Volatility_7d"]
        norm_vol = normalize_volatility(vol, vol_min, vol_max)

        confidence = 1 / (1 + norm_vol)

        price_pred = pred_data["ensemble"]

        upper_band = price_pred * (1 + vol)
        lower_band = price_pred * (1 - vol)

        risk_output[ticker] = {
            "volatility": vol,
            "confidence": confidence,
            "upper_band": upper_band,
            "lower_band": lower_band,
        }

    return risk_output
