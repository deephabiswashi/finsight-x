import numpy as np
import pandas as pd
import torch
import joblib
from pathlib import Path
import timesfm

from .lstm_model import LSTMModel


# ========================
# Paths & Constants
# ========================
DATA_PATH = Path("data/processed/feature_engineered_faang.csv")
XGB_MODEL_PATH = Path("models/xgboost_model.pkl")
LSTM_MODEL_PATH = Path("models/lstm_model.pt")

SEQ_LEN = 30
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

XGB_WEIGHT = 0.4
LSTM_WEIGHT = 0.4
TSFM_WEIGHT = 0.2


# ========================
# Load Models
# ========================
def load_xgboost():
    return joblib.load(XGB_MODEL_PATH)


def load_lstm(input_dim):
    model = LSTMModel(input_dim)
    model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def load_timesfm():
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


# ========================
# Feature Preparation
# ========================
def get_feature_columns(df):
    return [
        c for c in df.columns
        if c not in ["Date", "Ticker", "Next_Day_Close", "Market_Regime"]
    ]


def prepare_lstm_sequence(df, features):
    sequences = {}
    targets = {}

    for ticker, group in df.groupby("Ticker"):
        g = group.tail(SEQ_LEN)
        if len(g) == SEQ_LEN:
            sequences[ticker] = g[features].values.astype(np.float32)

    return sequences


def prepare_timesfm_series(df):
    series = {}
    for ticker, group in df.groupby("Ticker"):
        g = group.tail(SEQ_LEN)
        if len(g) == SEQ_LEN:
            series[ticker] = g["Close"].values.astype(np.float32)
    return series


# ========================
# Ensemble Prediction
# ========================
def ensemble_predict():
    print("🚀 Running Ensemble Prediction...")

    df = pd.read_csv(DATA_PATH).dropna()
    features = get_feature_columns(df)

    # Load models
    xgb = load_xgboost()
    lstm = load_lstm(len(features))
    tsfm = load_timesfm()

    # Prepare inputs
    lstm_sequences = prepare_lstm_sequence(df, features)
    tsfm_series = prepare_timesfm_series(df)

    predictions = {}

    # ---- XGBoost (tabular) ----
    latest_rows = (
        df.sort_values("Date")
        .groupby("Ticker")
        .tail(1)
    )

    xgb_preds = {
        row["Ticker"]: float(
            xgb.predict(row[features].values.reshape(1, -1))[0]
        )
        for _, row in latest_rows.iterrows()
    }

    # ---- LSTM (sequence) ----
    lstm_preds = {}
    with torch.no_grad():
        for ticker, seq in lstm_sequences.items():
            x = torch.tensor(seq).unsqueeze(0).to(DEVICE)
            lstm_preds[ticker] = float(lstm(x).cpu().numpy().flatten()[0])

    # ---- TimesFM (foundation model) ----
    tickers = list(tsfm_series.keys())
    inputs = list(tsfm_series.values())

    point_forecast, _ = tsfm.forecast(
        horizon=1,
        inputs=inputs
    )

    tsfm_preds = {
        ticker: float(pred[0])
        for ticker, pred in zip(tickers, point_forecast)
    }

    # ---- Ensemble aggregation ----
    for ticker in xgb_preds.keys():
        if ticker in lstm_preds and ticker in tsfm_preds:
            final_pred = (
                XGB_WEIGHT * xgb_preds[ticker]
                + LSTM_WEIGHT * lstm_preds[ticker]
                + TSFM_WEIGHT * tsfm_preds[ticker]
            )

            predictions[ticker] = {
                "xgboost": xgb_preds[ticker],
                "lstm": lstm_preds[ticker],
                "timesfm": tsfm_preds[ticker],
                "ensemble": final_pred,
            }

    print("\n📊 Ensemble Predictions:")
    for k, v in predictions.items():
        print(
            f"{k} | XGB: {v['xgboost']:.2f} | "
            f"LSTM: {v['lstm']:.2f} | "
            f"TimesFM: {v['timesfm']:.2f} | "
            f"FINAL: {v['ensemble']:.2f}"
        )

    return predictions


if __name__ == "__main__":
    ensemble_predict()
