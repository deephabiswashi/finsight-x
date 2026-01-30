import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import mean_squared_error
import joblib


DATA_PATH = Path("data/processed/feature_engineered_faang.csv")
MODEL_PATH = Path("models/xgboost_model.pkl")


EXCLUDE_COLS = ["Date", "Ticker", "Next_Day_Close", "Market_Regime"]


def train_xgboost():
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df = df.dropna()

    features = [c for c in df.columns if c not in EXCLUDE_COLS]

    X = df[features]
    y = df["Next_Day_Close"]

    split_idx = int(len(df) * 0.8)

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print(f"📉 XGBoost RMSE: {rmse:.4f}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"✅ XGBoost model saved → {MODEL_PATH}")


if __name__ == "__main__":
    train_xgboost()
