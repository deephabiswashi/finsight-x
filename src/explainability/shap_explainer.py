import shap
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = Path("data/processed/feature_engineered_faang.csv")
XGB_MODEL_PATH = Path("models/xgboost_model.pkl")


def run_shap_explainer(ticker="AAPL"):
    print("🔍 Running SHAP Explainer (MODEL-AGNOSTIC – FINAL FIX)...")

    df = pd.read_csv(DATA_PATH).dropna()

    features = [
        c for c in df.columns
        if c not in ["Date", "Ticker", "Next_Day_Close", "Market_Regime"]
    ]

    model = joblib.load(XGB_MODEL_PATH)

    # Background dataset (small, representative)
    background = (
        df.groupby("Ticker")
        .tail(20)[features]
        .sample(50, random_state=42)
    )

    # Instance to explain
    instance = (
        df[df["Ticker"] == ticker]
        .sort_values("Date")
        .tail(1)[features]
    )

    # ✅ MODEL-AGNOSTIC SHAP (THIS NEVER TOUCHES base_score)
    explainer = shap.Explainer(
        model.predict,
        background,
        feature_names=features
    )

    shap_values = explainer(instance)

    # -------------------------
    # Global explanation
    # -------------------------
    print("📊 Global Feature Importance")
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.show()

    # -------------------------
    # Local explanation
    # -------------------------
    print("📊 Local Explanation (Latest Prediction)")
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_shap_explainer("AAPL")
