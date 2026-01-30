import pandas as pd
from src.models.ensemble import ensemble_predict
from src.risk.volatility_estimator import compute_risk_metrics
from src.risk.decision_engine import decision_support


DATA_PATH = "data/processed/feature_engineered_faang.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    print("🔹 Running ensemble...")
    ensemble_preds = ensemble_predict()

    print("🔹 Computing risk metrics...")
    risk_metrics = compute_risk_metrics(df, ensemble_preds)

    print("🔹 Generating trading decisions...")
    decisions = decision_support(df, ensemble_preds, risk_metrics)

    print("\n📊 FINAL DECISION SUPPORT OUTPUT\n")
    for ticker, info in decisions.items():
        print(
            f"{ticker} | "
            f"Current: {info['current_price']:.2f} | "
            f"Predicted: {info['predicted_price']:.2f} | "
            f"Confidence: {info['confidence']:.2f} | "
            f"Signal: {info['signal']}"
        )


if __name__ == "__main__":
    main()
