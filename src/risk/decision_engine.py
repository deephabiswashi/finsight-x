def generate_signal(
    current_price,
    predicted_price,
    volatility,
    confidence,
):
    price_change = (predicted_price - current_price) / current_price

    # Thresholds (tuned, explainable)
    LOW_VOL = 0.02
    HIGH_VOL = 0.05

    if price_change > 0.01 and volatility < LOW_VOL and confidence > 0.6:
        return "BUY"

    elif price_change < -0.01 and volatility > HIGH_VOL and confidence < 0.4:
        return "SELL"

    else:
        return "HOLD"


def decision_support(df, ensemble_predictions, risk_metrics):
    decisions = {}

    for ticker, pred_data in ensemble_predictions.items():
        latest_row = df[df["Ticker"] == ticker].sort_values("Date").iloc[-1]

        signal = generate_signal(
            current_price=latest_row["Close"],
            predicted_price=pred_data["ensemble"],
            volatility=risk_metrics[ticker]["volatility"],
            confidence=risk_metrics[ticker]["confidence"],
        )

        decisions[ticker] = {
            "current_price": latest_row["Close"],
            "predicted_price": pred_data["ensemble"],
            "confidence": risk_metrics[ticker]["confidence"],
            "volatility": risk_metrics[ticker]["volatility"],
            "upper_band": risk_metrics[ticker]["upper_band"],
            "lower_band": risk_metrics[ticker]["lower_band"],
            "signal": signal,
        }

    return decisions
