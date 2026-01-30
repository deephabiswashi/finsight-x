from src.explainability.shap_explainer import run_shap_explainer
from src.explainability.lstm_explainer import explain_lstm_temporal
from src.explainability.transformer_explainer import explain_transformer_context


def run_all_xai(ticker="AAPL"):
    run_shap_explainer(ticker)
    explain_lstm_temporal(ticker)
    explain_transformer_context(ticker)


if __name__ == "__main__":
    run_all_xai("AAPL")
