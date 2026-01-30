# 📈 FinSight-X: Risk-Aware & Explainable Stock Forecasting System

FinSight-X is an **end-to-end, production-grade machine learning system for stock price forecasting and decision support**. It goes beyond raw price prediction by integrating **risk awareness** and **explainable AI (XAI)**, transforming forecasts into actionable **BUY / HOLD / SELL** signals.

This project is designed as a industry-style ML system**, closely aligned with real-world quantitative finance pipelines.

---

## 🚀 Key Highlights

* 🔮 **Multi-Model Forecasting**

  * XGBoost (strong tabular baseline)
  * LSTM (temporal sequence learning)
  * Google TimesFM (foundation time-series Transformer)

* 🧠 **Ensemble Intelligence**

  * Weighted fusion of heterogeneous models
  * Robust predictions with reduced variance

* ⚠️ **Risk & Decision Layer**

  * Volatility-based confidence estimation
  * Prediction confidence bands
  * Automated BUY / HOLD / SELL signals

* 🔍 **Explainable AI (XAI)**

  * SHAP explanations for tree models
  * Temporal sensitivity analysis for LSTM
  * Context-window trend explanation for Transformers

* 🌐 **Deployment-Ready Architecture**

  * Modular Python package design
  * API & frontend-ready (FastAPI + Web UI)

---

## 🏗️ System Architecture

The system follows a layered design:

1. **Data Ingestion & Validation**
2. **Feature Engineering (Technical + Statistical)**
3. **Model Training (XGBoost, LSTM, Transformer)**
4. **Ensemble Forecasting**
5. **Risk & Decision Engine**
6. **Explainability Layer (XAI)**
7. **Backend API & Frontend (Phase 8)**

Architecture diagrams and figures are available in the `diagrams/` directory.

---

## 📂 Project Structure

```
FinSight-X/
├── api/                # FastAPI backend (Phase 8)
├── data/               # Raw & processed datasets
├── diagrams/           # Architecture & result figures
├── frontend/           # HTML/CSS/JS frontend
├── models/             # Trained models & TimesFM
├── notebooks/          # EDA & research notebooks
├── src/                # Core ML pipeline
│   ├── data/           # Preprocessing & validation
│   ├── features/       # Feature engineering
│   ├── models/         # Training & ensemble
│   ├── risk/           # Risk & decision logic
│   ├── explainability/ # XAI modules
│   └── inference/      # Prediction interface
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Technologies Used

### 🧠 Machine Learning & AI

* **XGBoost** – Gradient boosting for tabular data
* **PyTorch** – LSTM deep learning models
* **TimesFM (Google Research)** – Foundation time-series Transformer
* **SHAP** – Explainable AI for feature attribution

### 📊 Data & Analysis

* pandas, numpy
* scikit-learn
* statsmodels

### 🌐 Backend & Frontend

* FastAPI (Phase 8)
* HTML, CSS, JavaScript

---

## 🔧 Setup & Installation

### 1️⃣ Create Environment

```bash
conda create -n finsight-x python=3.10 -y
conda activate finsight-x
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Install TimesFM (Required)

```bash
git clone https://github.com/deephabiswashi/finsight-x.git
cd models/timesfm
pip install -e "[torch]"
cd ../..
```

---

## ▶️ How to Run the System

### 🔹 Ensemble Prediction

```bash
python -m src.models.ensemble
```

### 🔹 Risk & Decision Layer

```bash
python -m src.risk.run_decision_layer
```

### 🔹 Explainable AI (XAI)

```bash
python -m src.explainability.run_xai
```

Each step prints interpretable outputs including predictions, confidence scores, and trading signals.

---

## 📊 Sample Output

```
AAPL | Current: 247.90 | Predicted: 252.34 | Confidence: 0.81 | Signal: BUY
```

---

## 🔍 Explainability Philosophy

FinSight-X applies **model-appropriate explainability techniques**:

| Model   | Explainability Method                     |
| ------- | ----------------------------------------- |
| XGBoost | SHAP (global & local feature attribution) |
| LSTM    | Temporal sensitivity analysis             |
| TimesFM | Context-window trend visualization        |

This ensures explanations are **faithful, stable, and human-interpretable**.

---

## ⚠️ Disclaimer

This project is developed **strictly for educational and research purposes**.
It **does NOT provide financial or investment advice**.

---

## 👨‍💻 Author

**Deep Habiswashi**

---

⭐ If you find this project useful, consider starring the repository on GitHub!
