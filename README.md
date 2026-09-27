# 🌱 AgriSmart AI – Intelligent Crop Suitability & Soil Analysis System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-2e7d32.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)
[![Accuracy](https://img.shields.io/badge/accuracy-99.55%25-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **AgriSmart AI** is a precision agriculture decision-support system that combines machine learning with domain agronomic science to identify suitable crops, analyze soil chemical stoichiometry (N, P, K, pH), and evaluate microclimatic abiotic stress factors in real time.

---

## 🌾 Key Features

1. **Intelligent Crop Suitability Prediction:** Real-time multi-class classification across 22 crop species powered by a validated Random Forest ensemble (99.55% test accuracy).
2. **Soil Stoichiometry & Nutrient Diagnostics:** Automated analysis of total NPK load, N:P:K ratios, soil fertility index, and pH bioavailability risk profiling with specific amendment advisories (e.g., agricultural lime, elemental sulfur).
3. **Agro-Climatic & Abiotic Stress Assessment:** Real-time evaluation of thermal stress (heat/chilling), hydrological risk (drought/waterlogging), and foliar fungal disease vulnerabilities.
4. **Interactive What-If & Benchmark Profiles:** Empirical distribution comparison between field inputs and optimal crop benchmarks derived directly from dataset distributions.
5. **Explainable AI (XAI):** Global Gini feature attribution rankings illustrating how environmental factors dictate classification boundaries.
6. **Audit Trail & Persistent History:** Local SQLite database logging field IDs, timestamps, sensor inputs, and recommendation confidence scores with CSV export and management.
7. **Automated PDF Diagnostic Reports:** Single-click generation of professional agronomic dossiers compiled with ReportLab.

---

## 🏛️ System Architecture

```
AgriSmart AI
├── app/
│   └── app.py                     # Streamlit frontend (7 intuitive navigation pages)
├── src/
│   ├── data_processing.py        # Dataset validation, cleaning & statistical profiling
│   ├── train_model.py            # Multi-algorithm benchmark & pipeline serialization
│   ├── evaluate_model.py         # Multi-class macro & weighted evaluation metrics
│   ├── prediction.py             # Inference engine & input sanity boundary validation
│   ├── soil_analysis.py          # Soil chemical stoichiometry & pH amendment logic
│   ├── environmental_analysis.py # Agro-climatic regime & abiotic stress diagnostics
│   ├── explainability.py         # Feature attribution & benchmark z-score alignment
│   ├── database.py               # Embedded SQLite prediction logging & audit trail
│   └── report_generator.py       # Automated PDF agronomic dossier generator
├── data/
│   └── Crop_recommendation.csv   # Kaggle Crop Recommendation benchmark (2,200 rows)
├── models/
│   ├── final_model.pkl           # Champion Random Forest model artifact
│   ├── model_results.json        # Machine-readable performance metrics schema
│   ├── model_metadata.json       # Serialized training metadata & feature configurations
│   └── model_comparison.json     # Complete benchmark results across 4 algorithms
├── notebooks/
│   └── crop_suitability_analysis.ipynb # End-to-end data science and training walkthrough
├── reports/
│   ├── technical_paper.md        # Comprehensive academic research & capstone paper
│   └── presentation.md           # Formal capstone presentation slide deck outline
├── tests/
│   └── smoke_test.py             # Automated end-to-end smoke test suite
├── database/
│   └── predictions.db            # SQLite database file for audit history
├── .streamlit/
│   └── config.toml               # Agricultural UI theme configuration
├── requirements.txt              # Pinned Python package dependencies
└── README.md                     # Project documentation & quickstart guide
```

---

## 📊 Dataset & Benchmark Performance

- **Dataset:** Kaggle Crop Recommendation Benchmark
- **Sample Count:** 2,200 non-synthetic field observations
- **Target Distribution:** 22 balanced crop classes (100 samples each)
- **Features ($D=7$):** Nitrogen (N), Phosphorus (P), Potassium (K), Temperature (°C), Relative Humidity (%), Soil pH, Rainfall (mm).

### Model Benchmarking Results (5-Fold Stratified Cross-Validation)

| Algorithm | 5-Fold CV F1 (Mean) | Test Accuracy | Test Precision (Macro) | Test Recall (Macro) | Test F1 (Macro) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Champion)** | **99.49%** | **99.55%** | **99.57%** | **99.55%** | **99.55%** |
| **K-Nearest Neighbors** | 97.84% | 98.18% | 98.29% | 98.18% | 98.16% |
| **Decision Tree** | 97.78% | 97.95% | 98.10% | 97.95% | 97.94% |
| **Logistic Regression** | 97.04% | 97.27% | 97.40% | 97.27% | 97.25% |

---

## 🚀 Quickstart & Installation

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/your-username/AgriSmart-AI.git
cd "AgriSmart AI"

# Create and activate virtual environment (Windows PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# (macOS/Linux)
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Automated Verification & Smoke Tests

```bash
python tests/smoke_test.py
```

### 3. Launch the Streamlit Application

```bash
streamlit run app/app.py --server.port 8502
```

Navigate to **`http://localhost:8502`** in your browser.

---

## 🔄 Retraining & Model Serialization

To retrain and re-benchmark all 4 classification algorithms:

```bash
python src/train_model.py
```

This will automatically evaluate candidate models, select the top-performing configuration via 5-Fold Stratified Cross-Validation, and write the updated artifacts to `models/final_model.pkl` and `models/model_results.json`.

---

## 📑 Application Navigation Guide

The user-facing application features exactly 7 dedicated sections:

1. **🏠 Home:** Executive overview, system capabilities, key metrics, and supported crop catalog.
2. **🌱 Crop Analysis:** The primary interactive portal for entering soil & climate conditions (with 1-click agronomic presets), generating predictions, viewing top 5 probabilities, and exporting PDF dossiers.
3. **🧪 Soil Analysis:** Comprehensive NPK stoichiometry, nutrient status classifications, and pH chemical reaction advisories.
4. **⛅ Environmental Analysis:** Agro-climatic regime classification and abiotic stress diagnostics (thermal, moisture, fungal risk).
5. **💡 Insights:** Global feature attribution (XAI), empirical crop benchmark profile inspections, and multi-model benchmark matrices.
6. **📜 Prediction History:** Persistent SQLite history table with field ID tracking, metric aggregations, and CSV export.
7. **ℹ️ About:** System mission, mathematical methodology, and dataset specifications.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
