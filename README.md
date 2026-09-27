# 🌾 AgriSmart AI – Intelligent Crop Suitability & Soil Analysis System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Frontend-Streamlit-red.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Build-Passing-brightgreen.svg)]()

> **Academic Project Title:** *An Explainable Machine Learning System for Crop Suitability Classification and Soil Analysis*  
> **Target:** 10-Day ML Internship Final Capstone Project  
> **Authoritative Dataset:** `Crop_recommendation.csv` (2,200 empirical agricultural observations across 22 crop species)

---

## 📖 Executive Summary

**AgriSmart AI** is a decision-support and agronomic diagnostic system engineered to optimize agricultural yield and minimize fertilizer misapplication. By analyzing soil chemical properties (Nitrogen, Phosphorus, Potassium, Soil pH) and meteorological parameters (Temperature, Relative Humidity, Rainfall), the system predicts the most suitable crop with probabilistic confidence scoring.

### 🌟 Key Technical Highlights
- **100% Authoritative Data:** Exclusively trained on 2,200 empirical field observations without synthetic data fabrication.
- **Strict Data Leakage Prevention:** Preprocessing transformers (`StandardScaler`) are encapsulated in `Pipeline` objects and fitted solely on training splits within Stratified 5-Fold Cross-Validation.
- **Explainable AI (XAI):** Transparent feature attribution using Global Gini Impurity Reduction and Local Empirical Crop Niche Benchmarking.
- **Multi-Model Benchmarking:** Comprehensive comparison across 8 configurations (Logistic Regression, KNN, Decision Tree, Random Forest across 2 feature sets).
- **Automated PDF Dossier Generation:** Publication-grade diagnostic reports generated using ReportLab.
- **Audit & Persistence:** Local SQLite database tracking all historical predictions and user queries.

---

## 📊 Experimental Results & Model Benchmarking

Models were evaluated using an **80/20 Stratified Split** with **5-Fold Stratified Cross-Validation**:

| Model Configuration | Feature Set | Test Accuracy | Weighted F1 | Macro F1 | 5-Fold CV F1 (Mean ± Std) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **Original (7 Features)** | **99.55%** | **99.55%** | **99.55%** | **99.49% (± 0.42%)** | **Selected Champion** |
| Random Forest | Engineered (13 Features) | 99.09% | 99.09% | 99.09% | 99.37% (± 0.65%) | Benchmark |
| K-Nearest Neighbors | Original (7 Features) | 97.95% | 97.96% | 97.95% | 97.67% (± 0.70%) | Benchmark |
| Logistic Regression | Original (7 Features) | 96.82% | 96.81% | 96.81% | 95.80% (± 0.73%) | Benchmark |
| K-Nearest Neighbors | Engineered (13 Features) | 96.59% | 96.58% | 96.58% | 96.31% (± 0.61%) | Benchmark |
| Logistic Regression | Engineered (13 Features) | 96.14% | 96.12% | 96.11% | 95.40% (± 0.96%) | Benchmark |
| Decision Tree | Original (7 Features) | 95.23% | 95.34% | 95.32% | 95.51% (± 0.66%) | Benchmark |
| Decision Tree | Engineered (13 Features) | 94.77% | 94.88% | 94.86% | 95.23% (± 0.82%) | Benchmark |

### 🔬 Feature Engineering Analysis
- **Original 7 Features:** Nitrogen ($N$), Phosphorus ($P$), Potassium ($K$), Temperature, Relative Humidity, Soil pH, and Rainfall.
- **Engineered Features:** $\text{NPK\_total}$, $\text{NPK\_average}$, $N/P$, $N/K$, $P/K$, and $\text{Temperature} \times \text{Humidity}$.
- **Finding:** The original physical dimensions provide orthogonal decision boundaries. Derived collinear ratios did not add mutual information, confirming that **parsimonious Random Forest on the original 7 features** is the optimal production model.

---

## 🏛️ Project Directory Structure

```text
AgriSmart AI/
├── app/
│   └── app.py                             # 10-Page Interactive Streamlit Web Dashboard
├── data/
│   ├── Crop_recommendation.csv            # Authoritative Original Dataset (2,200 rows)
│   └── Crop_recommendation_engineered.csv # Derived Stoichiometric Dataset (14 columns)
├── database/
│   └── predictions.db                     # Local SQLite DB for Prediction Logs & Auditing
├── models/
│   ├── final_model.pkl                    # Production Random Forest Model Pipeline
│   ├── preprocessing_pipeline.pkl         # Production Preprocessing Transformer
│   ├── model_comparison.json              # Full 8-Model Cross-Validation Benchmarks
│   └── model_metadata.json                # Schema, Class Labels & Metadata
├── notebooks/
│   └── crop_suitability_analysis.ipynb    # 16-Section Academic Research Notebook
├── reports/
│   └── sample_report.pdf                  # Automated ReportLab PDF Crop Suitability Dossier
├── src/
│   ├── __init__.py
│   ├── data_processing.py                 # Ingestion, Schema Checks, IQR Outliers, Summary Stats
│   ├── feature_engineering.py             # Zero-Safe Stoichiometric & Climate Interactions
│   ├── evaluate_model.py                  # Multiclass Evaluation (Acc, Prec, Rec, F1, ROC-AUC)
│   ├── train_model.py                     # Stratified 5-Fold CV & Dynamic Winning Model Selector
│   ├── prediction.py                      # Multi-Class Inference, Probability & What-If Simulator
│   ├── soil_analysis.py                   # NPK Stoichiometry, Percentile Scoring & pH Agronomics
│   ├── environmental_analysis.py          # Climate Regimes, Microclimate Envelopes & Risk Assessment
│   ├── explainability.py                  # Global Feature Importances & Target Crop Benchmarks
│   ├── database.py                        # SQLite CRUD, Query Filters & CSV Exporters
│   ├── report_generator.py                # Publication-Grade PDF Diagnostic Generator
│   ├── test_pipeline.py                   # 8-Stage End-to-End System Test Suite
│   └── safety_audit_test.py               # Exhaustive 12-Stage Safety & Robustness Auditor
├── .gitignore                             # Clean VCS hygiene
├── requirements.txt                       # Production Pinning of all Dependencies
└── README.md                              # Comprehensive Academic & Technical Documentation
```

---

## ⚡ Quickstart & Installation

### 1. Clone & Set Up Environment
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/AgriSmart-AI.git
cd AgriSmart-AI

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train & Evaluate Models
```bash
python src/train_model.py
```

### 3. Run Automated Safety & Verification Test Suite
```bash
# Run 12-Stage Safety Audit
python src/safety_audit_test.py

# Run 8-Stage Integration Test Pipeline
python src/test_pipeline.py
```

### 4. Launch the Web Application
```bash
streamlit run app/app.py
```

---

## 💻 Streamlit Web Application Pages

1. **🏠 Executive Overview:** High-level metrics, agricultural KPIs, pipeline summary, and architecture diagram.
2. **📈 Exploratory Data Analysis (EDA):** Interactive Plotly distributions, class balance charts, correlation heatmaps, pairwise bivariate scatter plots, and IQR outlier analysis.
3. **🌱 Crop Suitability Prediction:** Real-time multi-input form, agronomic range validation, Top-5 probability distribution bar charts, and one-click PDF report downloads.
4. **🧪 Soil Nutrient Diagnostic:** Radar chart stoichiometry ($N:P:K$), empirical dataset quartile scoring, and soil pH classification (from strongly acidic to strongly alkaline).
5. **⛅ Environmental Analysis:** Microclimate envelope matching, thermal and moisture stress checks, and climate regime tagging.
6. **💡 Explainable AI (XAI):** Global Gini feature importance rankings and input-vs-crop benchmark comparison tables.
7. **🔬 What-If Scenario Simulator:** Interactive sliders to perturb soil or weather features and observe real-time probability shifts and recommendation transitions.
8. **📊 Model Benchmark & Comparison:** Multi-metric comparison tables, interactive test-metric bar charts, and 5-Fold cross-validation stability plots.
9. **📜 Prediction History & Audit:** SQLite query viewer with search/filter capabilities, database statistics, and one-click CSV export.
10. **📖 Academic Documentation & Viva Prep:** Detailed methodology, viva question bank with model answers, mathematical definitions, and references.

---

## 🎓 Academic Viva Defense Guide

### Q1: Why choose Random Forest over Deep Neural Networks?
**Answer:** The dataset comprises 2,200 structured tabular observations across 7 physical dimensions. Deep neural networks risk overfitting and lack intrinsic feature interpretability on small-to-medium tabular data. Random Forest constructs orthogonal axis-aligned partitions matching physical agronomic thresholds (e.g. Rice requiring rainfall $> 200\text{ mm}$), demonstrates low variance via bootstrap aggregation, and provides clear Gini impurity feature attribution.

### Q2: How did you ensure zero data leakage?
**Answer:** Feature normalization (`StandardScaler`) was strictly encapsulated in `sklearn.pipeline.Pipeline` objects and fitted exclusively on training splits during both Stratified Cross-Validation and final holdout testing. The target column `label` was isolated from all feature transformations.

### Q3: Why did derived stoichiometric ratios not improve performance?
**Answer:** The original 7 features are physically independent measurements. The decision trees in Random Forest natively discover non-linear bivariate interactions without requiring explicit polynomial feature expansion. Adding collinear ratios introduced minor noise without adding net mutual information.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
