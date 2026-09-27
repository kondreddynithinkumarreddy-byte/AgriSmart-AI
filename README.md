# 🌾 AgriSmart AI – Intelligent Crop Suitability & Soil Analysis System

> **Academic Title:** *An Explainable Machine Learning System for Crop Suitability Classification and Soil Analysis*
> **Target:** 10-Day ML Internship Final Capstone Project
> **Dataset:** Included crop recommendation dataset (2,200 records across 22 crop classes)

AgriSmart AI is a Streamlit decision-support application that predicts a crop from soil nutrients (N, P, K, pH) and climate inputs (temperature, humidity, rainfall). It also provides soil and environmental summaries, feature importance, scenario comparisons, prediction history, and downloadable PDF reports.

Predictions are based on the included dataset and model; they are not a substitute for local agronomic advice or field validation.

## Requirements

- Python 3.10 or newer
- The packages listed in [`requirements.txt`](requirements.txt)

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux, activate it with `source .venv/bin/activate`.

## Run the application

```powershell
streamlit run app/app.py
```

The application uses the checked-in model under `models/` and initializes its local SQLite history database under `database/`.

## Verify the project

Run these checks from the project root:

```powershell
python -m compileall -q app src
python src/safety_audit_test.py
python src/test_pipeline.py
```

The test scripts use a temporary database and generate reports in memory so running them does not add test records to the application database or overwrite project data.

## Retrain and benchmark

```powershell
python src/train_model.py
```

The training pipeline compares four classifiers on original and engineered features. It selects the production configuration using stratified cross-validation F1, then reports performance on a separate stratified holdout set. Preprocessing is kept inside scikit-learn pipelines.

The checked-in benchmark reports 99.55% holdout accuracy and 99.49% mean five-fold cross-validation weighted F1 for the selected Random Forest on the original seven features. These figures describe this dataset and split; they should not be interpreted as a guarantee of performance on new farms or regions.

## Project structure

```text
app/        Streamlit dashboard
data/       Input and engineered datasets
models/     Trained model and benchmark metadata
notebooks/  Exploratory analysis and project walkthrough
src/        Data, training, inference, analysis, reporting, and audit modules
```
