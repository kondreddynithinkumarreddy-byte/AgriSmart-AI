"""
AgriSmart AI - System Integration Test Suite
"""

import os
import sys
import json
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_processing import (
    load_and_validate_dataset,
    get_dataset_summary_statistics,
    detect_outliers_iqr,
    get_crop_profiles,
)
from src.feature_engineering import create_engineered_features, save_engineered_dataset
from src.prediction import (
    load_production_model,
    predict_crop_suitability,
    validate_input_features,
    simulate_what_if_scenario,
)
from src.soil_analysis import analyze_soil_nutrients
from src.environmental_analysis import analyze_environmental_conditions
from src.explainability import get_model_feature_importance, generate_crop_comparison_explanation
from src.database import init_db, log_prediction, get_recent_predictions, export_predictions_to_dataframe
from src.report_generator import generate_pdf_report


def run_integration_tests():
    print("=" * 75)
    print("       AGRISMART AI - SYSTEM INTEGRATION TEST SUITE")
    print("=" * 75)

    print("\n[STAGE 1/8] Testing Data Ingestion & Statistical Integrity...")
    df, summary = load_and_validate_dataset()
    assert len(df) == 2200, f"Expected 2200 rows, got {len(df)}"
    assert summary["missing_values"] == 0, "Found unexpected null values"
    assert summary["duplicate_rows"] == 0, "Found unexpected duplicate rows"
    assert summary["class_count"] == 22, f"Expected 22 classes, got {summary['class_count']}"
    print("  -> PASSED: 2,200 rows, 0 nulls, 0 duplicates, 22 balanced classes.")

    print("\n[STAGE 2/8] Testing Feature Engineering & Math Safeguards...")
    df_eng = create_engineered_features(df)
    assert df_eng.shape[1] == 14, f"Expected 14 columns, got {df_eng.shape[1]}"
    assert not df_eng.isnull().values.any(), "Engineered dataset contains NaNs"
    save_engineered_dataset(df)
    print("  -> PASSED: Derived 6 features safely; saved Crop_recommendation_engineered.csv.")

    print("\n[STAGE 3/8] Testing Serialized Production Model & Metadata...")
    model = load_production_model()
    assert hasattr(model, "predict"), "Model missing predict method"
    assert hasattr(model, "predict_proba"), "Model missing predict_proba method"
    meta_path = os.path.join(PROJECT_ROOT, "models", "model_metadata.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    print(f"  -> PASSED: Production model loaded successfully ({meta['winning_config']}).")

    print("\n[STAGE 4/8] Testing Prediction & Inference Pipeline...")
    sample_inputs = {"N": 90.0, "P": 42.0, "K": 43.0, "temperature": 20.87, "humidity": 82.0, "ph": 6.5, "rainfall": 202.9}
    pred_res = predict_crop_suitability(sample_inputs, model=model, metadata=meta)
    assert pred_res["predicted_crop"] == "rice", f"Expected 'rice', got {pred_res['predicted_crop']}"
    assert len(pred_res["top_recommendations"]) == 5, "Top recommendations should contain 5 items"
    print(f"  -> PASSED: Predicted '{pred_res['predicted_crop']}' ({pred_res['confidence_pct']}% confidence).")

    print("\n[STAGE 5/8] Testing Soil & Environmental Analysis...")
    soil_res = analyze_soil_nutrients(sample_inputs["N"], sample_inputs["P"], sample_inputs["K"], sample_inputs["ph"])
    env_res = analyze_environmental_conditions(sample_inputs["temperature"], sample_inputs["humidity"], sample_inputs["rainfall"], "rice")
    assert soil_res["fertility_score"] > 0, "Fertility score error"
    assert env_res["microclimate_index"] > 0, "Microclimate score error"
    print(f"  -> PASSED: Soil Fertility={soil_res['fertility_score']}/100, Climate Regime='{env_res['climate_regime']['classification']}'.")

    print("\n[STAGE 6/8] Testing Explainability (XAI) & Benchmarking...")
    importances = get_model_feature_importance(model, meta["features"])
    comp_df = generate_crop_comparison_explanation(sample_inputs, "rice")
    assert len(importances) == 7, "Feature importances count mismatch"
    assert not comp_df.empty, "Crop comparison table empty"
    print(f"  -> PASSED: Top feature is '{list(importances.keys())[0]}' ({list(importances.values())[0]*100:.1f}%).")

    print("\n[STAGE 7/8] Testing Database CRUD & Transactions...")
    init_db()
    rec_id = log_prediction(sample_inputs, pred_res, notes="Integration Test Run")
    assert rec_id > 0, "Failed to log prediction to SQLite"
    recent = get_recent_predictions(limit=5)
    assert len(recent) > 0, "Failed to retrieve recent predictions"
    df_db = export_predictions_to_dataframe()
    assert not df_db.empty, "Failed to export dataframe"
    print(f"  -> PASSED: Logged transaction ID #{rec_id}, verified retrieval and dataframe export.")

    print("\n[STAGE 8/8] Testing PDF Report Generation Engine...")
    pdf_path = os.path.join(PROJECT_ROOT, "reports", "sample_report.pdf")
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    pdf_bytes = generate_pdf_report(sample_inputs, pred_res, soil_res, env_res, output_path=pdf_path)
    assert os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 1000, "PDF file generation failed"
    print(f"  -> PASSED: Generated PDF report at {pdf_path} ({os.path.getsize(pdf_path):,} bytes).")

    print("\n" + "=" * 75)
    print(" >>> ALL 8 SYSTEM INTEGRATION STAGES PASSED WITH ZERO ERRORS! <<<")
    print("=" * 75)


if __name__ == "__main__":
    run_integration_tests()
