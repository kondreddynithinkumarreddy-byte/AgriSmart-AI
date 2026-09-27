"""
AgriSmart AI - Automated Smoke Test Suite
Validates model artifacts, database operations, inference engine, and diagnostics.
"""

import os
import sys
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_processing import load_and_validate_dataset, NUMERICAL_FEATURES
from src.prediction import (
    load_production_model,
    load_model_metadata,
    predict_crop_suitability,
    validate_input_features,
)
from src.soil_analysis import analyze_soil_nutrients
from src.environmental_analysis import analyze_environmental_conditions
from src.database import (
    init_db,
    log_prediction,
    get_recent_predictions,
    get_prediction_count,
    export_predictions_to_dataframe,
)
from src.report_generator import generate_pdf_report


def test_dataset_integrity():
    df, summary = load_and_validate_dataset()
    assert len(df) == 2200, f"Expected 2200 rows, got {len(df)}"
    assert summary["class_count"] == 22, f"Expected 22 classes, got {summary['class_count']}"
    assert summary["missing_values"] == 0, "Dataset contains unexpected nulls"
    for feat in NUMERICAL_FEATURES:
        assert feat in df.columns, f"Feature {feat} missing from dataset"


def test_model_and_metadata_loading():
    model = load_production_model()
    assert model is not None, "Failed to load final_model.pkl"
    
    metadata = load_model_metadata()
    assert "metrics" in metadata, "Missing metrics in metadata"
    assert "model_name" in metadata, "Missing model_name in metadata"

    results_path = os.path.join(PROJECT_ROOT, "models", "model_results.json")
    assert os.path.exists(results_path), "model_results.json does not exist"
    with open(results_path, "r") as f:
        res_json = json.load(f)
    assert res_json["dataset_rows"] == 2200
    assert res_json["dataset_classes"] == 22
    assert "random_forest" in res_json["metrics"]
    assert "logistic_regression" in res_json["metrics"]


def test_inference_pipeline():
    sample_inputs = {
        "N": 90.0,
        "P": 42.0,
        "K": 43.0,
        "temperature": 20.87,
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": 202.91,
    }
    val = validate_input_features(sample_inputs)
    assert val["is_valid"] is True, f"Validation failed: {val['errors']}"

    pred_res = predict_crop_suitability(sample_inputs)
    assert "predicted_crop" in pred_res
    assert isinstance(pred_res["predicted_crop"], str)
    assert pred_res["confidence"] > 0.0
    assert len(pred_res["top_recommendations"]) <= 5
    
    # Probabilities must sum close to 1.0
    if pred_res["all_probabilities"]:
        total_p = sum(item["probability"] for item in pred_res["all_probabilities"])
        assert abs(total_p - 1.0) < 1e-3, f"Probabilities do not sum to 1.0: {total_p}"


def test_soil_and_environmental_diagnostics():
    s_diag = analyze_soil_nutrients(90.0, 42.0, 43.0, 6.5)
    assert "npk_total" in s_diag
    assert "fertility_score" in s_diag
    assert 0 <= s_diag["fertility_score"] <= 100

    e_diag = analyze_environmental_conditions(20.87, 82.0, 202.91, "rice")
    assert "climate_regime" in e_diag
    assert "microclimate_index" in e_diag
    assert 0 <= e_diag["microclimate_index"] <= 100


def test_database_and_audit():
    init_db()
    initial_count = get_prediction_count()
    
    sample_inputs = {"N": 90.0, "P": 42.0, "K": 43.0, "temperature": 20.87, "humidity": 82.0, "ph": 6.5, "rainfall": 202.91}
    pred_res = predict_crop_suitability(sample_inputs)
    
    record_id = log_prediction(sample_inputs, pred_res, notes="Automated Smoke Test")
    assert record_id > 0
    assert get_prediction_count() == initial_count + 1

    df_hist = export_predictions_to_dataframe()
    assert not df_hist.empty
    assert "predicted_crop" in df_hist.columns


def test_pdf_report_generation():
    sample_inputs = {"N": 90.0, "P": 42.0, "K": 43.0, "temperature": 20.87, "humidity": 82.0, "ph": 6.5, "rainfall": 202.91}
    pred_res = predict_crop_suitability(sample_inputs)
    s_diag = analyze_soil_nutrients(90.0, 42.0, 43.0, 6.5)
    e_diag = analyze_environmental_conditions(20.87, 82.0, 202.91, "rice")

    pdf_bytes = generate_pdf_report(sample_inputs, pred_res, s_diag, e_diag)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING AGRISMART AI SMOKE TESTS...")
    print("=" * 60)
    test_dataset_integrity()
    print("[PASS] Dataset integrity test passed.")
    test_model_and_metadata_loading()
    print("[PASS] Model & metadata loading test passed.")
    test_inference_pipeline()
    print("[PASS] Inference pipeline test passed.")
    test_soil_and_environmental_diagnostics()
    print("[PASS] Soil & environmental diagnostic test passed.")
    test_database_and_audit()
    print("[PASS] SQLite database logging & audit test passed.")
    test_pdf_report_generation()
    print("[PASS] PDF report generation test passed.")
    print("=" * 60)
    print("ALL SMOKE TESTS PASSED SUCCESSFULLY! [OK]")
    print("=" * 60)

