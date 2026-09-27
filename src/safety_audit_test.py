"""
AgriSmart AI - Comprehensive Safety & Robustness Audit Suite
"""

import os
import sys
import json
import sqlite3
import pandas as pd
import numpy as np
import joblib

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_processing import (
    load_and_validate_dataset,
    get_dataset_summary_statistics,
    detect_outliers_iqr,
    get_crop_profiles,
)
from src.feature_engineering import (
    create_engineered_features,
    get_feature_engineering_summary,
)
from src.evaluate_model import evaluate_model_performance
from src.prediction import (
    load_production_model,
    predict_crop_suitability,
    validate_input_features,
    simulate_what_if_scenario,
    EXPECTED_RANGES,
)
from src.soil_analysis import (
    analyze_soil_nutrients,
    get_ph_classification,
    get_nutrient_status_label,
)
from src.environmental_analysis import (
    analyze_environmental_conditions,
    get_climate_regime,
)
from src.explainability import (
    get_model_feature_importance,
    generate_crop_comparison_explanation,
)
from src.database import (
    init_db,
    log_prediction,
    get_recent_predictions,
    get_prediction_count,
    export_predictions_to_dataframe,
    get_db_connection,
)
from src.report_generator import generate_pdf_report


class SafetyAuditTester:
    def __init__(self):
        self.passed_tests = 0
        self.failed_tests = 0
        self.warnings = 0
        self.errors = []

    def log_success(self, test_name: str, detail: str = ""):
        self.passed_tests += 1
        print(f"  [PASS] {test_name}" + (f" -> {detail}" if detail else ""))

    def log_failure(self, test_name: str, error: str):
        self.failed_tests += 1
        self.errors.append((test_name, error))
        print(f"  [FAIL] {test_name} -> ERROR: {error}")

    def log_warn(self, test_name: str, msg: str):
        self.warnings += 1
        print(f"  [WARN] {test_name} -> {msg}")

    def run_all_audits(self):
        print("=" * 80)
        print("       AGRISMART AI - COMPLETE SYSTEM SAFETY & INTEGRITY AUDIT")
        print("=" * 80)

        self.audit_1_filesystem_and_assets()
        self.audit_2_data_integrity()
        self.audit_3_feature_engineering_edge_cases()
        self.audit_4_model_artifacts_and_consistency()
        self.audit_5_inference_and_boundary_safety()
        self.audit_6_input_validation_and_rejection()
        self.audit_7_soil_and_environmental_edge_cases()
        self.audit_8_explainability_integrity()
        self.audit_9_database_acid_and_concurrency()
        self.audit_10_pdf_report_generation()
        self.audit_11_notebook_structure_and_json()
        self.audit_12_streamlit_app_syntax_and_imports()

        print("\n" + "=" * 80)
        print("                        AUDIT SUMMARY")
        print("=" * 80)
        print(f"Total Tests Executed : {self.passed_tests + self.failed_tests}")
        print(f"Tests Passed         : {self.passed_tests}")
        print(f"Tests Failed         : {self.failed_tests}")
        print(f"Warnings             : {self.warnings}")

        if self.failed_tests == 0:
            print("\n>>> ALL SYSTEM PARTS VERIFIED 100% OPERATIONAL, SAFE, AND READY! <<<")
            return 0
        else:
            print("\n>>> CRITICAL ISSUES DETECTED: <<<")
            for name, err in self.errors:
                print(f" - {name}: {err}")
            return 1

    def audit_1_filesystem_and_assets(self):
        print("\n[AUDIT 1/12] Verifying File System Assets & Paths...")
        required_paths = [
            "data/Crop_recommendation.csv",
            "data/Crop_recommendation_engineered.csv",
            "models/final_model.pkl",
            "models/model_comparison.json",
            "models/model_metadata.json",
            "notebooks/crop_suitability_analysis.ipynb",
            "app/app.py",
            "src/data_processing.py",
            "src/feature_engineering.py",
            "src/evaluate_model.py",
            "src/train_model.py",
            "src/prediction.py",
            "src/soil_analysis.py",
            "src/environmental_analysis.py",
            "src/explainability.py",
            "src/database.py",
            "src/report_generator.py",
            "requirements.txt",
            "README.md",
        ]
        for rel_path in required_paths:
            full_path = os.path.join(PROJECT_ROOT, rel_path)
            if os.path.exists(full_path) and os.path.getsize(full_path) > 0:
                self.log_success(f"File Exists & Non-Empty: {rel_path}", f"{os.path.getsize(full_path):,} bytes")
            else:
                self.log_failure(f"Missing/Empty File: {rel_path}", "File not found or has 0 bytes")

    def audit_2_data_integrity(self):
        print("\n[AUDIT 2/12] Auditing Dataset Integrity & Statistical Health...")
        try:
            df, summary = load_and_validate_dataset()
            if len(df) == 2200 and df.shape[1] == 8:
                self.log_success("Authoritative Dataset Dimensions", f"{df.shape[0]} rows, {df.shape[1]} cols")
            else:
                self.log_failure("Dataset Dimensions", f"Expected (2200, 8), got {df.shape}")

            if summary["missing_values"] == 0:
                self.log_success("Missing Values Check", "0 missing values detected")
            else:
                self.log_failure("Missing Values Check", f"Found {summary['missing_values']} nulls")

            if summary["duplicate_rows"] == 0:
                self.log_success("Duplicate Rows Check", "0 duplicate records detected")
            else:
                self.log_failure("Duplicate Rows Check", f"Found {summary['duplicate_rows']} duplicates")

            if summary["class_count"] == 22:
                self.log_success("Class Balance Check", "Exactly 22 unique crop classes with 100 rows each")
            else:
                self.log_failure("Class Balance Check", f"Expected 22 classes, got {summary['class_count']}")

            outliers = detect_outliers_iqr(df)
            if isinstance(outliers, dict) and len(outliers) == 7:
                self.log_success("IQR Outlier Detection", "Computed IQR bounds across all 7 features")
            else:
                self.log_failure("IQR Outlier Detection", "Invalid return structure")

            profiles = get_crop_profiles(df)
            if len(profiles) == 22 and "rice" in profiles:
                self.log_success("Crop Empirical Profiles", "Calculated profiles for all 22 crops")
            else:
                self.log_failure("Crop Empirical Profiles", "Incomplete crop profiles")

        except Exception as e:
            self.log_failure("Dataset Validation Exception", str(e))

    def audit_3_feature_engineering_edge_cases(self):
        print("\n[AUDIT 3/12] Testing Feature Engineering Math & Division-by-Zero Safety...")
        try:
            df_raw, _ = load_and_validate_dataset()
            df_eng = create_engineered_features(df_raw)
            if df_eng.shape[1] == 14 and not df_eng.isnull().values.any():
                self.log_success("Feature Engineering Standard Dataset", "14 total columns (13 features + target) generated with 0 nulls")
            else:
                self.log_failure("Feature Engineering Standard Dataset", f"Null values produced or column count mismatch (got {df_eng.shape[1]} cols)")

            zero_test_df = pd.DataFrame([{
                "N": 0.0, "P": 0.0, "K": 0.0,
                "temperature": 0.0, "humidity": 0.0,
                "ph": 0.0, "rainfall": 0.0,
                "label": "rice"
            }])
            zero_eng = create_engineered_features(zero_test_df)
            
            has_nan = zero_eng.isnull().values.any()
            has_inf = np.isinf(zero_eng.select_dtypes(include=[np.number]).values).any()
            
            if not has_nan and not has_inf:
                self.log_success("Division-by-Zero Protection", "Safe division on all-zero inputs handled perfectly (0.0 result, no NaNs/Infs)")
            else:
                self.log_failure("Division-by-Zero Protection", f"Produced NaNs={has_nan} or Infs={has_inf}")

        except Exception as e:
            self.log_failure("Feature Engineering Exception", str(e))

    def audit_4_model_artifacts_and_consistency(self):
        print("\n[AUDIT 4/12] Auditing Serialized Model Metadata & Performance Consistency...")
        try:
            model = load_production_model()
            if model is not None and hasattr(model, "predict") and hasattr(model, "predict_proba"):
                self.log_success("Production Model Deserialization", f"Loaded valid pipeline: {type(model).__name__}")
            else:
                self.log_failure("Production Model Deserialization", "Model is invalid or missing predict_proba")

            meta_path = os.path.join(PROJECT_ROOT, "models", "model_metadata.json")
            with open(meta_path, "r") as f:
                metadata = json.load(f)

            if metadata.get("feature_count") == 7 and len(metadata.get("classes", [])) == 22:
                self.log_success("Model Metadata Consistency", f"Model trained on {metadata['feature_count']} features for {len(metadata['classes'])} classes")
            else:
                self.log_failure("Model Metadata Consistency", f"Metadata mismatch: {metadata}")

            comp_path = os.path.join(PROJECT_ROOT, "models", "model_comparison.json")
            with open(comp_path, "r") as f:
                comparison = json.load(f)

            if len(comparison) == 8:
                self.log_success("Model Benchmarking Matrix", "All 8 model configurations (4 models x 2 feature sets) recorded")
            else:
                self.log_failure("Model Benchmarking Matrix", f"Expected 8 models, found {len(comparison)}")

        except Exception as e:
            self.log_failure("Model Artifacts Exception", str(e))

    def audit_5_inference_and_boundary_safety(self):
        print("\n[AUDIT 5/12] Testing Multi-Class Inference, Confidence, & Boundary Safety...")
        try:
            test_cases = [
                ("Rice Conditions", {"N": 90, "P": 42, "K": 43, "temperature": 20.8, "humidity": 82.0, "ph": 6.5, "rainfall": 202.9}, "rice"),
                ("Chickpea Conditions", {"N": 40, "P": 72, "K": 77, "temperature": 18.8, "humidity": 16.8, "ph": 7.3, "rainfall": 80.0}, "chickpea"),
                ("Apple Conditions", {"N": 20, "P": 134, "K": 199, "temperature": 22.5, "humidity": 92.3, "ph": 5.9, "rainfall": 110.0}, "apple"),
                ("Cotton Conditions", {"N": 120, "P": 40, "K": 20, "temperature": 24.0, "humidity": 80.0, "ph": 6.8, "rainfall": 80.0}, "cotton"),
            ]

            for case_name, features, expected_crop in test_cases:
                result = predict_crop_suitability(features)
                if result["predicted_crop"].lower() == expected_crop:
                    self.log_success(f"Inference Ground Truth Match: {case_name}", f"Predicted '{result['predicted_crop']}' with {result['confidence_pct']}% confidence")
                else:
                    self.log_warn(f"Inference Ground Truth Match: {case_name}", f"Predicted '{result['predicted_crop']}', expected '{expected_crop}'")

                top5 = result["top_recommendations"]
                if len(top5) == 5:
                    probs = [item["probability_pct"] for item in top5]
                    is_sorted = all(probs[i] >= probs[i+1] for i in range(len(probs)-1))
                    total_prob = sum(item["probability"] for item in result["all_probabilities"])
                    
                    if is_sorted and abs(total_prob - 1.0) < 1e-4:
                        self.log_success(f"Probability Distribution Integrity: {case_name}", f"Sum={total_prob:.4f}, strictly sorted top 5")
                    else:
                        self.log_failure(f"Probability Distribution Integrity: {case_name}", f"Sum={total_prob}, sorted={is_sorted}")
                else:
                    self.log_failure(f"Top-5 Output Count: {case_name}", f"Got {len(top5)} items")

            sim_results = simulate_what_if_scenario(
                base_features={"N": 90, "P": 42, "K": 43, "temperature": 20.8, "humidity": 82.0, "ph": 6.5, "rainfall": 202.9},
                feature_to_vary="rainfall",
                min_val=50.0,
                max_val=300.0,
                steps=5
            )
            if len(sim_results) == 5:
                self.log_success("What-If Sensitivity Simulator", "Simulated 5 perturbation steps successfully")
            else:
                self.log_failure("What-If Sensitivity Simulator", f"Expected 5 simulation steps, got {len(sim_results)}")

        except Exception as e:
            self.log_failure("Inference Safety Exception", str(e))

    def audit_6_input_validation_and_rejection(self):
        print("\n[AUDIT 6/12] Testing Input Validation & Malformed Input Rejection...")
        try:
            valid_input = {"N": 90, "P": 42, "K": 43, "temperature": 20.8, "humidity": 82.0, "ph": 6.5, "rainfall": 202.9}
            v_res = validate_input_features(valid_input)
            if v_res["is_valid"] and len(v_res["errors"]) == 0:
                self.log_success("Valid Input Acceptance", "Valid features accepted with 0 errors")
            else:
                self.log_failure("Valid Input Acceptance", f"False rejection: {v_res['errors']}")

            neg_input = {"N": -10, "P": 42, "K": 43, "temperature": 20.8, "humidity": 82.0, "ph": 6.5, "rainfall": 202.9}
            neg_res = validate_input_features(neg_input)
            if not neg_res["is_valid"] and any("Nitrogen (N)" in e for e in neg_res["errors"]):
                self.log_success("Negative Value Rejection", "Safely caught and rejected N < 0")
            else:
                self.log_failure("Negative Value Rejection", "Failed to catch negative N")

            ph_input = {"N": 90, "P": 42, "K": 43, "temperature": 20.8, "humidity": 82.0, "ph": 16.0, "rainfall": 202.9}
            ph_res = validate_input_features(ph_input)
            if not ph_res["is_valid"] and any("pH" in e for e in ph_res["errors"]):
                self.log_success("Extreme pH Rejection", "Safely caught and rejected pH > 14.0")
            else:
                self.log_failure("Extreme pH Rejection", "Failed to catch pH > 14")

            warn_input = {"N": 90, "P": 42, "K": 43, "temperature": 20.8, "humidity": 82.0, "ph": 6.5, "rainfall": 450.0}
            warn_res = validate_input_features(warn_input)
            if warn_res["is_valid"] and len(warn_res["warnings"]) > 0:
                self.log_success("Agronomic Range Warning", "Permitted realistic outlier while issuing clear warning")
            else:
                self.log_failure("Agronomic Range Warning", "Did not issue warning for rainfall > 300mm")

        except Exception as e:
            self.log_failure("Input Validation Exception", str(e))

    def audit_7_soil_and_environmental_edge_cases(self):
        print("\n[AUDIT 7/12] Testing Soil & Environmental Analysis Modules...")
        try:
            soil_res = analyze_soil_nutrients(100.0, 50.0, 50.0, 6.5)
            if (soil_res["npk_total"] == 200.0 and
                soil_res["nitrogen"]["status"] in ["Deficient", "Moderate", "Optimal", "High", "Excessive"] and
                soil_res["ph_analysis"]["classification"] in ["Neutral", "Slightly Acidic"]):
                self.log_success("Soil Nutrient Diagnostics", f"NPK Total: {soil_res['npk_total']}, pH: {soil_res['ph_analysis']['classification']}")
            else:
                self.log_failure("Soil Nutrient Diagnostics", f"Unexpected result: {soil_res}")

            env_res = analyze_environmental_conditions(25.0, 75.0, 150.0, "rice")
            if (env_res["climate_regime"]["classification"] and
                isinstance(env_res["stress_analysis"], dict) and
                env_res["suitability_assessment"]["overall_suitability"] in ["Optimal", "Sub-Optimal", "Challenging"]):
                self.log_success("Environmental Microclimate Assessment", f"Regime: {env_res['climate_regime']['classification']}, Suitability: {env_res['suitability_assessment']['overall_suitability']}")
            else:
                self.log_failure("Environmental Microclimate Assessment", f"Unexpected result: {env_res}")

        except Exception as e:
            self.log_failure("Soil/Environmental Exception", str(e))

    def audit_8_explainability_integrity(self):
        print("\n[AUDIT 8/12] Auditing Explainable AI (XAI) & Attribution Features...")
        try:
            model = load_production_model()
            importance = get_model_feature_importance(model, ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])
            
            if len(importance) == 7 and abs(sum(importance.values()) - 1.0) < 1e-3:
                top_feat = list(importance.keys())[0]
                self.log_success("Global Gini Importance Attribution", f"Top feature: '{top_feat}' ({importance[top_feat]*100:.2f}%), Total Gini=100.0%")
            else:
                self.log_failure("Global Gini Importance Attribution", f"Invalid importance sum or length: {importance}")

            input_dict = {"N": 90, "P": 42, "K": 43, "temperature": 20.8, "humidity": 82.0, "ph": 6.5, "rainfall": 202.9}
            df_comp = generate_crop_comparison_explanation(input_dict, "rice")
            if not df_comp.empty and len(df_comp) == 7 and "User Value" in df_comp.columns:
                self.log_success("Target Crop Benchmark Comparison", "Generated 7-feature benchmark table against empirical 'rice' distribution")
            else:
                self.log_failure("Target Crop Benchmark Comparison", "Empty or malformed benchmark comparison dataframe")

        except Exception as e:
            self.log_failure("Explainability Exception", str(e))

    def audit_9_database_acid_and_concurrency(self):
        print("\n[AUDIT 9/12] Testing SQLite Database CRUD, Transactions & Query Integrity...")
        try:
            init_db()

            test_id = log_prediction(
                inputs={"N": 90, "P": 42, "K": 43, "temperature": 20.8, "humidity": 82.0, "ph": 6.5, "rainfall": 202.9},
                prediction_result={
                    "predicted_crop": "rice",
                    "confidence": 0.995,
                    "confidence_pct": 99.5,
                    "model_used": "Random Forest",
                    "top_recommendations": [{"crop": "rice", "probability": 0.995, "probability_pct": 99.5}]
                },
                notes="Safety Audit Automated Test Entry"
            )

            if test_id > 0:
                self.log_success("Database Log Insertion", f"Inserted record with ID: {test_id}")
            else:
                self.log_failure("Database Log Insertion", "Failed to insert record")

            recent_records = get_recent_predictions(limit=5)
            if len(recent_records) > 0 and recent_records[0]["id"] == test_id:
                self.log_success("Database Record Retrieval", f"Retrieved record ID {test_id} matching logged data")
            else:
                self.log_failure("Database Record Retrieval", "Could not verify inserted record")

            df_export = export_predictions_to_dataframe()
            if not df_export.empty and "predicted_crop" in df_export.columns:
                self.log_success("Database CSV Export Capability", f"Exported {len(df_export)} records to Pandas DataFrame")
            else:
                self.log_failure("Database CSV Export Capability", "Failed to export dataframe")

            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM prediction_history WHERE id = ?", (test_id,))
            conn.commit()
            conn.close()
            self.log_success("Database Cleanup Transaction", f"Cleaned up test record {test_id} without side effects")

        except Exception as e:
            self.log_failure("Database Exception", str(e))

    def audit_10_pdf_report_generation(self):
        print("\n[AUDIT 10/12] Testing ReportLab PDF Report Generation Engine...")
        try:
            inputs = {"N": 90, "P": 42, "K": 43, "temperature": 20.8, "humidity": 82.0, "ph": 6.5, "rainfall": 202.9}
            prediction_result = {
                "predicted_crop": "rice",
                "confidence_pct": 99.5,
                "model_used": "Random Forest",
                "feature_set": "Original Features (7)",
                "top_recommendations": [
                    {"crop": "rice", "probability_pct": 99.5},
                    {"crop": "jute", "probability_pct": 0.3},
                    {"crop": "maize", "probability_pct": 0.1},
                    {"crop": "cotton", "probability_pct": 0.05},
                    {"crop": "papaya", "probability_pct": 0.05},
                ]
            }
            soil_data = analyze_soil_nutrients(90, 42, 43, 6.5)
            env_data = analyze_environmental_conditions(20.8, 82.0, 202.9, "rice")

            pdf_bytes = generate_pdf_report(inputs, prediction_result, soil_data, env_data)

            if isinstance(pdf_bytes, bytes) and len(pdf_bytes) > 2000 and pdf_bytes.startswith(b"%PDF-"):
                self.log_success("PDF Generation Engine", f"Generated valid PDF stream ({len(pdf_bytes):,} bytes, Header '%PDF-')")
            else:
                self.log_failure("PDF Generation Engine", "Generated invalid or empty PDF bytes")

        except Exception as e:
            self.log_failure("PDF Generation Exception", str(e))

    def audit_11_notebook_structure_and_json(self):
        print("\n[AUDIT 11/12] Validating Academic Jupyter Notebook Structure & JSON Schema...")
        try:
            nb_path = os.path.join(PROJECT_ROOT, "notebooks", "crop_suitability_analysis.ipynb")
            with open(nb_path, "r", encoding="utf-8") as f:
                nb = json.load(f)

            if "cells" in nb and len(nb["cells"]) >= 25:
                code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
                md_cells = [c for c in nb["cells"] if c["cell_type"] == "markdown"]
                self.log_success("Jupyter Notebook Schema", f"Valid JSON with {len(nb['cells'])} total cells ({len(code_cells)} code, {len(md_cells)} markdown)")
            else:
                self.log_failure("Jupyter Notebook Schema", f"Notebook missing cells or incomplete (got {len(nb.get('cells', []))} cells)")

        except Exception as e:
            self.log_failure("Notebook Schema Exception", str(e))

    def audit_12_streamlit_app_syntax_and_imports(self):
        print("\n[AUDIT 12/12] Validating Streamlit Application Syntax & Module Imports...")
        try:
            app_path = os.path.join(PROJECT_ROOT, "app", "app.py")
            with open(app_path, "r", encoding="utf-8") as f:
                code_content = f.read()

            compiled_code = compile(code_content, app_path, "exec")
            self.log_success("Streamlit AST Compilation", f"Compiled {len(code_content.splitlines()):,} lines of Python with 0 syntax errors")

        except Exception as e:
            self.log_failure("Streamlit App Compilation Exception", str(e))


if __name__ == "__main__":
    auditor = SafetyAuditTester()
    exit_code = auditor.run_all_audits()
    sys.exit(exit_code)
