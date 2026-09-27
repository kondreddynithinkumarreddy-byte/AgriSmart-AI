"""
AgriSmart AI - Model Training & Benchmarking Pipeline
"""

import os
import sys
import json
import joblib
import numpy as np
from typing import Dict, Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.data_processing import load_and_validate_dataset, NUMERICAL_FEATURES, TARGET_COLUMN
from src.feature_engineering import create_engineered_features, save_engineered_dataset
from src.evaluate_model import evaluate_model_performance


def build_candidate_models() -> Dict[str, Any]:
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "K-Nearest Neighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=5, weights="distance"))
        ]),
        "Decision Tree": Pipeline([
            ("classifier", DecisionTreeClassifier(max_depth=12, min_samples_split=5, random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("classifier", RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=4, random_state=42))
        ]),
    }


def train_and_evaluate_all_models() -> Dict[str, Any]:
    print("=" * 70)
    print(" AGRISMART AI - MULTI-MODEL BENCHMARKING & EVALUATION PIPELINE")
    print("=" * 70)

    df_raw, data_summary = load_and_validate_dataset()
    classes = data_summary["unique_classes"]

    save_engineered_dataset(df_raw)
    df_eng = create_engineered_features(df_raw)

    feature_sets = {
        "Original Features (7)": {
            "df": df_raw,
            "feature_cols": NUMERICAL_FEATURES,
        },
        "Engineered Features (13)": {
            "df": df_eng,
            "feature_cols": [c for c in df_eng.columns if c != TARGET_COLUMN],
        },
    }

    models_dict = build_candidate_models()
    benchmark_results = []
    trained_pipelines = {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fs_name, fs_info in feature_sets.items():
        curr_df = fs_info["df"]
        feature_cols = fs_info["feature_cols"]

        X = curr_df[feature_cols]
        y = curr_df[TARGET_COLUMN]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        for model_name, base_pipe in models_dict.items():
            pipe = clone(base_pipe)
            print(f"--> Training {model_name} on [{fs_name}]...")

            cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_weighted")
            cv_mean = float(np.mean(cv_scores))
            cv_std = float(np.std(cv_scores))

            pipe.fit(X_train, y_train)

            y_pred = pipe.predict(X_test)
            y_prob = pipe.predict_proba(X_test) if hasattr(pipe, "predict_proba") else None

            metrics = evaluate_model_performance(y_test, y_pred, y_prob, classes=classes)

            config_key = f"{model_name} | {fs_name}"
            trained_pipelines[config_key] = {
                "pipeline": pipe,
                "feature_cols": feature_cols,
                "fs_name": fs_name,
                "model_name": model_name,
            }

            record = {
                "config_name": config_key,
                "model_name": model_name,
                "feature_set": fs_name,
                "feature_count": len(feature_cols),
                "features": feature_cols,
                "cv_f1_mean": round(cv_mean, 4),
                "cv_f1_std": round(cv_std, 4),
                "test_accuracy": metrics["accuracy"],
                "test_precision_weighted": metrics["precision_weighted"],
                "test_precision_macro": metrics["precision_macro"],
                "test_recall_weighted": metrics["recall_weighted"],
                "test_recall_macro": metrics["recall_macro"],
                "test_f1_weighted": metrics["f1_weighted"],
                "test_f1_macro": metrics["f1_macro"],
                "test_roc_auc_ovr": metrics["roc_auc_ovr"],
                "confusion_matrix": metrics["confusion_matrix"],
            }
            benchmark_results.append(record)

    benchmark_results.sort(
        key=lambda x: (x["cv_f1_mean"], -x["cv_f1_std"]),
        reverse=True
    )
    winner_meta = benchmark_results[0]
    winner_key = winner_meta["config_name"]
    winner_obj = trained_pipelines[winner_key]

    print(f"\nWINNING MODEL SELECTED: {winner_key}")
    print(f"Test Accuracy: {winner_meta['test_accuracy']*100:.2f}% | Test F1: {winner_meta['test_f1_weighted']*100:.2f}%")

    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "final_model.pkl")
    joblib.dump(winner_obj["pipeline"], model_path)

    preprocessor = None
    if "scaler" in winner_obj["pipeline"].named_steps:
        preprocessor = winner_obj["pipeline"].named_steps["scaler"]
    prep_path = os.path.join(models_dir, "preprocessing_pipeline.pkl")
    joblib.dump(preprocessor, prep_path)

    comp_path = os.path.join(models_dir, "model_comparison.json")
    with open(comp_path, "w") as f:
        json.dump(benchmark_results, f, indent=2)

    metadata = {
        "winning_config": winner_key,
        "model_name": winner_meta["model_name"],
        "feature_set": winner_meta["feature_set"],
        "feature_count": winner_meta["feature_count"],
        "features": winner_meta["features"],
        "classes": classes,
        "metrics": {
            "accuracy": winner_meta["test_accuracy"],
            "f1_weighted": winner_meta["test_f1_weighted"],
            "precision_weighted": winner_meta["test_precision_weighted"],
            "recall_weighted": winner_meta["test_recall_weighted"],
            "cv_f1_mean": winner_meta["cv_f1_mean"],
            "cv_f1_std": winner_meta["cv_f1_std"],
        },
        "confusion_matrix": winner_meta["confusion_matrix"],
    }
    meta_path = os.path.join(models_dir, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


if __name__ == "__main__":
    train_and_evaluate_all_models()
