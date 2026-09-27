"""
AgriSmart AI - Prediction & Inference Engine
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.feature_engineering import create_engineered_features

EXPECTED_RANGES: Dict[str, Dict[str, Any]] = {
    "N": {"min": 0.0, "max": 140.0, "unit": "kg/ha", "name": "Nitrogen (N)"},
    "P": {"min": 5.0, "max": 145.0, "unit": "kg/ha", "name": "Phosphorus (P)"},
    "K": {"min": 5.0, "max": 205.0, "unit": "kg/ha", "name": "Potassium (K)"},
    "temperature": {"min": 8.0, "max": 45.0, "unit": "°C", "name": "Temperature"},
    "humidity": {"min": 10.0, "max": 100.0, "unit": "%", "name": "Relative Humidity"},
    "ph": {"min": 3.5, "max": 10.0, "unit": "pH", "name": "Soil pH"},
    "rainfall": {"min": 20.0, "max": 300.0, "unit": "mm", "name": "Rainfall"},
}

CROP_ICONS: Dict[str, str] = {
    "rice": "🌾", "maize": "🌽", "chickpea": "🧆", "kidneybeans": "🫘",
    "pigeonpeas": "🌱", "mothbeans": "🌿", "mungbean": "🥗", "blackgram": "🪴",
    "lentil": "🍲", "pomegranate": "🍎", "banana": "🍌", "mango": "🥭",
    "grapes": "🍇", "watermelon": "🍉", "muskmelon": "🍈", "apple": "🍏",
    "orange": "🍊", "papaya": "🍈", "coconut": "🥥", "cotton": "👕",
    "jute": "🧵", "coffee": "☕",
}


def load_production_model():
    model_path = os.path.join(PROJECT_ROOT, "models", "final_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}.")
    return joblib.load(model_path)


def load_model_metadata() -> Dict[str, Any]:
    meta_path = os.path.join(PROJECT_ROOT, "models", "model_metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found at {meta_path}.")
    with open(meta_path, "r") as f:
        return json.load(f)


def validate_input_features(features: Dict[str, float]) -> Dict[str, Any]:
    errors = []
    warnings = []

    required_keys = list(EXPECTED_RANGES)
    numeric_features: Dict[str, float] = {}

    for k in required_keys:
        if k not in features:
            errors.append(f"Missing required input parameter: '{k}'")
            continue

        try:
            value = float(features[k])
        except (TypeError, ValueError, OverflowError):
            errors.append(f"{EXPECTED_RANGES[k]['name']} must be a finite number.")
            continue

        if not np.isfinite(value):
            errors.append(f"{EXPECTED_RANGES[k]['name']} must be a finite number.")
            continue

        numeric_features[k] = value

    if errors:
        return {"is_valid": False, "errors": errors, "warnings": warnings}

    if numeric_features["N"] < 0:
        errors.append("Nitrogen (N) cannot be negative.")
    if numeric_features["P"] < 0:
        errors.append("Phosphorus (P) cannot be negative.")
    if numeric_features["K"] < 0:
        errors.append("Potassium (K) cannot be negative.")
    if numeric_features["temperature"] < -273.15:
        errors.append("Temperature cannot be below absolute zero (-273.15 °C).")
    if numeric_features["humidity"] < 0 or numeric_features["humidity"] > 100:
        errors.append("Relative Humidity must be between 0% and 100%.")
    if numeric_features["ph"] < 0 or numeric_features["ph"] > 14:
        errors.append("Soil pH must be within the chemical range of 0.0 to 14.0.")
    if numeric_features["rainfall"] < 0:
        errors.append("Rainfall cannot be negative.")

    for key, val in numeric_features.items():
        bounds = EXPECTED_RANGES[key]
        if val < bounds["min"] or val > bounds["max"]:
            warnings.append(
                f"{bounds['name']} ({val} {bounds['unit']}) is outside typical training range "
                f"[{bounds['min']} - {bounds['max']} {bounds['unit']}]."
            )

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


def predict_crop_suitability(
    inputs: Dict[str, float],
    model=None,
    metadata: Dict[str, Any] = None,
) -> Dict[str, Any]:
    validation = validate_input_features(inputs)
    if not validation["is_valid"]:
        raise ValueError("Invalid prediction inputs: " + "; ".join(validation["errors"]))

    normalized_inputs = {key: float(inputs[key]) for key in EXPECTED_RANGES}

    if model is None:
        model = load_production_model()
    if metadata is None:
        metadata = load_model_metadata()

    raw_df = pd.DataFrame([normalized_inputs])
    feature_cols = metadata.get("features", ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])
    if len(feature_cols) > 7:
        input_df = create_engineered_features(raw_df)[feature_cols]
    else:
        input_df = raw_df[feature_cols]

    predicted_class = str(model.predict(input_df)[0])
    has_proba = hasattr(model, "predict_proba")
    top_recommendations = []
    all_probabilities = []
    confidence = 1.0

    if has_proba:
        proba_matrix = model.predict_proba(input_df)[0]
        classes = model.classes_
        
        for cls_name, prob in zip(classes, proba_matrix):
            all_probabilities.append({
                "crop": str(cls_name),
                "icon": CROP_ICONS.get(str(cls_name).lower(), "🌱"),
                "probability": float(prob),
                "probability_pct": round(float(prob) * 100, 2),
            })

        all_probabilities.sort(key=lambda x: x["probability"], reverse=True)
        top_recommendations = all_probabilities[:5]
        confidence = float(top_recommendations[0]["probability"])
    else:
        top_recommendations = [{
            "crop": predicted_class,
            "icon": CROP_ICONS.get(predicted_class.lower(), "🌱"),
            "probability": 1.0,
            "probability_pct": 100.0,
        }]

    return {
        "predicted_crop": predicted_class,
        "crop_icon": CROP_ICONS.get(predicted_class.lower(), "🌱"),
        "confidence": confidence,
        "confidence_pct": round(confidence * 100, 2),
        "model_used": metadata.get("model_name", "Random Forest"),
        "feature_set": metadata.get("feature_set", "Original Features (7)"),
        "top_recommendations": top_recommendations,
        "all_probabilities": all_probabilities,
    }


def simulate_what_if_scenario(
    base_features: Dict[str, float],
    feature_to_vary: str,
    min_val: float,
    max_val: float,
    steps: int = 10,
) -> List[Dict[str, Any]]:
    if feature_to_vary not in EXPECTED_RANGES:
        raise ValueError(f"Unsupported feature to vary: {feature_to_vary!r}.")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer.")
    if not np.isfinite(min_val) or not np.isfinite(max_val) or min_val > max_val:
        raise ValueError("Scenario bounds must be finite and min_val must not exceed max_val.")

    validation = validate_input_features(base_features)
    if not validation["is_valid"]:
        raise ValueError("Invalid base features: " + "; ".join(validation["errors"]))

    model = load_production_model()
    metadata = load_model_metadata()

    values = np.linspace(min_val, max_val, steps)
    results = []

    for v in values:
        test_inputs = base_features.copy()
        test_inputs[feature_to_vary] = round(float(v), 2)
        pred_res = predict_crop_suitability(test_inputs, model=model, metadata=metadata)
        results.append({
            "varied_feature": feature_to_vary,
            "value": round(float(v), 2),
            "predicted_crop": pred_res["predicted_crop"],
            "crop_icon": pred_res["crop_icon"],
            "confidence_pct": pred_res["confidence_pct"],
        })

    return results
