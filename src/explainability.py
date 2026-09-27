"""
AgriSmart AI - Explainable AI (XAI) Module
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_processing import load_and_validate_dataset, get_crop_profiles, NUMERICAL_FEATURES


def get_model_feature_importance(model, feature_names: List[str]) -> Dict[str, float]:
    classifier = model.named_steps["classifier"] if hasattr(model, "named_steps") else model

    if hasattr(classifier, "feature_importances_"):
        raw_importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        raw_importances = np.mean(np.abs(classifier.coef_), axis=0)
    else:
        raw_importances = np.ones(len(feature_names))

    total = np.sum(raw_importances)
    normalized = raw_importances / total if total > 0 else raw_importances

    importance_dict = {
        name: round(float(imp), 4)
        for name, imp in zip(feature_names, normalized)
    }

    return dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))


def generate_crop_comparison_explanation(
    user_inputs: Dict[str, float],
    predicted_crop: str
) -> pd.DataFrame:
    df_raw, _ = load_and_validate_dataset()
    profiles = get_crop_profiles(df_raw)
    
    crop_key = str(predicted_crop).lower()
    if crop_key not in profiles:
        crop_profile = {col: {"mean": float(df_raw[col].mean()), "min": float(df_raw[col].min()), "max": float(df_raw[col].max()), "std": float(df_raw[col].std())} for col in NUMERICAL_FEATURES}
    else:
        crop_profile = profiles[crop_key]

    rows = []
    for feature in NUMERICAL_FEATURES:
        user_val = user_inputs.get(feature, 0.0)
        p = crop_profile.get(feature, {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 1.0})
        mean_val = p["mean"]
        min_val = p["min"]
        max_val = p["max"]
        std_val = p["std"] if p["std"] > 0 else 1.0

        diff = user_val - mean_val
        z_score = diff / std_val

        if abs(z_score) <= 1.0:
            status = "Within Optimal Niche"
        elif z_score > 1.0:
            status = "Above Typical Average"
        else:
            status = "Below Typical Average"

        rows.append({
            "Feature": feature.upper() if feature in ["n", "p", "k", "ph"] else feature.capitalize(),
            "User Value": round(user_val, 2),
            f"{predicted_crop.capitalize()} Mean": round(mean_val, 2),
            "Benchmark Range": f"[{min_val:.1f} - {max_val:.1f}]",
            "Difference": f"{diff:+.2f}",
            "Alignment": status,
        })

    return pd.DataFrame(rows)
