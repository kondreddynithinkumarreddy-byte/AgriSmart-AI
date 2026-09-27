"""
AgriSmart AI - Feature Engineering Module
"""

import os
import sys
import pandas as pd
from typing import List, Dict, Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

ORIGINAL_FEATURES: List[str] = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
DERIVED_FEATURES: List[str] = [
    "NPK_total",
    "NPK_average",
    "N_P_ratio",
    "N_K_ratio",
    "P_K_ratio",
    "temperature_humidity_interaction",
]
ENGINEERED_ALL_FEATURES: List[str] = ORIGINAL_FEATURES + DERIVED_FEATURES


def safe_divide(numerator: pd.Series, denominator: pd.Series, epsilon: float = 1e-5) -> pd.Series:
    safe_denom = denominator.replace(0, epsilon)
    result = numerator / safe_denom
    return result.fillna(0.0)


def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df_eng = df.copy()
    df_eng["NPK_total"] = df_eng["N"] + df_eng["P"] + df_eng["K"]
    df_eng["NPK_average"] = df_eng["NPK_total"] / 3.0
    df_eng["N_P_ratio"] = safe_divide(df_eng["N"], df_eng["P"])
    df_eng["N_K_ratio"] = safe_divide(df_eng["N"], df_eng["K"])
    df_eng["P_K_ratio"] = safe_divide(df_eng["P"], df_eng["K"])
    df_eng["temperature_humidity_interaction"] = (df_eng["temperature"] * df_eng["humidity"]) / 100.0
    return df_eng


def save_engineered_dataset(df_raw: pd.DataFrame, output_path: str = None) -> str:
    if output_path is None:
        output_path = os.path.join(PROJECT_ROOT, "data", "Crop_recommendation_engineered.csv")
    df_eng = create_engineered_features(df_raw)
    df_eng.to_csv(output_path, index=False)
    return output_path


def get_feature_engineering_summary() -> Dict[str, Any]:
    return {
        "original_features": ORIGINAL_FEATURES,
        "derived_features": DERIVED_FEATURES,
        "total_engineered_count": len(ENGINEERED_ALL_FEATURES),
        "formulations": {
            "NPK_total": "N + P + K",
            "NPK_average": "(N + P + K) / 3",
            "N_P_ratio": "N / P",
            "N_K_ratio": "N / K",
            "P_K_ratio": "P / K",
            "temperature_humidity_interaction": "(Temperature * Humidity) / 100",
        }
    }
