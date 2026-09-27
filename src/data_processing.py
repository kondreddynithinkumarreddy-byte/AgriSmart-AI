"""
AgriSmart AI - Data Processing & Exploration Module
"""

import os
import sys
import pandas as pd
from typing import Tuple, Dict, Any, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

NUMERICAL_FEATURES: List[str] = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET_COLUMN: str = "label"
ALL_COLUMNS: List[str] = NUMERICAL_FEATURES + [TARGET_COLUMN]


def get_dataset_path() -> str:
    return os.path.join(PROJECT_ROOT, "data", "Crop_recommendation.csv")


def load_and_validate_dataset(csv_path: str = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if csv_path is None:
        csv_path = get_dataset_path()

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at expected path: {csv_path}")

    df = pd.read_csv(csv_path)

    missing_cols = [c for c in ALL_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    for col in NUMERICAL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    total_nulls = int(df.isnull().sum().sum())
    total_duplicates = int(df.duplicated().sum())
    unique_classes = sorted(df[TARGET_COLUMN].dropna().unique().tolist())
    class_counts = df[TARGET_COLUMN].value_counts().to_dict()

    summary = {
        "total_rows": len(df),
        "total_columns": df.shape[1],
        "features": NUMERICAL_FEATURES,
        "target": TARGET_COLUMN,
        "missing_values": total_nulls,
        "duplicate_rows": total_duplicates,
        "class_count": len(unique_classes),
        "unique_classes": unique_classes,
        "class_distribution": class_counts,
        "is_balanced": len(set(class_counts.values())) == 1 if class_counts else False,
    }

    return df, summary


def get_dataset_overview(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "features_count": len(NUMERICAL_FEATURES),
        "classes_count": df[TARGET_COLUMN].nunique(),
        "missing_count": int(df.isnull().sum().sum()),
        "duplicate_count": int(df.duplicated().sum()),
        "memory_usage_kb": round(df.memory_usage().sum() / 1024, 2),
    }


def get_dataset_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    stats = df[NUMERICAL_FEATURES].describe().T
    stats["median"] = df[NUMERICAL_FEATURES].median()
    stats["iqr"] = stats["75%"] - stats["25%"]
    stats["skew"] = df[NUMERICAL_FEATURES].skew()
    return stats[["count", "mean", "std", "min", "25%", "median", "75%", "max", "iqr", "skew"]]


def detect_outliers_iqr(df: pd.DataFrame, factor: float = 1.5) -> Dict[str, Dict[str, Any]]:
    outlier_info = {}
    for col in NUMERICAL_FEATURES:
        q25 = float(df[col].quantile(0.25))
        q75 = float(df[col].quantile(0.75))
        iqr = q75 - q25
        lower_bound = q25 - (factor * iqr)
        upper_bound = q75 + (factor * iqr)
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_info[col] = {
            "q25": round(q25, 2),
            "q75": round(q75, 2),
            "iqr": round(iqr, 2),
            "lower_bound": round(lower_bound, 2),
            "upper_bound": round(upper_bound, 2),
            "outlier_count": len(outliers),
            "outlier_pct": round((len(outliers) / len(df)) * 100, 2),
            "min_val": round(float(df[col].min()), 2),
            "max_val": round(float(df[col].max()), 2),
        }
    return outlier_info


def get_crop_profiles(df: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, float]]]:
    profiles = {}
    grouped = df.groupby(TARGET_COLUMN)
    
    for crop_name, group in grouped:
        profiles[str(crop_name).lower()] = {
            col: {
                "mean": round(float(group[col].mean()), 2),
                "std": round(float(group[col].std()), 2),
                "min": round(float(group[col].min()), 2),
                "max": round(float(group[col].max()), 2),
                "median": round(float(group[col].median()), 2),
                "q25": round(float(group[col].quantile(0.25)), 2),
                "q75": round(float(group[col].quantile(0.75)), 2),
            }
            for col in NUMERICAL_FEATURES
        }
    return profiles
