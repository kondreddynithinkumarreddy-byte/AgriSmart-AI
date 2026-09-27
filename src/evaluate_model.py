"""
AgriSmart AI - Model Evaluation Module
"""

import numpy as np
from typing import Dict, Any, List
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)


def evaluate_model_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
    classes: List[str] = None,
) -> Dict[str, Any]:
    acc = float(accuracy_score(y_true, y_pred))
    prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=classes).tolist() if classes is not None else confusion_matrix(y_true, y_pred).tolist()

    roc_auc_val = None
    if y_prob is not None and classes is not None:
        try:
            roc_auc_val = float(
                roc_auc_score(
                    y_true,
                    y_prob,
                    multi_class="ovr",
                    average="weighted",
                    labels=classes,
                )
            )
        except Exception:
            roc_auc_val = None

    return {
        "accuracy": round(acc, 4),
        "accuracy_pct": round(acc * 100, 2),
        "precision_weighted": round(prec_weighted, 4),
        "precision_macro": round(prec_macro, 4),
        "recall_weighted": round(rec_weighted, 4),
        "recall_macro": round(rec_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_pct": round(f1_weighted * 100, 2),
        "roc_auc_ovr": round(roc_auc_val, 4) if roc_auc_val is not None else None,
        "confusion_matrix": cm,
    }
