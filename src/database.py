"""
AgriSmart AI - Database Storage & History Module
"""

import os
import sqlite3
import json
from datetime import datetime
import pandas as pd
from typing import Dict, Any, List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def get_db_path() -> str:
    db_dir = os.path.join(PROJECT_ROOT, "database")
    os.makedirs(db_dir, exist_ok=True)
    return os.path.join(db_dir, "predictions.db")


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(get_db_path(), timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS prediction_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME NOT NULL,
        n REAL NOT NULL,
        p REAL NOT NULL,
        k REAL NOT NULL,
        temperature REAL NOT NULL,
        humidity REAL NOT NULL,
        ph REAL NOT NULL,
        rainfall REAL NOT NULL,
        predicted_crop TEXT NOT NULL,
        confidence REAL NOT NULL,
        model_used TEXT NOT NULL,
        top_recommendations TEXT,
        notes TEXT
    )
    """)

    conn.commit()
    conn.close()


def log_prediction(
    inputs: Dict[str, float],
    prediction_result: Dict[str, Any],
    notes: str = ""
) -> int:
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    top_json = json.dumps(prediction_result.get("top_recommendations", []))

    cursor.execute("""
    INSERT INTO prediction_history (
        timestamp, n, p, k, temperature, humidity, ph, rainfall,
        predicted_crop, confidence, model_used, top_recommendations, notes
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        now_iso,
        float(inputs.get("N", 0.0)),
        float(inputs.get("P", 0.0)),
        float(inputs.get("K", 0.0)),
        float(inputs.get("temperature", 0.0)),
        float(inputs.get("humidity", 0.0)),
        float(inputs.get("ph", 0.0)),
        float(inputs.get("rainfall", 0.0)),
        str(prediction_result.get("predicted_crop", "")),
        float(prediction_result.get("confidence", 0.0)),
        str(prediction_result.get("model_used", "Random Forest")),
        top_json,
        str(notes)
    ))

    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return record_id


def get_recent_predictions(limit: int = 50, crop_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    if crop_filter:
        cursor.execute(
            "SELECT * FROM prediction_history WHERE LOWER(predicted_crop) = LOWER(?) ORDER BY id DESC LIMIT ?",
            (crop_filter, limit)
        )
    else:
        cursor.execute("SELECT * FROM prediction_history ORDER BY id DESC LIMIT ?", (limit,))

    rows = cursor.fetchall()
    conn.close()

    results = []
    for r in rows:
        d = dict(r)
        try:
            d["top_recommendations"] = json.loads(d["top_recommendations"])
        except Exception:
            pass
        results.append(d)
    return results


def get_prediction_count() -> int:
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM prediction_history")
    count = cursor.fetchone()[0]
    conn.close()
    return int(count)


def export_predictions_to_dataframe() -> pd.DataFrame:
    init_db()
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM prediction_history ORDER BY id DESC", conn)
    conn.close()
    return df
