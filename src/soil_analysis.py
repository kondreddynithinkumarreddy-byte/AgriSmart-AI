"""
AgriSmart AI - Soil Analysis Module
"""

from typing import Dict, Any


def get_nutrient_status_label(value: float, q25: float, median: float, q75: float, max_val: float) -> str:
    if value < q25 * 0.7:
        return "Deficient"
    elif value < q25:
        return "Moderate"
    elif value <= q75:
        return "Optimal"
    elif value <= q75 * 1.3:
        return "High"
    else:
        return "Excessive"


def get_ph_classification(ph: float) -> Dict[str, str]:
    if ph < 4.5:
        return {
            "classification": "Extremely Acidic",
            "risk": "Severe aluminum/manganese toxicity; phosphorus fixation.",
            "recommendation": "Apply agricultural lime (calcium carbonate) or dolomite.",
            "color": "#e74c3c"
        }
    elif ph < 5.5:
        return {
            "classification": "Strongly Acidic",
            "risk": "Reduced availability of phosphorus, calcium, and magnesium.",
            "recommendation": "Incorporate ground limestone or wood ash.",
            "color": "#e67e22"
        }
    elif ph < 6.5:
        return {
            "classification": "Slightly Acidic",
            "risk": "Generally favorable for most agricultural crops.",
            "recommendation": "Maintain organic matter with compost or manure.",
            "color": "#27ae60"
        }
    elif ph <= 7.5:
        return {
            "classification": "Neutral",
            "risk": "Optimal nutrient bioavailability for the majority of crops.",
            "recommendation": "Ideal soil reaction. Standard balanced fertilization.",
            "color": "#2ecc71"
        }
    elif ph <= 8.5:
        return {
            "classification": "Moderately Alkaline",
            "risk": "Reduced iron, zinc, and manganese solubility; chlorosis risk.",
            "recommendation": "Apply elemental sulfur, gypsum, or acidifying fertilizers.",
            "color": "#f39c12"
        }
    else:
        return {
            "classification": "Strongly Alkaline / Calcareous",
            "risk": "Severe micronutrient lockup; potential sodicity hazards.",
            "recommendation": "Gypsum application with heavy leaching; sulfur amendment.",
            "color": "#c0392b"
        }


def analyze_soil_nutrients(n: float, p: float, k: float, ph: float) -> Dict[str, Any]:
    total_npk = n + p + k
    base_val = max(min(n, p, k), 1.0)
    ratio_str = f"{round(n / base_val, 1)} : {round(p / base_val, 1)} : {round(k / base_val, 1)}"

    n_status = get_nutrient_status_label(n, 21.0, 37.0, 84.0, 140.0)
    p_status = get_nutrient_status_label(p, 28.0, 51.0, 68.0, 145.0)
    k_status = get_nutrient_status_label(k, 20.0, 32.0, 49.0, 205.0)

    ph_info = get_ph_classification(ph)

    fertility_score = 0
    fertility_score += 25 if n_status == "Optimal" else (15 if n_status in ["Moderate", "High"] else 8)
    fertility_score += 25 if p_status == "Optimal" else (15 if p_status in ["Moderate", "High"] else 8)
    fertility_score += 25 if k_status == "Optimal" else (15 if k_status in ["Moderate", "High"] else 8)
    fertility_score += 25 if ph_info["classification"] in ["Neutral", "Slightly Acidic"] else (15 if "Acidic" in ph_info["classification"] or "Alkaline" in ph_info["classification"] else 5)

    return {
        "npk_total": round(total_npk, 2),
        "npk_average": round(total_npk / 3.0, 2),
        "npk_ratio": ratio_str,
        "fertility_score": min(fertility_score, 100),
        "nitrogen": {
            "value": round(n, 2),
            "status": n_status,
            "role": "Vegetative shoot development, leaf canopy expansion, chlorophyll synthesis.",
        },
        "phosphorus": {
            "value": round(p, 2),
            "status": p_status,
            "role": "Root architecture, cellular ATP energy transfer, flowering, seed set.",
        },
        "potassium": {
            "value": round(k, 2),
            "status": k_status,
            "role": "Stomatal osmoregulation, disease resistance, drought tolerance, enzyme activation.",
        },
        "ph_analysis": ph_info,
    }
