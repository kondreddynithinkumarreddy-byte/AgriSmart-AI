"""
AgriSmart AI - Environmental & Microclimate Analysis Module
"""

from typing import Dict, Any


def get_climate_regime(temperature: float, humidity: float, rainfall: float) -> Dict[str, str]:
    if rainfall >= 180 and humidity >= 70:
        return {
            "classification": "Tropical Wet / Humid",
            "description": "High precipitation with prolonged high atmospheric moisture; optimal for wetland crops.",
            "crops_suited": "Rice, Jute, Coconut, Banana",
            "color": "#1abc9c"
        }
    elif rainfall < 80 and humidity < 50:
        return {
            "classification": "Semi-Arid / Dryland",
            "description": "Low precipitation and low humidity; requires drought-tolerant or deep-rooting cultivars.",
            "crops_suited": "Chickpea, Mothbeans, Mungbean, Lentil",
            "color": "#e67e22"
        }
    elif temperature < 20 and rainfall > 70:
        return {
            "classification": "Temperate / Sub-Tropical Highland",
            "description": "Cooler ambient temperatures with moderate rainfall; favors pomological crops and winter pulses.",
            "crops_suited": "Apple, Grapes, Kidneybeans, Lentil",
            "color": "#3498db"
        }
    elif temperature >= 28 and rainfall >= 80:
        return {
            "classification": "Tropical Warm / Sub-Humid",
            "description": "Warm thermal regime with adequate moisture; high photosynthetic potential.",
            "crops_suited": "Maize, Cotton, Papaya, Mango",
            "color": "#f39c12"
        }
    else:
        return {
            "classification": "Moderate / Transitional Agricultural",
            "description": "Balanced climatic conditions suitable for diversified multi-cropping.",
            "crops_suited": "Pigeonpeas, Blackgram, Pomegranate, Orange",
            "color": "#2ecc71"
        }


def analyze_environmental_conditions(
    temperature: float,
    humidity: float,
    rainfall: float,
    predicted_crop: str = None
) -> Dict[str, Any]:
    climate_info = get_climate_regime(temperature, humidity, rainfall)

    heat_stress = temperature > 35.0
    cold_stress = temperature < 15.0
    drought_stress = rainfall < 60.0 and humidity < 40.0
    waterlogging_risk = rainfall > 220.0
    fungal_disease_risk = humidity > 85.0 and temperature > 22.0

    stresses = []
    if heat_stress:
        stresses.append("Elevated thermal load (heat stress) may cause pollen sterility or excessive transpiration.")
    if cold_stress:
        stresses.append("Low temperature (chilling stress) may delay vegetative germination and flowering.")
    if drought_stress:
        stresses.append("Deficit moisture regime: Supplementary drip or furrow irrigation strongly advised.")
    if waterlogging_risk:
        stresses.append("Excessive precipitation: Ensure active field drainage to prevent root hypoxia.")
    if fungal_disease_risk:
        stresses.append("High humidity and warmth increase foliar fungal pathogen pressure; monitor canopy airflow.")

    score = 100
    if heat_stress or cold_stress:
        score -= 20
    if drought_stress or waterlogging_risk:
        score -= 20
    if fungal_disease_risk:
        score -= 10

    return {
        "temperature": round(temperature, 2),
        "humidity": round(humidity, 2),
        "rainfall": round(rainfall, 2),
        "climate_regime": climate_info,
        "microclimate_index": max(score, 30),
        "stress_analysis": {
            "heat_stress": heat_stress,
            "cold_stress": cold_stress,
            "drought_stress": drought_stress,
            "waterlogging_risk": waterlogging_risk,
            "fungal_disease_risk": fungal_disease_risk,
            "active_stress_count": len(stresses),
            "recommendations": stresses if stresses else ["Climatic conditions are stable with low abiotic risk."],
        },
        "suitability_assessment": {
            "overall_suitability": "Optimal" if score >= 80 else ("Sub-Optimal" if score >= 60 else "Challenging"),
            "predicted_crop_context": f"Environmental regime aligns with the ecological niche of {predicted_crop.capitalize() if predicted_crop else 'target crops'}."
        }
    }
