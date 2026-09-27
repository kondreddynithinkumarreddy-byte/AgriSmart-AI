"""
AgriSmart AI – Intelligent Crop Suitability & Soil Analysis System
===================================================================
A polished, user-facing agricultural decision-support application.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Setup dynamic project root path (relative, cross-platform)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data_processing import (
    load_and_validate_dataset,
    get_dataset_summary_statistics,
    detect_outliers_iqr,
    get_crop_profiles,
    get_dataset_overview,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
)
from src.feature_engineering import create_engineered_features
from src.prediction import (
    load_production_model,
    load_model_metadata,
    predict_crop_suitability,
    validate_input_features,
    simulate_what_if_scenario,
    EXPECTED_RANGES,
    CROP_ICONS,
)
from src.soil_analysis import analyze_soil_nutrients
from src.environmental_analysis import analyze_environmental_conditions
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
    clear_history,
)
from src.report_generator import generate_pdf_report

# ==============================================================================
# STREAMLIT CONFIGURATION & POLISHED AGRICULTURAL THEME
# ==============================================================================
st.set_page_config(
    page_title="AgriSmart AI – Crop & Soil Intelligence",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Theme CSS
st.markdown("""
<style>
    /* Global Typography & Palette */
    :root {
        --primary: #1b5e20;
        --primary-light: #2e7d32;
        --accent: #e8f5e9;
        --text-main: #0f172a;
        --text-muted: #475569;
        --border-color: #e2e8f0;
    }
    
    .app-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 0.25rem;
    }
    .app-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1b5e20;
        margin: 0;
    }
    .app-subtitle {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
        line-height: 1.5;
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 60%, #388e3c 100%);
        color: white;
        padding: 2.2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(27, 94, 32, 0.25);
    }
    .hero-banner h1 {
        font-size: 2.4rem;
        font-weight: 800;
        color: white !important;
        margin-bottom: 0.5rem;
    }
    .hero-banner p {
        font-size: 1.15rem;
        color: #e8f5e9;
        margin-bottom: 1.5rem;
        max-width: 800px;
        line-height: 1.6;
    }

    /* Result Card */
    .prediction-hero {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 8px 20px -4px rgba(27, 94, 32, 0.3);
    }
    .prediction-hero .label-text {
        font-size: 1rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #c8e6c9;
        font-weight: 600;
    }
    .prediction-hero .crop-name {
        font-size: 3.5rem;
        font-weight: 900;
        margin: 0.4rem 0;
        color: white;
    }
    .prediction-hero .confidence-badge {
        font-size: 1.2rem;
        background: rgba(255, 255, 255, 0.22);
        padding: 6px 20px;
        border-radius: 30px;
        display: inline-block;
        font-weight: 700;
    }

    /* Product Feature Cards */
    .product-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.04);
        height: 100%;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .product-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.08);
        border-color: #cbd5e1;
    }
    .product-card .card-icon {
        font-size: 2rem;
        margin-bottom: 0.75rem;
    }
    .product-card .card-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1b5e20;
        margin-bottom: 0.5rem;
    }
    .product-card .card-desc {
        font-size: 0.95rem;
        color: #475569;
        line-height: 1.5;
    }

    /* Compact Diagnostic Summary Box */
    .summary-box {
        background: #f8fafc;
        border-left: 4px solid #2e7d32;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }
    .summary-box-title {
        font-size: 0.9rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #1b5e20;
        margin-bottom: 0.25rem;
    }
    .summary-box-val {
        font-size: 1.1rem;
        font-weight: 600;
        color: #0f172a;
    }

    /* Buttons */
    .stButton>button {
        background-color: #2e7d32;
        color: white;
        font-weight: 700;
        font-size: 1rem;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        border: none;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# CACHED DATA & MODEL LOADERS
# ==============================================================================
@st.cache_data
def get_cached_dataset():
    df, summary = load_and_validate_dataset()
    return df, summary


@st.cache_resource
def get_cached_model():
    model = load_production_model()
    metadata = load_model_metadata()
    return model, metadata


# Initialize SQLite
init_db()

# Load authoritative resources
df_raw, data_summary = get_cached_dataset()
model, model_metadata = get_cached_model()

# Calculate dataset medians for sensible defaults
DATASET_MEDIANS = {
    "N": float(df_raw["N"].median()),
    "P": float(df_raw["P"].median()),
    "K": float(df_raw["K"].median()),
    "temperature": float(df_raw["temperature"].median()),
    "humidity": float(df_raw["humidity"].median()),
    "ph": float(df_raw["ph"].median()),
    "rainfall": float(df_raw["rainfall"].median()),
}

# Calculate dataset ranges
DATASET_RANGES = {
    col: (float(df_raw[col].min()), float(df_raw[col].max()))
    for col in NUMERICAL_FEATURES
}

# Session State for navigation & persistence
if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = "🏠 Home"

if "current_inputs" not in st.session_state:
    st.session_state["current_inputs"] = DATASET_MEDIANS.copy()

if "last_analysis" not in st.session_state:
    st.session_state["last_analysis"] = None


# ==============================================================================
# SIDEBAR NAVIGATION (STREAMLINED 7 CORE PAGES)
# ==============================================================================
st.sidebar.markdown("## 🌱 **AgriSmart AI**")
st.sidebar.caption("Intelligent Crop Suitability & Soil System")
st.sidebar.markdown("---")

NAV_OPTIONS = [
    "🏠 Home",
    "🌱 Crop Analysis",
    "🧪 Soil Analysis",
    "🌤 Environmental Analysis",
    "📊 Insights",
    "📜 History",
    "ℹ️ About",
]

# Synchronize sidebar selection with session state
selected_page = st.sidebar.radio(
    "Navigation",
    NAV_OPTIONS,
    index=NAV_OPTIONS.index(st.session_state["nav_page"]) if st.session_state["nav_page"] in NAV_OPTIONS else 0,
)

if selected_page != st.session_state["nav_page"]:
    st.session_state["nav_page"] = selected_page
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info(
    f"**Engine:** `{model_metadata.get('model_name', 'Random Forest')}`\n\n"
    f"**Accuracy:** `{model_metadata['metrics']['accuracy']*100:.1f}%`\n\n"
    f"**Saved Analyses:** `{get_prediction_count()}`"
)


# ==============================================================================
# PAGE 1: 🏠 HOME (USER-FOCUSED PRODUCT DASHBOARD)
# ==============================================================================
if st.session_state["nav_page"] == "🏠 Home":
    # Hero Section
    st.markdown("""
    <div class="hero-banner">
        <h1>🌱 AgriSmart AI</h1>
        <p><b>Intelligent Crop Suitability & Soil Analysis</b><br>
        Analyze your soil chemistry and meteorological conditions using trained machine learning to identify optimal crop varieties and understand the factors driving each recommendation.</p>
    </div>
    """, unsafe_allow_html=True)

    col_btn, _ = st.columns([1, 2])
    with col_btn:
        if st.button("🚀 Start Crop Analysis →", use_container_width=True):
            st.session_state["nav_page"] = "🌱 Crop Analysis"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # 4 Clean Product Feature Cards
    st.markdown("### Core Capabilities")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="product-card">
            <div class="card-icon">🌾</div>
            <div class="card-title">Crop Prediction</div>
            <div class="card-desc">Predict suitable crop varieties from soil macronutrients and climate parameters with confidence scoring.</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="product-card">
            <div class="card-icon">🧪</div>
            <div class="card-title">Soil Analysis</div>
            <div class="card-desc">Understand N, P, K stoichiometry, nutrient tiers, soil pH reaction, and tailored amendment guidelines.</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="product-card">
            <div class="card-icon">🌤</div>
            <div class="card-title">Environmental Analysis</div>
            <div class="card-desc">Analyze temperature, humidity, and rainfall regimes to diagnose potential abiotic stresses.</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="product-card">
            <div class="card-icon">📊</div>
            <div class="card-title">Smart Insights</div>
            <div class="card-desc">Explore feature importance, underlying patterns, model performance, and historical audits.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Quick Demo Presets
    st.markdown("### ⚡ Quick Presets")
    st.write("Load sample regional conditions to test immediate inference:")

    colP1, colP2, colP3, colP4 = st.columns(4)
    with colP1:
        if st.button("🌾 Rice Profile (Wetland)", use_container_width=True):
            st.session_state["current_inputs"] = {"N": 90.0, "P": 42.0, "K": 43.0, "temperature": 20.87, "humidity": 82.0, "ph": 6.5, "rainfall": 202.9}
            st.session_state["nav_page"] = "🌱 Crop Analysis"
            st.rerun()
    with colP2:
        if st.button("🧆 Chickpea Profile (Dryland)", use_container_width=True):
            st.session_state["current_inputs"] = {"N": 40.0, "P": 72.0, "K": 77.0, "temperature": 18.8, "humidity": 16.8, "ph": 7.3, "rainfall": 80.0}
            st.session_state["nav_page"] = "🌱 Crop Analysis"
            st.rerun()
    with colP3:
        if st.button("🍏 Apple Profile (Highland)", use_container_width=True):
            st.session_state["current_inputs"] = {"N": 20.0, "P": 134.0, "K": 199.0, "temperature": 22.5, "humidity": 92.3, "ph": 5.9, "rainfall": 110.0}
            st.session_state["nav_page"] = "🌱 Crop Analysis"
            st.rerun()
    with colP4:
        if st.button("👕 Cotton Profile (Warm)", use_container_width=True):
            st.session_state["current_inputs"] = {"N": 120.0, "P": 40.0, "K": 20.0, "temperature": 24.0, "humidity": 80.0, "ph": 6.8, "rainfall": 80.0}
            st.session_state["nav_page"] = "🌱 Crop Analysis"
            st.rerun()


# ==============================================================================
# PAGE 2: 🌱 CROP ANALYSIS (CORE PREDICTION & ANALYSIS EXPERIENCE)
# ==============================================================================
elif st.session_state["nav_page"] == "🌱 Crop Analysis":
    st.markdown('<div class="app-title">Crop Suitability Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Enter the soil and environmental conditions below. AgriSmart AI will analyze the values using the trained machine-learning model.</div>', unsafe_allow_html=True)

    inputs = st.session_state["current_inputs"]

    # Input Form
    with st.form("crop_analysis_form"):
        col_soil, col_env = st.columns(2)

        with col_soil:
            st.markdown("#### 🧪 Soil Conditions")
            
            n_val = st.number_input(
                f"Nitrogen (N) [kg/ha] — Observed range: {DATASET_RANGES['N'][0]:.0f} to {DATASET_RANGES['N'][1]:.0f}",
                min_value=0.0, max_value=250.0, value=float(inputs.get("N", DATASET_MEDIANS["N"])), step=1.0,
            )
            p_val = st.number_input(
                f"Phosphorus (P) [kg/ha] — Observed range: {DATASET_RANGES['P'][0]:.0f} to {DATASET_RANGES['P'][1]:.0f}",
                min_value=0.0, max_value=250.0, value=float(inputs.get("P", DATASET_MEDIANS["P"])), step=1.0,
            )
            k_val = st.number_input(
                f"Potassium (K) [kg/ha] — Observed range: {DATASET_RANGES['K'][0]:.0f} to {DATASET_RANGES['K'][1]:.0f}",
                min_value=0.0, max_value=300.0, value=float(inputs.get("K", DATASET_MEDIANS["K"])), step=1.0,
            )
            ph_val = st.number_input(
                f"Soil pH — Observed range: {DATASET_RANGES['ph'][0]:.1f} to {DATASET_RANGES['ph'][1]:.1f}",
                min_value=0.0, max_value=14.0, value=float(inputs.get("ph", DATASET_MEDIANS["ph"])), step=0.1,
            )

        with col_env:
            st.markdown("#### 🌤 Environmental Conditions")
            
            t_val = st.number_input(
                f"Temperature (°C) — Observed range: {DATASET_RANGES['temperature'][0]:.1f} to {DATASET_RANGES['temperature'][1]:.1f}",
                min_value=0.0, max_value=60.0, value=float(inputs.get("temperature", DATASET_MEDIANS["temperature"])), step=0.1,
            )
            h_val = st.number_input(
                f"Humidity (%) — Observed range: {DATASET_RANGES['humidity'][0]:.1f} to {DATASET_RANGES['humidity'][1]:.1f}",
                min_value=0.0, max_value=100.0, value=float(inputs.get("humidity", DATASET_MEDIANS["humidity"])), step=0.1,
            )
            r_val = st.number_input(
                f"Rainfall (mm) — Observed range: {DATASET_RANGES['rainfall'][0]:.1f} to {DATASET_RANGES['rainfall'][1]:.1f}",
                min_value=0.0, max_value=600.0, value=float(inputs.get("rainfall", DATASET_MEDIANS["rainfall"])), step=1.0,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.form_submit_button("🌱 Analyze & Predict", use_container_width=True)

    if analyze_btn:
        user_input_dict = {
            "N": n_val,
            "P": p_val,
            "K": k_val,
            "temperature": t_val,
            "humidity": h_val,
            "ph": ph_val,
            "rainfall": r_val,
        }
        st.session_state["current_inputs"] = user_input_dict

        # Validate
        val_res = validate_input_features(user_input_dict)
        if not val_res["is_valid"]:
            for err in val_res["errors"]:
                st.error(f"❌ {err}")
        else:
            if val_res["warnings"]:
                for warn in val_res["warnings"]:
                    st.warning(f"⚠️ {warn}")

            # Run Real Model Prediction
            pred_res = predict_crop_suitability(user_input_dict, model=model, metadata=model_metadata)
            soil_res = analyze_soil_nutrients(n_val, p_val, k_val, ph_val)
            env_res = analyze_environmental_conditions(t_val, h_val, r_val, pred_res["predicted_crop"])

            # Auto-save to SQLite
            log_id = log_prediction(user_input_dict, pred_res, notes="Web Dashboard Analysis")

            # Store in session state
            st.session_state["last_analysis"] = {
                "inputs": user_input_dict,
                "prediction": pred_res,
                "soil": soil_res,
                "env": env_res,
                "log_id": log_id,
            }

    # Display Result Screen if analysis is available
    if st.session_state["last_analysis"] is not None:
        analysis = st.session_state["last_analysis"]
        p_res = analysis["prediction"]
        s_res = analysis["soil"]
        e_res = analysis["env"]
        in_vals = analysis["inputs"]

        st.markdown("---")

        # 1. Prediction Hero Card
        st.markdown(f"""
        <div class="prediction-hero">
            <div class="label-text">Recommended Crop</div>
            <div class="crop-name">{p_res['crop_icon']} {p_res['predicted_crop'].upper()}</div>
            <div class="confidence-badge">Prediction Confidence: {p_res['confidence_pct']}%</div>
        </div>
        """, unsafe_allow_html=True)

        # 2. Result Summary (3 compact cards)
        st.markdown("### Analysis Summary")
        colS1, colS2, colS3 = st.columns(3)
        with colS1:
            st.markdown(f"""
            <div class="summary-box">
                <div class="summary-box-title">🌱 Crop Recommendation</div>
                <div class="summary-box-val">{p_res['predicted_crop'].capitalize()} ({p_res['confidence_pct']}%)</div>
            </div>
            """, unsafe_allow_html=True)
        with colS2:
            st.markdown(f"""
            <div class="summary-box">
                <div class="summary-box-title">🧪 Soil Status</div>
                <div class="summary-box-val">{s_res['ph_analysis']['classification']} (Score: {s_res['fertility_score']}/100)</div>
            </div>
            """, unsafe_allow_html=True)
        with colS3:
            st.markdown(f"""
            <div class="summary-box">
                <div class="summary-box-title">🌤 Climate Regime</div>
                <div class="summary-box-val">{e_res['climate_regime']['classification']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # 3. Top Recommendations & Soil Profile
        colTop, colSoilP = st.columns([3, 2])
        
        with colTop:
            st.markdown("### Other Suitable Crops")
            top5_df = pd.DataFrame(p_res["top_recommendations"])
            fig_top5 = px.bar(
                top5_df,
                x="probability_pct",
                y="crop",
                orientation="h",
                text="probability_pct",
                color="probability_pct",
                color_continuous_scale="greens",
                labels={"probability_pct": "Confidence (%)", "crop": "Crop Variety"},
            )
            fig_top5.update_layout(yaxis=dict(autorange="reversed"), showlegend=False, height=260, margin=dict(l=0, r=0, t=10, b=10))
            st.plotly_chart(fig_top5, use_container_width=True)

        with colSoilP:
            st.markdown("### Soil Profile")
            st.write(f"**Nitrogen (N):** `{in_vals['N']:.0f} kg/ha` — *{s_res['nitrogen']['status']}* (Median: {DATASET_MEDIANS['N']:.0f})")
            st.progress(min(int((in_vals["N"] / 140.0) * 100), 100))

            st.write(f"**Phosphorus (P):** `{in_vals['P']:.0f} kg/ha` — *{s_res['phosphorus']['status']}* (Median: {DATASET_MEDIANS['P']:.0f})")
            st.progress(min(int((in_vals["P"] / 145.0) * 100), 100))

            st.write(f"**Potassium (K):** `{in_vals['K']:.0f} kg/ha` — *{s_res['potassium']['status']}* (Median: {DATASET_MEDIANS['K']:.0f})")
            st.progress(min(int((in_vals["K"] / 205.0) * 100), 100))

            st.write(f"**Soil pH:** `{in_vals['ph']:.1f}` — *{s_res['ph_analysis']['classification']}* (Median: {DATASET_MEDIANS['ph']:.1f})")

        st.markdown("---")

        # 4. 🔍 Why this crop?
        st.markdown(f"### 🔍 Why was {p_res['predicted_crop'].capitalize()} predicted?")
        st.write(f"""
        The model identified **{p_res['predicted_crop'].capitalize()}** based on the combined pattern of the soil and environmental conditions provided.
        The prediction uses:
        • **Nitrogen:** {in_vals['N']:.1f} kg/ha  
        • **Phosphorus:** {in_vals['P']:.1f} kg/ha  
        • **Potassium:** {in_vals['K']:.1f} kg/ha  
        • **Temperature:** {in_vals['temperature']:.1f} °C  
        • **Humidity:** {in_vals['humidity']:.1f} %  
        • **Soil pH:** {in_vals['ph']:.2f}  
        • **Rainfall:** {in_vals['rainfall']:.1f} mm  
        """)

        comp_df = generate_crop_comparison_explanation(in_vals, p_res["predicted_crop"])
        st.dataframe(comp_df, use_container_width=True)

        # 5. Advanced What-If Simulation inside Expander
        with st.expander("🧪 Advanced Analysis (What-If Scenario Simulator)"):
            st.write("Modify one parameter to test how the model's recommendation responds in real time:")
            colW1, colW2 = st.columns([1, 2])
            with colW1:
                feat_vary = st.selectbox("Parameter to vary", NUMERICAL_FEATURES, index=6)
                b_range = DATASET_RANGES[feat_vary]
                new_v = st.slider(f"Test new {feat_vary}", min_value=float(b_range[0]), max_value=float(b_range[1]), value=float(in_vals[feat_vary]))
            with colW2:
                perturbed = in_vals.copy()
                perturbed[feat_vary] = new_v
                test_pred = predict_crop_suitability(perturbed, model=model, metadata=model_metadata)
                st.write(f"**Original Prediction:** {p_res['crop_icon']} {p_res['predicted_crop'].capitalize()} ({p_res['confidence_pct']}%)")
                st.write(f"**New Prediction:** {test_pred['crop_icon']} {test_pred['predicted_crop'].capitalize()} ({test_pred['confidence_pct']}%)")
                if p_res["predicted_crop"] == test_pred["predicted_crop"]:
                    st.success("✅ Prediction remained stable.")
                else:
                    st.warning(f"⚠️ Recommendation shifted from {p_res['predicted_crop'].capitalize()} to {test_pred['predicted_crop'].capitalize()}.")

        # 6. PDF Report Download Button
        pdf_bytes = generate_pdf_report(in_vals, p_res, s_res, e_res)
        st.download_button(
            label="📄 Download Official PDF Report",
            data=pdf_bytes,
            file_name=f"AgriSmart_Analysis_{p_res['predicted_crop']}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


# ==============================================================================
# PAGE 3: 🧪 SOIL ANALYSIS
# ==============================================================================
elif st.session_state["nav_page"] == "🧪 Soil Analysis":
    st.markdown('<div class="app-title">Soil Nutrient & Chemistry Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Evaluate macronutrient stoichiometry (N:P:K), empirical nutrient tiers, and soil pH dynamics against training data.</div>', unsafe_allow_html=True)

    inputs = st.session_state["current_inputs"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        n_in = st.number_input("Nitrogen (N) [kg/ha]", 0.0, 250.0, float(inputs.get("N", DATASET_MEDIANS["N"])), key="s_n")
    with col2:
        p_in = st.number_input("Phosphorus (P) [kg/ha]", 0.0, 250.0, float(inputs.get("P", DATASET_MEDIANS["P"])), key="s_p")
    with col3:
        k_in = st.number_input("Potassium (K) [kg/ha]", 0.0, 300.0, float(inputs.get("K", DATASET_MEDIANS["K"])), key="s_k")
    with col4:
        ph_in = st.number_input("Soil pH", 0.0, 14.0, float(inputs.get("ph", DATASET_MEDIANS["ph"])), key="s_ph")

    soil_res = analyze_soil_nutrients(n_in, p_in, k_in, ph_in)

    st.markdown("---")
    colM1, colM2, colM3, colM4 = st.columns(4)
    with colM1:
        st.metric("Total Macronutrients", f"{soil_res['npk_total']} kg/ha")
    with colM2:
        st.metric("N:P:K Ratio", f"{soil_res['npk_ratio']}")
    with colM3:
        st.metric("Soil Reaction", f"{soil_res['ph_analysis']['classification']}")
    with colM4:
        st.metric("Fertility Health Score", f"{soil_res['fertility_score']}/100")

    st.markdown("---")
    colL, colR = st.columns(2)
    with colL:
        st.markdown("### NPK Profile Chart")
        npk_df = pd.DataFrame({
            "Nutrient": ["Nitrogen", "Phosphorus", "Potassium"],
            "Measured (kg/ha)": [n_in, p_in, k_in],
            "Dataset Median (kg/ha)": [DATASET_MEDIANS["N"], DATASET_MEDIANS["P"], DATASET_MEDIANS["K"]],
        })
        fig_npk = px.bar(
            npk_df,
            x="Nutrient",
            y=["Measured (kg/ha)", "Dataset Median (kg/ha)"],
            barmode="group",
            color_discrete_map={"Measured (kg/ha)": "#2e7d32", "Dataset Median (kg/ha)": "#94a3b8"},
        )
        fig_npk.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=10))
        st.plotly_chart(fig_npk, use_container_width=True)

    with colR:
        st.markdown("### Soil pH & Amendment Advisory")
        ph_info = soil_res["ph_analysis"]
        st.info(f"**Classification:** {ph_info['classification']}\n\n**Agronomic Context:** {ph_info['risk']}")
        st.success(f"**Actionable Advisory:** {ph_info['recommendation']}")


# ==============================================================================
# PAGE 4: 🌤 ENVIRONMENTAL ANALYSIS
# ==============================================================================
elif st.session_state["nav_page"] == "🌤 Environmental Analysis":
    st.markdown('<div class="app-title">Environmental & Climate Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Analyze ambient thermal regime, atmospheric relative humidity, and precipitation dynamics.</div>', unsafe_allow_html=True)

    inputs = st.session_state["current_inputs"]

    col1, col2, col3 = st.columns(3)
    with col1:
        t_in = st.number_input("Temperature (°C)", 0.0, 50.0, float(inputs.get("temperature", DATASET_MEDIANS["temperature"])), key="e_t")
    with col2:
        h_in = st.number_input("Relative Humidity (%)", 0.0, 100.0, float(inputs.get("humidity", DATASET_MEDIANS["humidity"])), key="e_h")
    with col3:
        r_in = st.number_input("Precipitation / Rainfall (mm)", 0.0, 500.0, float(inputs.get("rainfall", DATASET_MEDIANS["rainfall"])), key="e_r")

    env_res = analyze_environmental_conditions(t_in, h_in, r_in)

    st.markdown("---")
    st.subheader(f"Agro-Climatic Regime: {env_res['climate_regime']['classification']}")
    st.info(f"**Regime Description:** {env_res['climate_regime']['description']}\n\n**Naturally Adapted Crops:** {env_res['climate_regime']['crops_suited']}")

    colE1, colE2 = st.columns(2)
    with colE1:
        st.markdown("### Meteorological Parameter Gauges")
        st.write(f"**Temperature:** `{t_in:.1f} °C` (Dataset Median: {DATASET_MEDIANS['temperature']:.1f} °C)")
        st.progress(min(int((t_in / 45.0) * 100), 100))

        st.write(f"**Humidity:** `{h_in:.1f} %` (Dataset Median: {DATASET_MEDIANS['humidity']:.1f} %)")
        st.progress(min(int((h_in / 100.0) * 100), 100))

        st.write(f"**Rainfall:** `{r_in:.1f} mm` (Dataset Median: {DATASET_MEDIANS['rainfall']:.1f} mm)")
        st.progress(min(int((r_in / 300.0) * 100), 100))

    with colE2:
        st.markdown("### Abiotic Stress & Suitability Diagnostics")
        st.metric("Microclimate Suitability Index", f"{env_res['microclimate_index']}/100")
        for rec in env_res["stress_analysis"]["recommendations"]:
            st.warning(f"⚠️ {rec}")


# ==============================================================================
# PAGE 5: 📊 INSIGHTS (CONSOLIDATED DATA & MODEL ANALYTICS)
# ==============================================================================
elif st.session_state["nav_page"] == "📊 Insights":
    st.markdown('<div class="app-title">Smart Insights & Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Understand dataset characteristics, feature relationships, and machine learning model benchmarks.</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["🌱 Crop Distribution", "📈 Feature Insights", "🔗 Relationships", "🤖 Model Performance"])

    with tab1:
        st.subheader("Crop Class Distribution")
        class_df = pd.DataFrame(list(data_summary["class_distribution"].items()), columns=["Crop", "Observations"])
        fig_class = px.bar(class_df, x="Crop", y="Observations", color="Observations", color_continuous_scale="greens")
        fig_class.update_layout(xaxis_tickangle=-45, height=350)
        st.plotly_chart(fig_class, use_container_width=True)
        st.caption("The training dataset contains 22 agricultural crop classes with exactly 100 empirical observations each.")

    with tab2:
        st.subheader("Feature Distributions")
        f_choice = st.selectbox("Select Feature to Inspect", NUMERICAL_FEATURES, index=6)
        fig_f = px.histogram(df_raw, x=f_choice, color=TARGET_COLUMN, nbins=30, title=f"Distribution of {f_choice} across Crops")
        st.plotly_chart(fig_f, use_container_width=True)

    with tab3:
        st.subheader("Pearson Correlation Heatmap")
        corr = df_raw[NUMERICAL_FEATURES].corr()
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="Viridis")
        fig_corr.update_layout(height=400)
        st.plotly_chart(fig_corr, use_container_width=True)

    with tab4:
        st.subheader("Model Benchmarking Results")
        comp_path = os.path.join(ROOT_DIR, "models", "model_comparison.json")
        if os.path.exists(comp_path):
            with open(comp_path, "r") as f:
                benchmarks = json.load(f)
            b_df = pd.DataFrame(benchmarks)[["config_name", "model_name", "feature_set", "test_accuracy", "test_f1_weighted", "cv_f1_mean", "cv_f1_std"]]
            st.dataframe(b_df, use_container_width=True)

            st.success(f"🏆 **Final Selected Model:** `{model_metadata['winning_config']}` with Test Accuracy: `{model_metadata['metrics']['accuracy']*100:.2f}%`.")

        with st.expander("🔍 View Multiclass Confusion Matrix (22 Classes)"):
            cm_data = np.array(model_metadata["confusion_matrix"])
            classes = model_metadata["classes"]
            fig_cm = px.imshow(cm_data, x=classes, y=classes, text_auto=True, color_continuous_scale="greens", labels=dict(x="Predicted", y="True"))
            fig_cm.update_layout(height=600, xaxis_tickangle=-45)
            st.plotly_chart(fig_cm, use_container_width=True)

        with st.expander("📋 View Raw Training Dataset Sample"):
            st.dataframe(df_raw, use_container_width=True)


# ==============================================================================
# PAGE 6: 📜 HISTORY
# ==============================================================================
elif st.session_state["nav_page"] == "📜 History":
    st.markdown('<div class="app-title">Prediction History</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Review previous crop recommendations and soil analyses logged in the local SQLite database.</div>', unsafe_allow_html=True)

    history_df = export_predictions_to_dataframe()

    if not history_df.empty:
        colH1, colH2, colH3 = st.columns(3)
        with colH1:
            st.metric("Total Analyses", f"{len(history_df)}")
        with colH2:
            st.metric("Top Recommended Crop", f"{history_df['predicted_crop'].mode()[0].capitalize()}")
        with colH3:
            st.metric("Average Confidence", f"{history_df['confidence'].mean()*100:.1f}%")

        st.markdown("---")
        
        # Display clean history table
        display_cols = ["id", "timestamp", "predicted_crop", "confidence", "n", "p", "k", "temperature", "humidity", "ph", "rainfall"]
        available_cols = [c for c in display_cols if c in history_df.columns]
        st.dataframe(history_df[available_cols], use_container_width=True)

        colD1, colD2 = st.columns([1, 1])
        with colD1:
            csv_bytes = history_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download History CSV",
                data=csv_bytes,
                file_name="agrismart_history.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with colD2:
            confirm_clear = st.checkbox("Confirm clearing history")
            if st.button("🗑️ Clear History", disabled=not confirm_clear, use_container_width=True):
                deleted_cnt = clear_history()
                st.success(f"Cleared {deleted_cnt} records from history.")
                st.rerun()
    else:
        st.info("No predictions logged yet. Run a prediction on the 'Crop Analysis' page to populate history.")


# ==============================================================================
# PAGE 7: ℹ️ ABOUT
# ==============================================================================
elif st.session_state["nav_page"] == "ℹ️ About":
    st.markdown('<div class="app-title">About AgriSmart AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Intelligent Crop Suitability Classification & Soil Analysis System</div>', unsafe_allow_html=True)

    st.markdown("""
    **AgriSmart AI** is an explainable machine learning decision-support application built for agricultural practitioners, agronomists, and researchers.

    It analyzes chemical soil parameters and meteorological indicators to predict optimal crop varieties:
    - **Macronutrients:** Nitrogen (N), Phosphorus (P), Potassium (K)
    - **Soil Reaction:** Soil pH
    - **Atmospheric Climate:** Ambient Temperature, Relative Humidity, Rainfall

    ---

    ### 🛠️ Technology Stack
    - **Language:** Python 3.10+
    - **Machine Learning:** Scikit-Learn (Random Forest, Decision Tree, KNN, Logistic Regression)
    - **Data Processing:** Pandas, NumPy
    - **Web Framework:** Streamlit
    - **Visualizations:** Plotly Express & Plotly Graph Objects
    - **Persistence:** SQLite3
    - **Reporting:** ReportLab PDF Engine

    ---

    ### 🎓 Machine Learning Methodology & Defense Guide

    #### 1. Why was Random Forest selected as the final production model?
    *Answer:* The dataset consists of 2,200 structured tabular observations across 7 physical parameters. Random Forest constructs orthogonal decision tree ensembles that match physical agronomic thresholds, minimizes variance through bagging, and achieved a **99.55% test accuracy** with 99.49% 5-fold cross-validation stability.

    #### 2. How was Data Leakage strictly prevented?
    *Answer:* All feature scalers (`StandardScaler`) are encapsulated within `sklearn.pipeline.Pipeline` objects and fitted solely on training splits during 5-Fold Stratified Cross-Validation. The target label was strictly isolated from input features.

    #### 3. Why were the original 7 features preferred over derived ratios?
    *Answer:* The 7 physical features represent orthogonal dimensions. Derived collinear ratios did not add mutual information, so the parsimonious 7-feature model was chosen in accordance with Occam's Razor.
    """)
