"""
AgriSmart AI – Intelligent Crop Suitability & Soil Analysis System
===================================================================
Academic Project Title: An Explainable Machine Learning System for Crop
Suitability Classification and Soil Analysis
Author: AgriSmart AI Capstone Team
Framework: Streamlit & Scikit-Learn
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Setup dynamic project root path (relative, works cross-platform)
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
from src.feature_engineering import create_engineered_features, get_feature_engineering_summary
from src.prediction import (
    load_production_model,
    load_model_metadata,
    predict_crop_suitability,
    validate_input_features,
    simulate_what_if_scenario,
    EXPECTED_RANGES,
    CROP_ICONS,
)
from src.soil_analysis import (
    analyze_soil_nutrients,
    get_ph_classification,
    get_nutrient_status_label,
)
from src.environmental_analysis import (
    analyze_environmental_conditions,
    get_climate_regime,
)
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
)
from src.report_generator import generate_pdf_report

# ==============================================================================
# STREAMLIT CONFIGURATION & CUSTOM STYLING
# ==============================================================================
st.set_page_config(
    page_title="AgriSmart AI – Crop Suitability & Soil Analysis",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Enterprise-grade CSS
st.markdown("""
<style>
    /* Global Typography & Palette */
    :root {
        --primary-green: #1b5e20;
        --secondary-green: #2e7d32;
        --light-green: #e8f5e9;
        --accent-orange: #e65100;
        --card-bg: #ffffff;
        --text-dark: #1e293b;
    }
    
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1b5e20;
        margin-bottom: 0.2rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }
    
    /* Result Hero Card */
    .result-hero {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 50%, #388e3c 100%);
        color: #ffffff !important;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 10px 25px -5px rgba(27, 94, 32, 0.3);
    }
    .result-hero h3 {
        color: #e8f5e9 !important;
        font-size: 1.1rem;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .result-hero .crop-title {
        font-size: 3.5rem;
        font-weight: 900;
        margin: 0.5rem 0;
        color: #ffffff !important;
    }
    .result-hero .conf-badge {
        font-size: 1.25rem;
        background: rgba(255, 255, 255, 0.2);
        padding: 6px 18px;
        border-radius: 30px;
        display: inline-block;
        font-weight: 600;
    }

    /* Cards & Containers */
    .feature-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.08);
    }
    .card-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1b5e20;
        margin-bottom: 0.5rem;
    }
    
    /* Custom Metric Badges */
    .badge-optimal {
        background-color: #dcfce7;
        color: #15803d;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .badge-warning {
        background-color: #ffedd5;
        color: #c2410c;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
    }

    /* Button Styling */
    .stButton>button {
        background-color: #2e7d32;
        color: white;
        font-weight: 700;
        border-radius: 8px;
        padding: 0.6rem 1.8rem;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
        color: white;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# CACHED DATA & MODEL LOADERS
# ==============================================================================
@st.cache_data
def get_authoritative_data():
    df, summary = load_and_validate_dataset()
    return df, summary


@st.cache_resource
def get_production_model_pipeline():
    model = load_production_model()
    metadata = load_model_metadata()
    return model, metadata


# Initialize local SQLite database
init_db()

# Load Core Data & Models
df_raw, data_summary = get_authoritative_data()
model, model_metadata = get_production_model_pipeline()

# Session State Initialization for Cross-Page Persistence
if "last_inputs" not in st.session_state:
    # Set realistic dataset median defaults (N=37, P=51, K=32, Temp=25.6, Humidity=80.5, pH=6.4, Rainfall=95)
    st.session_state["last_inputs"] = {
        "N": 37.0,
        "P": 51.0,
        "K": 32.0,
        "temperature": 25.6,
        "humidity": 80.5,
        "ph": 6.4,
        "rainfall": 95.0,
    }

if "last_prediction" not in st.session_state:
    st.session_state["last_prediction"] = None


# ==============================================================================
# SIDEBAR NAVIGATION
# ==============================================================================
st.sidebar.image("https://images.unsplash.com/photo-1500937386664-56d1dfef3854?w=600&auto=format&fit=crop&q=80", use_container_width=True)
st.sidebar.markdown("## 🌱 **AgriSmart AI**")
st.sidebar.caption("Intelligent Crop Suitability & Soil System")

nav_page = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏠 Home",
        "🌾 Crop Prediction",
        "🧪 Soil Analysis",
        "🌤 Environmental Analysis",
        "🔍 Explainability",
        "🔬 What-If Analysis",
        "📜 Prediction History",
        "📊 Data Analytics",
        "🤖 Model Performance",
        "📄 Reports",
        "ℹ️ About",
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Champion Model:** `{model_metadata.get('model_name', 'Random Forest')}`")
st.sidebar.markdown(f"**Test Accuracy:** `{model_metadata['metrics']['accuracy']*100:.2f}%`")
st.sidebar.markdown(f"**Logged Predictions:** `{get_prediction_count()}`")


# ==============================================================================
# 1. 🏠 HOME PAGE (USER-FOCUSED LANDING EXPERIENCE)
# ==============================================================================
if nav_page == "🏠 Home":
    st.markdown('<div class="main-title">🌱 AgriSmart AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title"><b>Intelligent Crop Suitability & Soil Analysis</b><br>Analyze soil chemistry and agro-climatic conditions using machine learning to identify the most suitable crop varieties with explainable recommendations.</div>', unsafe_allow_html=True)

    # Key Value Highlights
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Field Records", f"{len(df_raw):,}", "Authoritative 2,200")
    with col2:
        st.metric("Crop Varieties", f"{data_summary['class_count']}", "100% Balanced (100 ea.)")
    with col3:
        st.metric("ML Test Accuracy", f"{model_metadata['metrics']['accuracy']*100:.2f}%", "5-Fold CV: 99.49%")
    with col4:
        st.metric("Audit Logs", f"{get_prediction_count()}", "SQLite Database")

    st.markdown("---")

    # What Can You Do Section
    st.markdown("### 🌾 What Can You Do With AgriSmart AI?")
    
    colA, colB, colC = st.columns(3)
    with colA:
        st.markdown("""
        <div class="feature-card">
            <div class="card-header">🌱 Predict Suitable Crop</div>
            <p>Input macronutrients (N, P, K), soil pH, temperature, humidity, and rainfall to obtain real-time crop recommendations with confidence probabilities.</p>
        </div>
        <div class="feature-card">
            <div class="card-header">🧪 Soil Health Diagnostic</div>
            <p>Assess N:P:K stoichiometry, empirical nutrient tiers (Deficient to Excessive), soil pH reaction, and actionable agronomic amendment advisories.</p>
        </div>
        """, unsafe_allow_html=True)

    with colB:
        st.markdown("""
        <div class="feature-card">
            <div class="card-header">🌤 Environmental Analysis</div>
            <p>Classify agro-climatic zones (Tropical Wet, Semi-Arid, Temperate Highland) and identify potential abiotic stresses such as heat or drought risk.</p>
        </div>
        <div class="feature-card">
            <div class="card-header">🔍 Model Explainability (XAI)</div>
            <p>Understand why specific crops were selected using Global Gini feature importances and local empirical benchmark comparisons.</p>
        </div>
        """, unsafe_allow_html=True)

    with colC:
        st.markdown("""
        <div class="feature-card">
            <div class="card-header">🔬 What-If Sensitivity Simulator</div>
            <p>Simulate parameter changes (e.g. varying rainfall from 100mm to 250mm) to explore dynamic recommendation shifts in real time.</p>
        </div>
        <div class="feature-card">
            <div class="card-header">📄 Automated PDF Dossiers</div>
            <p>Generate publication-ready diagnostic dossiers and certificates for field practitioners and academic auditing.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Quick Action Presets
    st.markdown("### ⚡ Quick Launch Demo Profiles")
    st.write("Select a typical regional agro-climatic preset to immediately populate sample conditions:")

    colP1, colP2, colP3, colP4 = st.columns(4)
    with colP1:
        if st.button("🌾 Rice Profile (Wetland)"):
            st.session_state["last_inputs"] = {"N": 90.0, "P": 42.0, "K": 43.0, "temperature": 20.87, "humidity": 82.0, "ph": 6.5, "rainfall": 202.9}
            st.session_state["nav_redirect"] = "🌾 Crop Prediction"
            st.success("Loaded Rice (Wetland) conditions. Navigate to 'Crop Prediction' to execute.")
    with colP2:
        if st.button("🧆 Chickpea Profile (Dryland)"):
            st.session_state["last_inputs"] = {"N": 40.0, "P": 72.0, "K": 77.0, "temperature": 18.8, "humidity": 16.8, "ph": 7.3, "rainfall": 80.0}
            st.success("Loaded Chickpea (Semi-Arid) conditions. Navigate to 'Crop Prediction' to execute.")
    with colP3:
        if st.button("🍏 Apple Profile (Highland)"):
            st.session_state["last_inputs"] = {"N": 20.0, "P": 134.0, "K": 199.0, "temperature": 22.5, "humidity": 92.3, "ph": 5.9, "rainfall": 110.0}
            st.success("Loaded Apple (Highland) conditions. Navigate to 'Crop Prediction' to execute.")
    with colP4:
        if st.button("👕 Cotton Profile (Warm)"):
            st.session_state["last_inputs"] = {"N": 120.0, "P": 40.0, "K": 20.0, "temperature": 24.0, "humidity": 80.0, "ph": 6.8, "rainfall": 80.0}
            st.success("Loaded Cotton (Sub-Humid) conditions. Navigate to 'Crop Prediction' to execute.")


# ==============================================================================
# 2. 🌾 CROP PREDICTION PAGE (MAIN USER-FACING INFERENCE INTERFACE)
# ==============================================================================
elif nav_page == "🌾 Crop Prediction":
    st.markdown('<div class="main-title">🌾 Crop Suitability Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Enter soil chemistry and weather measurements to execute real-time machine learning inference.</div>', unsafe_allow_html=True)

    # Input Form
    with st.form("crop_prediction_form"):
        st.subheader("1. Enter Soil & Environmental Parameters")
        
        defaults = st.session_state["last_inputs"]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("##### 🧪 Soil Chemistry")
            n_in = st.number_input("Nitrogen (N) [kg/ha]", min_value=0.0, max_value=250.0, value=float(defaults.get("N", 37.0)), step=1.0, help="Available Nitrogen in topsoil (0-140 typical range)")
            p_in = st.number_input("Phosphorus (P) [kg/ha]", min_value=0.0, max_value=250.0, value=float(defaults.get("P", 51.0)), step=1.0, help="Available Phosphorus (5-145 typical range)")
            k_in = st.number_input("Potassium (K) [kg/ha]", min_value=0.0, max_value=300.0, value=float(defaults.get("K", 32.0)), step=1.0, help="Available Potassium (5-205 typical range)")
            ph_in = st.number_input("Soil pH [0.0 - 14.0]", min_value=0.0, max_value=14.0, value=float(defaults.get("ph", 6.4)), step=0.1, help="Chemical soil reaction (0=Acidic, 7=Neutral, 14=Alkaline)")

        with col2:
            st.markdown("##### ⛅ Meteorology & Climate")
            temp_in = st.number_input("Temperature [°C]", min_value=0.0, max_value=60.0, value=float(defaults.get("temperature", 25.6)), step=0.1, help="Average growing season temperature (8-45°C typical range)")
            hum_in = st.number_input("Relative Humidity [%]", min_value=0.0, max_value=100.0, value=float(defaults.get("humidity", 80.5)), step=0.1, help="Atmospheric relative humidity (10-100%)")
            rain_in = st.number_input("Rainfall [mm]", min_value=0.0, max_value=600.0, value=float(defaults.get("rainfall", 95.0)), step=1.0, help="Annual or seasonal precipitation (20-300mm typical range)")

        with col3:
            st.markdown("##### 📝 Farm Metadata")
            field_id = st.text_input("Field / Farm Identifier", value="Field Plot A-1")
            operator_notes = st.text_area("Agronomic Notes", value="Standard pre-sowing soil test sample.", height=105)

        predict_clicked = st.form_submit_button("🌱 PREDICT SUITABLE CROP", use_container_width=True)

    if predict_clicked:
        current_inputs = {
            "N": n_in,
            "P": p_in,
            "K": k_in,
            "temperature": temp_in,
            "humidity": hum_in,
            "ph": ph_in,
            "rainfall": rain_in,
        }
        st.session_state["last_inputs"] = current_inputs

        # 1. Validation
        val_result = validate_input_features(current_inputs)
        if not val_result["is_valid"]:
            for err in val_result["errors"]:
                st.error(f"❌ Input Error: {err}")
        else:
            if val_result["warnings"]:
                for warn in val_result["warnings"]:
                    st.warning(f"⚠️ Range Advisory: {warn}")

            # 2. Real ML Model Inference
            pred_res = predict_crop_suitability(current_inputs, model=model, metadata=model_metadata)
            soil_res = analyze_soil_nutrients(n_in, p_in, k_in, ph_in)
            env_res = analyze_environmental_conditions(temp_in, hum_in, rain_in, pred_res["predicted_crop"])

            # Store in session state for cross-page tabs
            st.session_state["last_prediction"] = {
                "inputs": current_inputs,
                "prediction": pred_res,
                "soil": soil_res,
                "env": env_res,
                "notes": f"{field_id} | {operator_notes}",
            }

            # 3. Log to SQLite
            log_id = log_prediction(current_inputs, pred_res, notes=f"{field_id} | {operator_notes}")

            # 4. Display Result Hero Card
            st.markdown(f"""
            <div class="result-hero">
                <h3>Recommended Crop Variety</h3>
                <div class="crop-title">{pred_res['crop_icon']} {pred_res['predicted_crop'].upper()}</div>
                <div class="conf-badge">Prediction Confidence: {pred_res['confidence_pct']}% | Engine: {pred_res['model_used']}</div>
            </div>
            """, unsafe_allow_html=True)

            # 5. Top 5 Probabilities & Quick Analysis
            colR1, colR2 = st.columns([3, 2])
            with colR1:
                st.subheader("Top 5 Probabilistic Candidates")
                top5_df = pd.DataFrame(pred_res["top_recommendations"])
                fig_top5 = px.bar(
                    top5_df,
                    x="probability_pct",
                    y="crop",
                    orientation="h",
                    text="probability_pct",
                    color="probability_pct",
                    color_continuous_scale="greens",
                    labels={"probability_pct": "Confidence Score (%)", "crop": "Crop Variety"},
                )
                fig_top5.update_layout(yaxis=dict(autorange="reversed"), showlegend=False, height=280)
                st.plotly_chart(fig_top5, use_container_width=True)

            with colR2:
                st.subheader("Integrated Diagnostics")
                st.write(f"**Total NPK Load:** `{soil_res['npk_total']} kg/ha`")
                st.write(f"**N:P:K Ratio:** `{soil_res['npk_ratio']}`")
                st.write(f"**Soil Reaction:** `{soil_res['ph_analysis']['classification']}`")
                st.write(f"**Climate Regime:** `{env_res['climate_regime']['classification']}`")
                st.write(f"**Database Audit Log:** `Record #{log_id}`")

                # Generate PDF Report
                pdf_bytes = generate_pdf_report(current_inputs, pred_res, soil_res, env_res)
                st.download_button(
                    label="📄 Download Diagnostic PDF Dossier",
                    data=pdf_bytes,
                    file_name=f"AgriSmart_Report_{pred_res['predicted_crop']}_{log_id}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

            # 6. "Why This Crop?" Inline Explanation
            st.markdown("---")
            st.subheader(f"🔎 Why was {pred_res['predicted_crop'].capitalize()} recommended?")
            st.write(f"""
            The model classified this soil and climate profile as **{pred_res['predicted_crop'].capitalize()}** 
            based on the combined pattern of Nitrogen ({n_in:.1f} kg/ha), Phosphorus ({p_in:.1f} kg/ha), 
            Potassium ({k_in:.1f} kg/ha), Temperature ({temp_in:.1f}°C), Humidity ({hum_in:.1f}%), 
            Soil pH ({ph_in:.2f}), and Rainfall ({rain_in:.1f} mm).
            """)
            
            comp_df = generate_crop_comparison_explanation(current_inputs, pred_res["predicted_crop"])
            st.dataframe(comp_df, use_container_width=True)


# ==============================================================================
# 3. 🧪 SOIL ANALYSIS PAGE
# ==============================================================================
elif nav_page == "🧪 Soil Analysis":
    st.markdown('<div class="main-title">🧪 Soil Nutrient & Stoichiometry Diagnostic</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Comprehensive chemical evaluation of Macronutrients (N, P, K), Stoichiometry Ratios, and pH Amendment Guidelines.</div>', unsafe_allow_html=True)

    inputs = st.session_state["last_inputs"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        n_val = st.number_input("Nitrogen (N) [kg/ha]", 0.0, 250.0, float(inputs.get("N", 37.0)), key="soil_n")
    with col2:
        p_val = st.number_input("Phosphorus (P) [kg/ha]", 0.0, 250.0, float(inputs.get("P", 51.0)), key="soil_p")
    with col3:
        k_val = st.number_input("Potassium (K) [kg/ha]", 0.0, 300.0, float(inputs.get("K", 32.0)), key="soil_k")
    with col4:
        ph_val = st.number_input("Soil pH", 0.0, 14.0, float(inputs.get("ph", 6.4)), key="soil_ph")

    soil_diag = analyze_soil_nutrients(n_val, p_val, k_val, ph_val)

    st.markdown("---")
    colM1, colM2, colM3, colM4 = st.columns(4)
    with colM1:
        st.metric("Total Macronutrient Load", f"{soil_diag['npk_total']} kg/ha")
    with colM2:
        st.metric("N:P:K Stoichiometry", f"{soil_diag['npk_ratio']}")
    with colM3:
        st.metric("Soil Reaction Class", f"{soil_diag['ph_analysis']['classification']}")
    with colM4:
        st.metric("Soil Fertility Index", f"{soil_diag['fertility_score']}/100")

    st.markdown("---")
    colL, colR = st.columns(2)
    with colL:
        st.subheader("NPK Profile & Dataset Distribution Comparison")
        
        # Nitrogen comparison
        st.write(f"**Nitrogen (N):** `{n_val} kg/ha` – *Status:* **{soil_diag['nitrogen']['status']}** (Dataset Median: 37.0)")
        st.progress(min(int((n_val / 140.0) * 100), 100))
        
        # Phosphorus comparison
        st.write(f"**Phosphorus (P):** `{p_val} kg/ha` – *Status:* **{soil_diag['phosphorus']['status']}** (Dataset Median: 51.0)")
        st.progress(min(int((p_val / 145.0) * 100), 100))
        
        # Potassium comparison
        st.write(f"**Potassium (K):** `{k_val} kg/ha` – *Status:* **{soil_diag['potassium']['status']}** (Dataset Median: 32.0)")
        st.progress(min(int((k_val / 205.0) * 100), 100))

        # Radar chart
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=[n_val, p_val, k_val, ph_val * 15],
            theta=["Nitrogen", "Phosphorus", "Potassium", "Soil pH (x15)"],
            fill="toself",
            name="Soil Profile",
            line_color="#2e7d32"
        ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 200])), showlegend=False, height=300)
        st.plotly_chart(fig_radar, use_container_width=True)

    with colR:
        st.subheader("Soil pH Dynamics & Agronomic Advisory")
        ph_info = soil_diag["ph_analysis"]
        st.info(f"**Classification:** {ph_info['classification']}\n\n**Agronomic Risk:** {ph_info['risk']}")
        st.success(f"**Recommended Soil Amendment:** {ph_info['recommendation']}")

        st.markdown("##### Biological Functions of Measured Nutrients")
        st.markdown(f"- **Nitrogen:** {soil_diag['nitrogen']['role']}")
        st.markdown(f"- **Phosphorus:** {soil_diag['phosphorus']['role']}")
        st.markdown(f"- **Potassium:** {soil_diag['potassium']['role']}")


# ==============================================================================
# 4. 🌤 ENVIRONMENTAL ANALYSIS PAGE
# ==============================================================================
elif nav_page == "🌤 Environmental Analysis":
    st.markdown('<div class="main-title">🌤 Environmental & Microclimate Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Evaluate ambient thermal regime, relative humidity, and precipitation dynamics against empirical crop niches.</div>', unsafe_allow_html=True)

    inputs = st.session_state["last_inputs"]

    col1, col2, col3 = st.columns(3)
    with col1:
        t_val = st.number_input("Ambient Temperature (°C)", 0.0, 50.0, float(inputs.get("temperature", 25.6)), key="env_t")
    with col2:
        h_val = st.number_input("Relative Humidity (%)", 0.0, 100.0, float(inputs.get("humidity", 80.5)), key="env_h")
    with col3:
        r_val = st.number_input("Precipitation / Rainfall (mm)", 0.0, 500.0, float(inputs.get("rainfall", 95.0)), key="env_r")

    env_diag = analyze_environmental_conditions(t_val, h_val, r_val)

    st.markdown("---")
    st.subheader(f"Agro-Climatic Regime: {env_diag['climate_regime']['classification']}")
    st.info(f"**Ecological Regime Description:** {env_diag['climate_regime']['description']}\n\n**Naturally Adapted Crops:** {env_diag['climate_regime']['crops_suited']}")

    colE1, colE2 = st.columns(2)
    with colE1:
        st.markdown("#### Environmental Parameter Positioning")
        st.write(f"**Temperature:** `{t_val:.1f} °C` (Dataset Range: 8.8 - 43.7°C, Median: 25.6°C)")
        st.progress(min(int((t_val / 45.0) * 100), 100))

        st.write(f"**Humidity:** `{h_val:.1f} %` (Dataset Range: 14.3 - 100.0%, Median: 80.5%)")
        st.progress(min(int((h_val / 100.0) * 100), 100))

        st.write(f"**Rainfall:** `{r_val:.1f} mm` (Dataset Range: 20.2 - 298.6mm, Median: 94.9mm)")
        st.progress(min(int((r_val / 300.0) * 100), 100))

    with colE2:
        st.markdown("#### Abiotic Stress & Microclimate Health")
        st.metric("Microclimate Suitability Index", f"{env_diag['microclimate_index']}/100")
        st.write("##### Active Stress Warnings:")
        for rec in env_diag["stress_analysis"]["recommendations"]:
            st.warning(f"⚠️ {rec}")


# ==============================================================================
# 5. 🔍 EXPLAINABILITY PAGE
# ==============================================================================
elif nav_page == "🔍 Explainability":
    st.markdown('<div class="main-title">🔍 Explainable AI (XAI) & Attribution</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Transparent Machine Learning Interpretability via Global Gini Feature Importance and Local Target Crop Benchmarking.</div>', unsafe_allow_html=True)

    tab_x1, tab_x2 = st.tabs(["🌳 Global Model Feature Importance", "🎯 Local Crop Benchmark Comparison"])

    with tab_x1:
        st.subheader("Global Feature Importance (Random Forest Gini Impurity Reduction)")
        importances = get_model_feature_importance(model, model_metadata["features"])
        imp_df = pd.DataFrame(list(importances.items()), columns=["Feature", "Gini Importance Score"])
        
        fig_imp = px.bar(
            imp_df,
            x="Gini Importance Score",
            y="Feature",
            orientation="h",
            color="Gini Importance Score",
            color_continuous_scale="Viridis",
            text="Gini Importance Score",
            title="Global Feature Importance Ranking (Sum = 100%)",
        )
        fig_imp.update_layout(yaxis=dict(autorange="reversed"), height=350)
        st.plotly_chart(fig_imp, use_container_width=True)
        st.info("💡 **Interpretation:** The chart shows the relative contribution of each parameter across all 2,200 observations. Meteorological drivers (Rainfall, Humidity) and macronutrients (Potassium, Phosphorus) have the highest discriminatory power in separating the 22 crop classes.")

    with tab_x2:
        st.subheader("Benchmark User Input Against Empirical Crop Profiles")
        target_crop = st.selectbox("Select Target Crop Species to Benchmark", model_metadata["classes"])
        inputs = st.session_state["last_inputs"]
        comp_df = generate_crop_comparison_explanation(inputs, target_crop)
        st.dataframe(comp_df, use_container_width=True)


# ==============================================================================
# 6. 🔬 WHAT-IF ANALYSIS PAGE
# ==============================================================================
elif nav_page == "🔬 What-If Analysis":
    st.markdown('<div class="main-title">🔬 What-If Scenario Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Simulate parameter variations and observe dynamic prediction transitions using the actual trained model.</div>', unsafe_allow_html=True)

    base_inputs = st.session_state["last_inputs"].copy()

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Configure Variable Perturbation")
        feat_to_vary = st.selectbox("Select Parameter to Vary", NUMERICAL_FEATURES, index=6) # Rainfall
        bounds = EXPECTED_RANGES[feat_to_vary]

        curr_val = base_inputs.get(feat_to_vary, bounds["min"])
        new_val = st.slider(
            f"Perturb {bounds['name']} ({bounds['unit']})",
            min_value=float(bounds["min"]),
            max_value=float(bounds["max"]),
            value=float(curr_val),
            step=1.0 if bounds["max"] > 20 else 0.1,
        )

        analyze_clicked = st.button("🔬 Analyze Change", use_container_width=True)

    with col2:
        st.subheader("Simulation Outcome")
        
        # Base Prediction
        orig_res = predict_crop_suitability(base_inputs, model=model, metadata=model_metadata)
        
        # New Prediction with perturbed feature
        perturbed_inputs = base_inputs.copy()
        perturbed_inputs[feat_to_vary] = new_val
        new_res = predict_crop_suitability(perturbed_inputs, model=model, metadata=model_metadata)

        colRes1, colRes2 = st.columns(2)
        with colRes1:
            st.markdown("##### Original Baseline Prediction")
            st.write(f"**Crop:** {orig_res['crop_icon']} **{orig_res['predicted_crop'].capitalize()}**")
            st.write(f"**Confidence:** `{orig_res['confidence_pct']}%`")
            st.write(f"**{bounds['name']}:** `{curr_val} {bounds['unit']}`")

        with colRes2:
            st.markdown("##### New Perturbed Prediction")
            st.write(f"**Crop:** {new_res['crop_icon']} **{new_res['predicted_crop'].capitalize()}**")
            st.write(f"**Confidence:** `{new_res['confidence_pct']}%`")
            st.write(f"**{bounds['name']}:** `{new_val} {bounds['unit']}`")

        st.markdown("---")
        if orig_res["predicted_crop"] == new_res["predicted_crop"]:
            st.success(f"✅ **Prediction Remained Stable:** The model continues to classify the profile as **{new_res['predicted_crop'].capitalize()}** despite changing {bounds['name']}.")
        else:
            st.warning(f"⚠️ **Prediction Transition Detected:** Shifting {bounds['name']} from {curr_val} to {new_val} {bounds['unit']} transitioned the recommendation from **{orig_res['predicted_crop'].capitalize()}** to **{new_res['predicted_crop'].capitalize()}**.")

    # Multi-point sensitivity curve
    st.markdown("---")
    st.subheader(f"Dynamic Sensitivity Curve across entire {bounds['name']} Range")
    sim_data = simulate_what_if_scenario(base_inputs, feat_to_vary, bounds["min"], bounds["max"], steps=15)
    sim_df = pd.DataFrame(sim_data)
    fig_sim = px.line(
        sim_df,
        x="value",
        y="confidence_pct",
        color="predicted_crop",
        markers=True,
        title=f"Crop Recommendation Shifts along {bounds['name']} Range",
        labels={"value": f"{bounds['name']} ({bounds['unit']})", "confidence_pct": "Confidence Score (%)", "predicted_crop": "Predicted Crop"},
    )
    st.plotly_chart(fig_sim, use_container_width=True)


# ==============================================================================
# 7. 📜 PREDICTION HISTORY PAGE
# ==============================================================================
elif nav_page == "📜 Prediction History":
    st.markdown('<div class="main-title">📜 Prediction History & Audit Trail</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Persistent local SQLite audit logs of all executed inferences and field diagnostics.</div>', unsafe_allow_html=True)

    history_df = export_predictions_to_dataframe()

    if not history_df.empty:
        colH1, colH2, colH3 = st.columns(3)
        with colH1:
            st.metric("Total Inferences Logged", f"{len(history_df)}")
        with colH2:
            st.metric("Most Frequent Recommendation", f"{history_df['predicted_crop'].mode()[0].capitalize()}")
        with colH3:
            st.metric("Average Confidence", f"{history_df['confidence'].mean()*100:.1f}%")

        st.markdown("---")
        st.subheader("Historical Prediction Records")
        
        # Display formatted table
        st.dataframe(history_df, use_container_width=True)

        # Download CSV Button
        csv_bytes = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Complete History CSV",
            data=csv_bytes,
            file_name="agrismart_prediction_history.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.info("No predictions logged in the SQLite database yet. Navigate to 'Crop Prediction' to run and record inferences.")


# ==============================================================================
# 8. 📊 DATA ANALYTICS PAGE
# ==============================================================================
elif nav_page == "📊 Data Analytics":
    st.markdown('<div class="main-title">📊 Authoritative Data Analytics & EDA</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Deep exploratory data analysis on the authoritative 2,200-record Crop_recommendation.csv dataset.</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Summary & Overview", "📈 Feature Distributions", "📦 Boxplots & Ranges", "🔗 Correlation Heatmap", "⚠️ IQR Outliers"])

    with tab1:
        st.subheader("Dataset Integrity & Health Overview")
        colO1, colO2, colO3 = st.columns(3)
        with colO1:
            st.write(f"**Total Records:** {len(df_raw):,}")
            st.write(f"**Feature Columns:** {len(NUMERICAL_FEATURES)}")
            st.write(f"**Target Column:** `{TARGET_COLUMN}`")
        with colO2:
            st.write(f"**Missing Values:** {data_summary['missing_values']} (0%)")
            st.write(f"**Duplicate Records:** {data_summary['duplicate_rows']} (0%)")
            st.write(f"**Class Balance:** 100% Uniform (100 rows/crop)")
        with colO3:
            st.write(f"**Total Crop Classes:** {data_summary['class_count']}")
            st.write(f"**Memory Usage:** {round(df_raw.memory_usage().sum()/1024, 2)} KB")

        st.markdown("#### Descriptive Statistics")
        st.dataframe(get_dataset_summary_statistics(df_raw), use_container_width=True)

        st.markdown("#### Crop Class Distribution")
        class_df = pd.DataFrame(list(data_summary["class_distribution"].items()), columns=["Crop", "Record Count"])
        fig_class = px.bar(class_df, x="Crop", y="Record Count", color="Record Count", color_continuous_scale="greens", title="Crop Class Frequency (100 Rows Each)")
        fig_class.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_class, use_container_width=True)

    with tab2:
        st.subheader("Univariate Feature Histograms")
        feat_choice = st.selectbox("Select Parameter to Plot", NUMERICAL_FEATURES, index=0)
        fig_hist = px.histogram(df_raw, x=feat_choice, color=TARGET_COLUMN, nbins=35, title=f"Empirical Distribution of {feat_choice} across Crops")
        st.plotly_chart(fig_hist, use_container_width=True)

    with tab3:
        st.subheader("Feature Boxplots by Crop Class")
        box_feat = st.selectbox("Select Parameter for Boxplot", NUMERICAL_FEATURES, index=6) # Rainfall
        fig_box = px.box(df_raw, x=TARGET_COLUMN, y=box_feat, color=TARGET_COLUMN, title=f"Boxplot of {box_feat} across 22 Crop Classes")
        fig_box.update_layout(xaxis_tickangle=-45, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    with tab4:
        st.subheader("Pearson Correlation Heatmap")
        corr = df_raw[NUMERICAL_FEATURES].corr()
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="Viridis", title="Inter-Feature Pearson Correlation Matrix")
        st.plotly_chart(fig_corr, use_container_width=True)

    with tab5:
        st.subheader("IQR Outlier Analysis & Agronomic Justification")
        outliers = detect_outliers_iqr(df_raw)
        st.dataframe(pd.DataFrame(outliers).T, use_container_width=True)
        st.info("💡 **Agronomic Justification:** Outliers in rainfall (e.g. Rice requiring >200mm) and potassium (e.g. Apple requiring >190kg/ha) represent true ecological niches rather than errors.")


# ==============================================================================
# 9. 🤖 MODEL PERFORMANCE PAGE
# ==============================================================================
elif nav_page == "🤖 Model Performance":
    st.markdown('<div class="main-title">🤖 Model Performance & Multi-Algorithm Benchmark</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Rigorous stratified benchmarking across 8 configurations (4 Algorithms x 2 Feature Sets).</div>', unsafe_allow_html=True)

    comp_path = os.path.join(ROOT_DIR, "models", "model_comparison.json")
    if os.path.exists(comp_path):
        with open(comp_path, "r") as f:
            benchmarks = json.load(f)

        bench_df = pd.DataFrame(benchmarks)[
            ["config_name", "model_name", "feature_set", "test_accuracy", "test_f1_weighted", "cv_f1_mean", "cv_f1_std"]
        ]

        st.subheader("1. Multi-Model Benchmark Comparison Table")
        st.dataframe(bench_df, use_container_width=True)

        st.subheader("2. Test F1-Score Comparison Chart")
        fig_comp = px.bar(
            bench_df,
            x="config_name",
            y="test_f1_weighted",
            color="model_name",
            text="test_f1_weighted",
            title="Test F1-Score Across Evaluated Configurations",
            labels={"test_f1_weighted": "Weighted F1-Score", "config_name": "Model & Feature Set Configuration"},
        )
        fig_comp.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig_comp, use_container_width=True)

        st.subheader("3. Production Champion Model Selection")
        st.success(f"""
        🏆 **Selected Champion Model:** `{model_metadata['winning_config']}`
        - **Test Accuracy:** `{model_metadata['metrics']['accuracy']*100:.2f}%`
        - **Weighted F1-Score:** `{model_metadata['metrics']['f1_weighted']*100:.2f}%`
        - **5-Fold Cross-Validation F1 Mean:** `{model_metadata['metrics']['cv_f1_mean']*100:.2f}% (+/- {model_metadata['metrics']['cv_f1_std']*100:.2f}%)`
        """)

        st.subheader("4. Multiclass Confusion Matrix (22 Crop Classes)")
        cm_data = np.array(model_metadata["confusion_matrix"])
        classes = model_metadata["classes"]
        
        fig_cm = px.imshow(
            cm_data,
            x=classes,
            y=classes,
            text_auto=True,
            color_continuous_scale="greens",
            labels=dict(x="Predicted Crop", y="True Ground Truth Crop"),
            title="Confusion Matrix (Holdout Test Split - 440 Observations)",
        )
        fig_cm.update_layout(height=650, xaxis_tickangle=-45)
        st.plotly_chart(fig_cm, use_container_width=True)


# ==============================================================================
# 10. 📄 REPORTS PAGE
# ==============================================================================
elif nav_page == "📄 Reports":
    st.markdown('<div class="main-title">📄 Automated Diagnostic Report Generator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Generate, preview, and download publication-ready ReportLab PDF crop suitability dossiers.</div>', unsafe_allow_html=True)

    inputs = st.session_state["last_inputs"]
    last_pred = st.session_state.get("last_prediction")

    if last_pred is None:
        # Generate on the fly
        pred_res = predict_crop_suitability(inputs, model=model, metadata=model_metadata)
        soil_res = analyze_soil_nutrients(inputs["N"], inputs["P"], inputs["K"], inputs["ph"])
        env_res = analyze_environmental_conditions(inputs["temperature"], inputs["humidity"], inputs["rainfall"], pred_res["predicted_crop"])
    else:
        pred_res = last_pred["prediction"]
        soil_res = last_pred["soil"]
        env_res = last_pred["env"]

    st.subheader("Current Diagnostic Dossier Preview")
    colRep1, colRep2 = st.columns(2)
    with colRep1:
        st.write(f"**Target Recommended Crop:** {pred_res['crop_icon']} `{pred_res['predicted_crop'].capitalize()}`")
        st.write(f"**Model Confidence:** `{pred_res['confidence_pct']}%`")
        st.write(f"**Model Used:** `{pred_res['model_used']}`")
        st.write(f"**Soil Health Score:** `{soil_res['fertility_score']}/100`")
    with colRep2:
        st.write(f"**Soil Reaction:** `{soil_res['ph_analysis']['classification']}`")
        st.write(f"**Climate Regime:** `{env_res['climate_regime']['classification']}`")
        st.write(f"**Microclimate Index:** `{env_res['microclimate_index']}/100`")

    pdf_bytes = generate_pdf_report(inputs, pred_res, soil_res, env_res)
    st.download_button(
        label="📄 Download Official PDF Diagnostic Dossier",
        data=pdf_bytes,
        file_name=f"AgriSmart_Official_Report_{pred_res['predicted_crop']}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


# ==============================================================================
# 11. ℹ️ ABOUT & ACADEMIC VIVA PAGE
# ==============================================================================
elif nav_page == "ℹ️ About":
    st.markdown('<div class="main-title">ℹ️ Academic Documentation & Viva Defense</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Capstone methodology, academic defense guidelines, and dataset attribution.</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 🎓 Internship Capstone Project Information
    - **Academic Title:** *An Explainable Machine Learning System for Crop Suitability Classification and Soil Analysis*
    - **Domain:** Applied Machine Learning, Agronomic Informatics & Decision Support Systems
    - **Authoritative Dataset:** `Crop_recommendation.csv` (2,200 empirical field records, 22 balanced classes)

    ---

    ### 🛡️ Viva Defense Questions & Expert Answers

    #### Q1: Why did you choose Random Forest over Deep Neural Networks?
    **Answer:** The dataset comprises 2,200 structured tabular rows across 7 physical features. Deep neural networks risk overfitting and lack intrinsic interpretability on small tabular datasets. Random Forest constructs orthogonal decision trees matching agricultural thresholds, maintains low variance via bagging, and offers transparent Gini feature attribution.

    #### Q2: How did you strictly prevent Data Leakage?
    **Answer:** Preprocessing transformers (`StandardScaler`) were encapsulated in `sklearn.pipeline.Pipeline` objects and fitted exclusively on training splits during 5-Fold Stratified Cross-Validation. The target label was strictly isolated from all feature transformations.

    #### Q3: Why did original 7 physical features outperform engineered stoichiometric ratios?
    **Answer:** The 7 physical features ($N, P, K, \text{temp}, \text{humidity}, \text{pH}, \text{rainfall}$) form orthogonal, independent physical dimensions. Decision trees naturally capture non-linear interactions. Adding collinear ratios added slight noise without providing additional mutual information.
    """)
