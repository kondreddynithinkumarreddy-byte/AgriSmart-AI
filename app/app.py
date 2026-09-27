"""
AgriSmart AI – Intelligent Crop Suitability & Soil Analysis System
Production Streamlit Application
"""

import os
import sys
import json
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_processing import (
    load_and_validate_dataset,
    get_crop_profiles,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
)
from src.prediction import (
    load_production_model,
    load_model_metadata,
    predict_crop_suitability,
    validate_input_features,
    EXPECTED_RANGES,
    CROP_ICONS,
)
from src.soil_analysis import analyze_soil_nutrients
from src.environmental_analysis import analyze_environmental_conditions
from src.database import (
    init_db,
    log_prediction,
    get_recent_predictions,
    get_prediction_count,
    export_predictions_to_dataframe,
    clear_history,
)
from src.report_generator import generate_pdf_report

# Page Configuration
st.set_page_config(
    page_title="AgriSmart AI – Crop & Soil Intelligence",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    /* Global Typography & Palette */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1b5e20;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4a5568;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #2e7d32;
        padding: 1.1rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 0.8rem;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #718096;
        font-weight: 600;
        text-transform: uppercase;
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1a202c;
    }
    .metric-sub {
        font-size: 0.82rem;
        color: #2e7d32;
        font-weight: 500;
    }
    .result-hero {
        background: linear-gradient(135deg, #2e7d32 0%, #1b5e20 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(27, 94, 32, 0.25);
    }
    .hero-crop-title {
        font-size: 3.2rem;
        font-weight: 800;
        margin: 0.4rem 0;
        color: #ffffff;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        color: #e8f5e9;
    }
    .badge-tag {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        background-color: #e8f5e9;
        color: #1b5e20;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    .stButton>button {
        background-color: #2e7d32;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        border: none;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
        color: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)


# Cached Data & Model Loading
@st.cache_data
def get_dataset():
    df, summary = load_and_validate_dataset()
    return df, summary


@st.cache_resource
def get_model():
    model = load_production_model()
    metadata = load_model_metadata()
    return model, metadata


# Initialize Database
init_db()

# Load Resources
df_raw, data_summary = get_dataset()
model, model_metadata = get_model()

# Canonical Session State Initialization
if "inputs" not in st.session_state:
    st.session_state.inputs = {
        "N": 90.0,
        "P": 42.0,
        "K": 43.0,
        "temperature": 20.87,
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": 202.91,
    }
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "confidence" not in st.session_state:
    st.session_state.confidence = None
if "probabilities" not in st.session_state:
    st.session_state.probabilities = None
if "model_name" not in st.session_state:
    st.session_state.model_name = model_metadata.get("model_name", "Random Forest")
if "timestamp" not in st.session_state:
    st.session_state.timestamp = None
if "field_notes" not in st.session_state:
    st.session_state.field_notes = "Field Zone A-1"
if "soil_analysis" not in st.session_state:
    st.session_state.soil_analysis = None
if "env_analysis" not in st.session_state:
    st.session_state.env_analysis = None


# Sidebar Navigation
st.sidebar.markdown("## 🌱 **AgriSmart AI**")
st.sidebar.caption("Intelligent Crop Suitability & Soil System")
st.sidebar.markdown("---")

nav_choice = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🌱 Crop Analysis",
        "🧪 Soil Analysis",
        "⛅ Environmental Analysis",
        "💡 Insights",
        "📜 Prediction History",
        "ℹ️ About",
    ],
    index=0,
)

st.sidebar.markdown("---")
pred_count = get_prediction_count()
acc_pct = model_metadata.get("metrics", {}).get("accuracy", 0.9955) * 100
st.sidebar.markdown(
    f"""
    <div style="background-color: #f1f8e9; padding: 0.8rem; border-radius: 8px; border: 1px solid #c8e6c9;">
        <div style="font-size: 0.8rem; color: #33691e; font-weight: bold;">SYSTEM STATUS</div>
        <div style="font-size: 0.85rem; color: #1b5e20; margin-top: 4px;">
            🤖 <b>Engine:</b> {model_metadata.get('model_name', 'Random Forest')}<br/>
            🎯 <b>Accuracy:</b> {acc_pct:.2f}%<br/>
            📊 <b>Classes:</b> {data_summary['class_count']} Crops<br/>
            💾 <b>Audits Logged:</b> {pred_count}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# 1. HOME PAGE
# ==============================================================================
if nav_choice == "🏠 Home":
    st.markdown('<div class="main-header">🌱 AgriSmart AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Intelligent Crop Suitability & Soil Analysis Decision-Support System</div>', unsafe_allow_html=True)

    # Hero Banner
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%); color: white; padding: 1.8rem; border-radius: 12px; margin-bottom: 1.5rem;">
        <h2 style="color: white; margin-top: 0; font-size: 1.8rem;">Precision Agriculture Powered by Machine Learning</h2>
        <p style="font-size: 1.05rem; color: #e8f5e9; max-width: 900px; margin-bottom: 0.8rem;">
            AgriSmart AI optimizes agricultural yield and reduces resource misallocation by evaluating soil chemical stoichiometry 
            (Nitrogen, Phosphorus, Potassium, pH) and microclimate parameters (Temperature, Humidity, Rainfall) 
            against 22 balanced crop classification profiles using trained ensemble learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Key Stat Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Supported Crops</div>
            <div class="metric-val">{data_summary['class_count']} Crops</div>
            <div class="metric-sub">Balanced Kaggle Benchmark</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Model Accuracy</div>
            <div class="metric-val">{acc_pct:.2f}%</div>
            <div class="metric-sub">5-Fold Stratified CV</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Features Evaluated</div>
            <div class="metric-val">7 Variables</div>
            <div class="metric-sub">Chemical & Meteorological</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Audits Logged</div>
            <div class="metric-val">{pred_count} Records</div>
            <div class="metric-sub">Persistent SQLite Database</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🌾 How AgriSmart AI Works")
    step_c1, step_c2, step_c3 = st.columns(3)
    with step_c1:
        st.markdown("""
        <div style="background-color: #f8faf8; border: 1px solid #e0e6ed; padding: 1.2rem; border-radius: 8px; height: 100%;">
            <h4 style="color: #2e7d32; margin-top: 0;">1. Input Telemetry</h4>
            <p style="font-size: 0.9rem; color: #4a5568;">
                Provide soil chemical levels (N, P, K, pH) and environmental weather data (temperature, humidity, rainfall) in the <b>Crop Analysis</b> module.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with step_c2:
        st.markdown("""
        <div style="background-color: #f8faf8; border: 1px solid #e0e6ed; padding: 1.2rem; border-radius: 8px; height: 100%;">
            <h4 style="color: #2e7d32; margin-top: 0;">2. Machine Learning Inference</h4>
            <p style="font-size: 0.9rem; color: #4a5568;">
                Our validated Random Forest ensemble evaluates the multi-dimensional parameter space to generate calibrated suitability probabilities.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with step_c3:
        st.markdown("""
        <div style="background-color: #f8faf8; border: 1px solid #e0e6ed; padding: 1.2rem; border-radius: 8px; height: 100%;">
            <h4 style="color: #2e7d32; margin-top: 0;">3. Actionable Diagnostics</h4>
            <p style="font-size: 0.9rem; color: #4a5568;">
                Review comprehensive soil stoichiometry, pH risk analysis, abiotic climate stress factors, and download an official diagnostic PDF.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🌿 Supported Crop Species")
    crop_list = sorted(data_summary["unique_classes"])
    tags_html = "".join([f'<span class="badge-tag">{CROP_ICONS.get(c.lower(), "🌱")} {c.capitalize()}</span>' for c in crop_list])
    st.markdown(f'<div style="line-height: 2.2;">{tags_html}</div>', unsafe_allow_html=True)


# ==============================================================================
# 2. CROP ANALYSIS PAGE (THE ONLY INPUT PAGE)
# ==============================================================================
elif nav_choice == "🌱 Crop Analysis":
    st.markdown('<div class="main-header">🌱 Crop Suitability Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Input Soil Chemical Parameters & Environmental Weather Conditions</div>', unsafe_allow_html=True)

    # Preset selector
    st.markdown("##### ⚡ Quick-Fill Agronomic Presets")
    p_col1, p_col2, p_col3, p_col4, p_col5, p_col6 = st.columns(6)
    
    presets = {
        "Rice": {"N": 90.0, "P": 42.0, "K": 43.0, "temperature": 20.87, "humidity": 82.0, "ph": 6.5, "rainfall": 202.91, "notes": "Zone A - Wetland Rice"},
        "Maize": {"N": 71.0, "P": 54.0, "K": 20.0, "temperature": 22.61, "humidity": 65.4, "ph": 5.7, "rainfall": 82.64, "notes": "Zone B - Maize Field"},
        "Chickpea": {"N": 40.0, "P": 67.0, "K": 79.0, "temperature": 18.87, "humidity": 16.8, "ph": 7.3, "rainfall": 80.05, "notes": "Zone C - Dryland Pulse"},
        "Cotton": {"N": 117.0, "P": 46.0, "K": 19.0, "temperature": 23.96, "humidity": 79.8, "ph": 6.9, "rainfall": 80.39, "notes": "Zone D - Warm Cotton"},
        "Coffee": {"N": 101.0, "P": 28.0, "K": 30.0, "temperature": 25.54, "humidity": 58.8, "ph": 6.7, "rainfall": 158.07, "notes": "Zone E - Coffee Plantation"},
        "Apple": {"N": 20.0, "P": 134.0, "K": 199.0, "temperature": 22.75, "humidity": 92.3, "ph": 5.9, "rainfall": 110.43, "notes": "Zone F - Highland Orchard"},
    }

    if p_col1.button("🌾 Rice", use_container_width=True):
        st.session_state.inputs = presets["Rice"].copy()
        st.session_state.field_notes = presets["Rice"]["notes"]
        st.rerun()
    if p_col2.button("🌽 Maize", use_container_width=True):
        st.session_state.inputs = presets["Maize"].copy()
        st.session_state.field_notes = presets["Maize"]["notes"]
        st.rerun()
    if p_col3.button("🧆 Chickpea", use_container_width=True):
        st.session_state.inputs = presets["Chickpea"].copy()
        st.session_state.field_notes = presets["Chickpea"]["notes"]
        st.rerun()
    if p_col4.button("👕 Cotton", use_container_width=True):
        st.session_state.inputs = presets["Cotton"].copy()
        st.session_state.field_notes = presets["Cotton"]["notes"]
        st.rerun()
    if p_col5.button("☕ Coffee", use_container_width=True):
        st.session_state.inputs = presets["Coffee"].copy()
        st.session_state.field_notes = presets["Coffee"]["notes"]
        st.rerun()
    if p_col6.button("🍏 Apple", use_container_width=True):
        st.session_state.inputs = presets["Apple"].copy()
        st.session_state.field_notes = presets["Apple"]["notes"]
        st.rerun()

    st.markdown("---")

    # The Core Input Form
    with st.form("crop_prediction_form"):
        form_col1, form_col2 = st.columns(2)

        with form_col1:
            st.markdown("#### 🧪 Soil Chemical Properties")
            n_val = st.number_input(
                "Nitrogen (N) [kg/ha]",
                min_value=0.0,
                max_value=250.0,
                value=float(st.session_state.inputs.get("N", 90.0)),
                step=1.0,
                help="Nitrogen content in topsoil (typical range: 0 - 140 kg/ha)",
            )
            p_val = st.number_input(
                "Phosphorus (P) [kg/ha]",
                min_value=0.0,
                max_value=250.0,
                value=float(st.session_state.inputs.get("P", 42.0)),
                step=1.0,
                help="Phosphorus content in topsoil (typical range: 5 - 145 kg/ha)",
            )
            k_val = st.number_input(
                "Potassium (K) [kg/ha]",
                min_value=0.0,
                max_value=300.0,
                value=float(st.session_state.inputs.get("K", 43.0)),
                step=1.0,
                help="Potassium content in topsoil (typical range: 5 - 205 kg/ha)",
            )
            ph_val = st.number_input(
                "Soil pH [0.0 – 14.0]",
                min_value=0.0,
                max_value=14.0,
                value=float(st.session_state.inputs.get("ph", 6.5)),
                step=0.1,
                help="Soil acidity / alkalinity level (optimal range: 5.5 - 7.5)",
            )

        with form_col2:
            st.markdown("#### ⛅ Environmental & Climate Conditions")
            temp_val = st.number_input(
                "Temperature [°C]",
                min_value=0.0,
                max_value=60.0,
                value=float(st.session_state.inputs.get("temperature", 20.87)),
                step=0.1,
                help="Average ambient growing temperature in Celsius",
            )
            hum_val = st.number_input(
                "Relative Humidity [%]",
                min_value=0.0,
                max_value=100.0,
                value=float(st.session_state.inputs.get("humidity", 82.0)),
                step=0.1,
                help="Atmospheric relative humidity percentage",
            )
            rain_val = st.number_input(
                "Rainfall [mm]",
                min_value=0.0,
                max_value=600.0,
                value=float(st.session_state.inputs.get("rainfall", 202.91)),
                step=1.0,
                help="Seasonal or annual precipitation depth in mm",
            )
            notes_val = st.text_input(
                "Field ID / Operational Reference",
                value=str(st.session_state.get("field_notes", "Field Zone A-1")),
                help="Custom reference tag to identify this audit in prediction history",
            )

        submit_btn = st.form_submit_button("🌱 PREDICT SUITABLE CROP", use_container_width=True)

    # Process Form Submission
    if submit_btn:
        user_inputs = {
            "N": n_val,
            "P": p_val,
            "K": k_val,
            "temperature": temp_val,
            "humidity": hum_val,
            "ph": ph_val,
            "rainfall": rain_val,
        }
        val_res = validate_input_features(user_inputs)

        if not val_res["is_valid"]:
            for err in val_res["errors"]:
                st.error(f"❌ {err}")
        else:
            if val_res["warnings"]:
                for warn in val_res["warnings"]:
                    st.warning(f"⚠️ {warn}")

            with st.spinner("Executing Machine Learning Inference..."):
                pred_res = predict_crop_suitability(user_inputs, model=model, metadata=model_metadata)
                soil_res = analyze_soil_nutrients(n_val, p_val, k_val, ph_val)
                env_res = analyze_environmental_conditions(temp_val, hum_val, rain_val, pred_res["predicted_crop"])
                log_id = log_prediction(user_inputs, pred_res, notes=notes_val)

                # Persist to Session State
                st.session_state.inputs = user_inputs
                st.session_state.prediction = pred_res["predicted_crop"]
                st.session_state.confidence = pred_res["confidence_pct"]
                st.session_state.probabilities = pred_res["top_recommendations"]
                st.session_state.model_name = pred_res["model_used"]
                st.session_state.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.field_notes = notes_val
                st.session_state.soil_analysis = soil_res
                st.session_state.env_analysis = env_res
                st.session_state.last_log_id = log_id

    # Display Prediction Results (if present in session state)
    if st.session_state.prediction:
        st.markdown("---")
        crop_name = st.session_state.prediction
        icon = CROP_ICONS.get(crop_name.lower(), "🌱")
        conf = st.session_state.confidence
        m_name = st.session_state.model_name

        # Result Hero Banner
        st.markdown(f"""
        <div class="result-hero">
            <div style="font-size: 1.1rem; text-transform: uppercase; letter-spacing: 1px; color: #c8e6c9; font-weight: 600;">
                OPTIMAL CROP RECOMMENDATION
            </div>
            <div class="hero-crop-title">
                {icon} {crop_name.capitalize()}
            </div>
            <div class="hero-subtitle">
                Model Confidence: <b>{conf:.2f}%</b> &nbsp;|&nbsp; Engine: <b>{m_name}</b> &nbsp;|&nbsp; Ref: <b>{st.session_state.field_notes}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Probabilities & Diagnostics Row
        res_col1, res_col2 = st.columns([3, 2])

        with res_col1:
            st.markdown("#### 📊 Top 5 Probabilistic Crop Candidates")
            top_probs = st.session_state.probabilities or []
            if top_probs:
                p_df = pd.DataFrame(top_probs)
                p_df["display_name"] = p_df["icon"] + " " + p_df["crop"].str.capitalize()
                
                fig = px.bar(
                    p_df,
                    x="probability_pct",
                    y="display_name",
                    orientation="h",
                    text=p_df["probability_pct"].apply(lambda v: f"{v:.2f}%"),
                    color="probability_pct",
                    color_continuous_scale="greens",
                )
                fig.update_layout(
                    yaxis=dict(autorange="reversed", title=""),
                    xaxis=dict(title="Probability (%)", range=[0, max(100, max(p_df["probability_pct"]) * 1.15)]),
                    showlegend=False,
                    height=280,
                    margin=dict(l=10, r=20, t=20, b=30),
                )
                st.plotly_chart(fig, use_container_width=True)

        with res_col2:
            st.markdown("#### 🩺 Instant Health Summary")
            s_data = st.session_state.soil_analysis or analyze_soil_nutrients(
                st.session_state.inputs["N"], st.session_state.inputs["P"], st.session_state.inputs["K"], st.session_state.inputs["ph"]
            )
            e_data = st.session_state.env_analysis or analyze_environmental_conditions(
                st.session_state.inputs["temperature"], st.session_state.inputs["humidity"], st.session_state.inputs["rainfall"], crop_name
            )

            st.markdown(f"""
            <div style="background-color: #f8faf8; border: 1px solid #e0e6ed; padding: 1rem; border-radius: 8px; margin-bottom: 0.8rem;">
                <div style="font-weight: 600; color: #1b5e20;">🧪 Soil Fertility Score: {s_data['fertility_score']}/100</div>
                <div style="font-size: 0.85rem; color: #4a5568;">
                    Reaction: <b>{s_data['ph_analysis']['classification']}</b> (pH {st.session_state.inputs['ph']})<br/>
                    NPK Total: <b>{s_data['npk_total']} kg/ha</b> | Ratio: <b>{s_data['npk_ratio']}</b>
                </div>
            </div>
            <div style="background-color: #f8faf8; border: 1px solid #e0e6ed; padding: 1rem; border-radius: 8px; margin-bottom: 0.8rem;">
                <div style="font-weight: 600; color: #1b5e20;">⛅ Climate Regime: {e_data['climate_regime']['classification']}</div>
                <div style="font-size: 0.85rem; color: #4a5568;">
                    Microclimate Index: <b>{e_data['microclimate_index']}/100</b><br/>
                    Active Abiotic Stresses: <b>{e_data['stress_analysis']['active_stress_count']} detected</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # PDF Dossier Download
        pred_dict = {
            "predicted_crop": crop_name,
            "confidence_pct": conf,
            "model_used": m_name,
            "top_recommendations": st.session_state.probabilities,
        }
        pdf_bytes = generate_pdf_report(st.session_state.inputs, pred_dict, s_data, e_data)
        st.download_button(
            label="📄 Download Diagnostic PDF Dossier",
            data=pdf_bytes,
            file_name=f"AgriSmart_Diagnostic_{crop_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


# ==============================================================================
# 3. SOIL ANALYSIS PAGE
# ==============================================================================
elif nav_choice == "🧪 Soil Analysis":
    st.markdown('<div class="main-header">🧪 Soil Nutrient & Stoichiometry Diagnostic</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">In-Depth Evaluation of Primary Macronutrients (NPK) and Soil Reaction (pH)</div>', unsafe_allow_html=True)

    inputs = st.session_state.inputs
    crop_ctx = st.session_state.prediction
    soil_res = analyze_soil_nutrients(inputs["N"], inputs["P"], inputs["K"], inputs["ph"])

    st.info(f"📋 **Current Field Telemetry:** N={inputs['N']} kg/ha, P={inputs['P']} kg/ha, K={inputs['K']} kg/ha, pH={inputs['ph']} | *Reference: {st.session_state.get('field_notes', 'Current Inputs')}*")

    # Top Nutrient Level Cards
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        n_st = soil_res["nitrogen"]["status"]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Nitrogen (N)</div>
            <div class="metric-val">{inputs['N']} kg/ha</div>
            <div class="metric-sub">Status: <b>{n_st}</b></div>
        </div>
        """, unsafe_allow_html=True)
    with sc2:
        p_st = soil_res["phosphorus"]["status"]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Phosphorus (P)</div>
            <div class="metric-val">{inputs['P']} kg/ha</div>
            <div class="metric-sub">Status: <b>{p_st}</b></div>
        </div>
        """, unsafe_allow_html=True)
    with sc3:
        k_st = soil_res["potassium"]["status"]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Potassium (K)</div>
            <div class="metric-val">{inputs['K']} kg/ha</div>
            <div class="metric-sub">Status: <b>{k_st}</b></div>
        </div>
        """, unsafe_allow_html=True)
    with sc4:
        ph_info = soil_res["ph_analysis"]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Soil pH Reaction</div>
            <div class="metric-val">{inputs['ph']} pH</div>
            <div class="metric-sub" style="color: {ph_info['color']};"><b>{ph_info['classification']}</b></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # NPK Stoichiometry & Visual Comparison
    s_col1, s_col2 = st.columns(2)

    with s_col1:
        st.markdown("#### ⚖️ NPK Stoichiometry & Nutrient Balance")
        st.write(f"**Total Macronutrient Load:** `{soil_res['npk_total']} kg/ha`")
        st.write(f"**Empirical N:P:K Ratio:** `{soil_res['npk_ratio']}`")
        st.write(f"**Overall Soil Fertility Score:** `{soil_res['fertility_score']} / 100`")

        # NPK Bar Chart
        npk_df = pd.DataFrame([
            {"Nutrient": "Nitrogen (N)", "Value": inputs["N"], "Dataset Median": 37.0, "Unit": "kg/ha"},
            {"Nutrient": "Phosphorus (P)", "Value": inputs["P"], "Dataset Median": 51.0, "Unit": "kg/ha"},
            {"Nutrient": "Potassium (K)", "Value": inputs["K"], "Dataset Median": 32.0, "Unit": "kg/ha"},
        ])
        fig_npk = go.Figure()
        fig_npk.add_trace(go.Bar(name="Your Field Value", x=npk_df["Nutrient"], y=npk_df["Value"], marker_color="#2e7d32"))
        fig_npk.add_trace(go.Bar(name="Benchmark Median", x=npk_df["Nutrient"], y=npk_df["Dataset Median"], marker_color="#90caf9"))
        fig_npk.update_layout(barmode="group", height=300, margin=dict(l=20, r=20, t=30, b=30), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_npk, use_container_width=True)

    with s_col2:
        st.markdown("#### 🔬 Soil Reaction & Chemical Agronomy")
        st.markdown(f"""
        <div style="background-color: #f8faf8; border-left: 5px solid {ph_info['color']}; padding: 1.2rem; border-radius: 8px;">
            <h4 style="color: {ph_info['color']}; margin-top: 0;">{ph_info['classification']} (pH {inputs['ph']})</h4>
            <p><b>Bioavailability & Risk:</b> {ph_info['risk']}</p>
            <p><b>Recommended Management:</b> {ph_info['recommendation']}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 🌿 Nutrient Physiological Roles")
        st.markdown(f"""
        - **Nitrogen (N):** {soil_res['nitrogen']['role']}
        - **Phosphorus (P):** {soil_res['phosphorus']['role']}
        - **Potassium (K):** {soil_res['potassium']['role']}
        """)


# ==============================================================================
# 4. ENVIRONMENTAL ANALYSIS PAGE
# ==============================================================================
elif nav_choice == "⛅ Environmental Analysis":
    st.markdown('<div class="main-header">⛅ Environmental & Microclimate Assessment</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Atmospheric Moisture, Thermal Regime, and Abiotic Stress Diagnostics</div>', unsafe_allow_html=True)

    inputs = st.session_state.inputs
    crop_ctx = st.session_state.prediction
    env_res = analyze_environmental_conditions(inputs["temperature"], inputs["humidity"], inputs["rainfall"], crop_ctx)

    st.info(f"📋 **Current Climate Telemetry:** Temperature={inputs['temperature']}°C, Humidity={inputs['humidity']}%, Rainfall={inputs['rainfall']} mm")

    ec1, ec2, ec3, ec4 = st.columns(4)
    with ec1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Temperature</div>
            <div class="metric-val">{inputs['temperature']} °C</div>
            <div class="metric-sub">Thermal Regime</div>
        </div>
        """, unsafe_allow_html=True)
    with ec2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Relative Humidity</div>
            <div class="metric-val">{inputs['humidity']} %</div>
            <div class="metric-sub">Atmospheric Moisture</div>
        </div>
        """, unsafe_allow_html=True)
    with ec3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Rainfall Depth</div>
            <div class="metric-val">{inputs['rainfall']} mm</div>
            <div class="metric-sub">Precipitation Inflow</div>
        </div>
        """, unsafe_allow_html=True)
    with ec4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Microclimate Index</div>
            <div class="metric-val">{env_res['microclimate_index']}/100</div>
            <div class="metric-sub">{env_res['suitability_assessment']['overall_suitability']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    e_col1, e_col2 = st.columns(2)

    with e_col1:
        st.markdown("#### 🌐 Agro-Climatic Regime Classification")
        regime = env_res["climate_regime"]
        st.markdown(f"""
        <div style="background-color: #f8faf8; border-left: 5px solid {regime['color']}; padding: 1.2rem; border-radius: 8px;">
            <h4 style="color: {regime['color']}; margin-top: 0;">{regime['classification']}</h4>
            <p>{regime['description']}</p>
            <p><b>Ecological Crop Affinity:</b> {regime['crops_suited']}</p>
        </div>
        """, unsafe_allow_html=True)

    with e_col2:
        st.markdown("#### ⚠️ Abiotic Stress Diagnostics")
        stress = env_res["stress_analysis"]
        for rec in stress["recommendations"]:
            st.warning(f"📌 {rec}") if "risk" in rec.lower() or "stress" in rec.lower() or "deficit" in rec.lower() or "excessive" in rec.lower() else st.success(f"✅ {rec}")


# ==============================================================================
# 5. INSIGHTS PAGE
# ==============================================================================
elif nav_choice == "💡 Insights":
    st.markdown('<div class="main-header">💡 Agronomic & Model Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Feature Importances, Crop Profiles, and Model Benchmarks</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Feature Importance (XAI)", "🌿 Crop Benchmark Profiles", "📈 Model Benchmarking"])

    with tab1:
        st.markdown("#### Global Feature Attribution (Gini Importance)")
        st.write("Relative influence of each agronomic parameter in the Random Forest classification architecture:")

        feat_imp = model_metadata.get("metrics", {}).get("feature_importances", {})
        if not feat_imp:
            # Fallback to model importances if needed
            from src.explainability import get_model_feature_importance
            feat_imp = get_model_feature_importance(model, model_metadata["features"])

        imp_df = pd.DataFrame(list(feat_imp.items()), columns=["Feature", "Importance Score"])
        imp_df["Percentage"] = (imp_df["Importance Score"] * 100).round(2)
        
        fig_imp = px.bar(
            imp_df,
            x="Percentage",
            y="Feature",
            orientation="h",
            text=imp_df["Percentage"].apply(lambda v: f"{v:.2f}%"),
            color="Percentage",
            color_continuous_scale="Viridis",
        )
        fig_imp.update_layout(yaxis=dict(autorange="reversed"), height=360, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_imp, use_container_width=True)

    with tab2:
        st.markdown("#### Empirical Crop Benchmark Profiles")
        profiles = get_crop_profiles(df_raw)
        selected_crop = st.selectbox("Select Crop Species to Inspect", sorted(list(profiles.keys())), index=0)
        
        c_prof = profiles[selected_crop]
        prof_df = pd.DataFrame(c_prof).T
        prof_df = prof_df[["mean", "std", "min", "median", "max", "q25", "q75"]]
        prof_df.columns = ["Mean", "Std Dev", "Min", "Median", "Max", "25th %ile", "75th %ile"]
        st.dataframe(prof_df, use_container_width=True)

        if st.session_state.prediction:
            st.markdown("##### 🔍 Alignment with Latest Inputs")
            from src.explainability import generate_crop_comparison_explanation
            comp_df = generate_crop_comparison_explanation(st.session_state.inputs, st.session_state.prediction)
            st.dataframe(comp_df, use_container_width=True)

    with tab3:
        st.markdown("#### Rigorous Multi-Model Benchmark Comparison")
        st.write("All models trained and cross-validated with 5-Fold Stratified K-Fold on the 2,200 Kaggle dataset:")

        comp_file = os.path.join(PROJECT_ROOT, "models", "model_comparison.json")
        if os.path.exists(comp_file):
            with open(comp_file, "r") as f:
                bench_data = json.load(f)
            b_df = pd.DataFrame(bench_data)
            display_cols = ["model_name", "feature_set", "test_accuracy", "test_f1_weighted", "cv_f1_mean", "cv_f1_std"]
            b_df_sub = b_df[display_cols].copy()
            b_df_sub.columns = ["Model Architecture", "Feature Set", "Test Accuracy", "Test F1 (Weighted)", "5-Fold CV F1 Mean", "5-Fold CV F1 Std"]
            st.dataframe(b_df_sub, use_container_width=True)
        else:
            st.info("Benchmark data loading from model_results.json...")


# ==============================================================================
# 6. PREDICTION HISTORY PAGE
# ==============================================================================
elif nav_choice == "📜 Prediction History":
    st.markdown('<div class="main-header">📜 Prediction History & Audit Trail</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Persistent SQLite Records of Field Audits and Diagnostic Outputs</div>', unsafe_allow_html=True)

    history_df = export_predictions_to_dataframe()

    if not history_df.empty:
        # Summary Metrics
        h1, h2, h3 = st.columns(3)
        with h1:
            st.metric("Total Logged Audits", f"{len(history_df)}")
        with h2:
            top_crop = history_df["predicted_crop"].mode()[0].capitalize() if not history_df.empty else "N/A"
            st.metric("Most Recommended Crop", top_crop)
        with h3:
            avg_conf = history_df["confidence"].mean() * 100
            st.metric("Average Confidence", f"{avg_conf:.2f}%")

        st.markdown("---")

        # History Table Display
        display_df = history_df[["id", "timestamp", "notes", "predicted_crop", "confidence", "n", "p", "k", "temperature", "humidity", "ph", "rainfall", "model_used"]].copy()
        display_df.columns = ["ID", "Timestamp", "Field Notes", "Recommended Crop", "Confidence", "N", "P", "K", "Temp (°C)", "Humidity (%)", "pH", "Rainfall (mm)", "Model"]
        display_df["Recommended Crop"] = display_df["Recommended Crop"].str.capitalize()
        display_df["Confidence"] = (display_df["Confidence"] * 100).round(2).astype(str) + "%"

        st.dataframe(display_df, use_container_width=True)

        act_col1, act_col2 = st.columns(2)
        with act_col1:
            csv_bytes = history_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export History to CSV",
                data=csv_bytes,
                file_name="AgriSmart_Prediction_History.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with act_col2:
            if st.button("🗑️ Clear Prediction History", use_container_width=True):
                clear_history()
                st.success("History cleared successfully.")
                st.rerun()
    else:
        st.info("No predictions logged yet. Run your first analysis in the **Crop Analysis** tab.")


# ==============================================================================
# 7. ABOUT PAGE
# ==============================================================================
elif nav_choice == "ℹ️ About":
    st.markdown('<div class="main-header">ℹ️ About AgriSmart AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Intelligent Crop Suitability & Soil Analysis Decision-Support System</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 🎯 System Purpose & Objectives
    **AgriSmart AI** is an agricultural decision-support application built to bridge agronomic science and machine learning.
    By evaluating the physical and chemical micro-environment of an agricultural plot, the system assists farmers, agronomists, 
    and agricultural extension workers in selecting crop species with the highest probability of optimal yield and resilience.

    ### 🏗️ Technical Architecture & Methodology
    1. **Data Ingestion & Integrity:** Grounded in 2,200 non-synthetic agricultural benchmark observations across 22 crop species.
    2. **Machine Learning Pipeline:** Evaluates 4 candidate model families (Logistic Regression, K-Nearest Neighbors, Decision Trees, Random Forests) using 5-Fold Stratified Cross-Validation.
    3. **Champion Architecture:** Random Forest Ensemble delivering **99.55% test accuracy** and **99.49% cross-validation F1-score**.
    4. **Explainability & Diagnostics:** Integrates feature attribution, chemical soil stoichiometry, and agro-climatic stress indicators.
    5. **Data Governance & Auditability:** Every prediction is stored locally in an embedded SQLite database with complete input tracking.

    ### 📚 Dataset Citation
    - **Dataset:** *Crop Recommendation Dataset* (Kaggle Benchmark)
    - **Total Records:** 2,200 balanced samples (100 per crop species)
    - **Variables:** Nitrogen (N), Phosphorus (P), Potassium (K), Temperature (°C), Humidity (%), Soil pH, Rainfall (mm)
    """)
