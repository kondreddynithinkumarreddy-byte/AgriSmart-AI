"""
AgriSmart AI - Web Dashboard Application
"""

import os
import sys
import json
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data_processing import (
    load_and_validate_dataset,
    get_dataset_summary_statistics,
    detect_outliers_iqr,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
)
from src.prediction import (
    load_production_model,
    load_model_metadata,
    predict_crop_suitability,
    validate_input_features,
    simulate_what_if_scenario,
    EXPECTED_RANGES,
)
from src.soil_analysis import analyze_soil_nutrients
from src.environmental_analysis import analyze_environmental_conditions
from src.explainability import get_model_feature_importance
from src.database import (
    init_db,
    log_prediction,
    get_prediction_count,
    export_predictions_to_dataframe,
)
from src.report_generator import generate_pdf_report

st.set_page_config(
    page_title="AgriSmart AI – Crop & Soil Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1b5e20;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8faf8;
        border-left: 5px solid #2e7d32;
        padding: 1.2rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }
    .result-hero {
        background: linear-gradient(135deg, #2e7d32 0%, #1b5e20 100%);
        color: white;
        padding: 1.8rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 10px rgba(27, 94, 32, 0.25);
    }
    .stButton>button {
        background-color: #2e7d32;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        border: none;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_cached_data():
    df, summary = load_and_validate_dataset()
    return df, summary


@st.cache_resource
def get_cached_model():
    model = load_production_model()
    metadata = load_model_metadata()
    return model, metadata


init_db()

df_raw, data_summary = get_cached_data()
model, model_metadata = get_cached_model()

st.sidebar.image("https://images.unsplash.com/photo-1500937386664-56d1dfef3854?w=600&auto=format&fit=crop&q=80", width="stretch")
st.sidebar.markdown("### 🌾 **AgriSmart AI**")
st.sidebar.caption("Intelligent Crop Suitability & Soil System")

nav_choice = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏠 Executive Overview",
        "📈 Exploratory Data Analysis (EDA)",
        "🌱 Crop Suitability Prediction",
        "🧪 Soil Nutrient Diagnostic",
        "⛅ Environmental Analysis",
        "💡 Explainable AI (XAI)",
        "🔬 What-If Scenario Simulator",
        "📊 Model Benchmark & Comparison",
        "📜 Prediction History & Audit",
        "📖 Academic Documentation & Viva",
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    f"**Active Model:** {model_metadata.get('model_name', 'Random Forest')}\n\n"
    f"**Test Accuracy:** {model_metadata['metrics']['accuracy']*100:.2f}%\n\n"
    f"**Database Records:** {get_prediction_count()}"
)

# 1. Executive Overview
if nav_choice == "🏠 Executive Overview":
    st.markdown('<div class="main-header">🌾 AgriSmart AI – Executive Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">An Explainable Machine Learning System for Crop Suitability Classification and Soil Diagnostics</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Dataset Records", f"{len(df_raw):,}", "Authoritative 2,200")
    with col2:
        st.metric("Crop Classes Supported", f"{data_summary['class_count']}", "100% Balanced (100 ea.)")
    with col3:
        st.metric("Champion Test Accuracy", f"{model_metadata['metrics']['accuracy']*100:.2f}%", "5-Fold CV: 99.49%")
    with col4:
        st.metric("Logged Audits", f"{get_prediction_count()}", "SQLite Persistent")

    st.markdown("---")
    st.write("""
    **AgriSmart AI** is a decision-support platform designed to address agricultural misallocation and unsustainable fertilizer application.
    By synthesizing chemical soil conditions (N, P, K, pH) with atmospheric meteorological indicators (Temperature, Relative Humidity, Rainfall),
    the system predicts the single most suitable agronomic crop while providing complete interpretability, nutrient balance scoring, and abiotic stress risk analysis.
    """)

# 2. EDA
elif nav_choice == "📈 Exploratory Data Analysis (EDA)":
    st.markdown('<div class="main-header">📈 Exploratory Data Analysis (EDA)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Authoritative Empirical Dataset Inspection & Statistical Characterization</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset Overview", "📉 Feature Distributions", "🔗 Correlations", "⚠️ Outliers (IQR)"])

    with tab1:
        st.subheader("Authoritative Dataset Summary")
        st.dataframe(get_dataset_summary_statistics(df_raw), width="stretch")

    with tab2:
        selected_feature = st.selectbox("Select Parameter to Inspect", NUMERICAL_FEATURES)
        fig_hist = px.histogram(df_raw, x=selected_feature, color=TARGET_COLUMN, nbins=30, title=f"Distribution of {selected_feature} across Crops")
        st.plotly_chart(fig_hist, width="stretch")

    with tab3:
        corr_matrix = df_raw[NUMERICAL_FEATURES].corr()
        fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale="Viridis", title="Pearson Correlation Matrix")
        st.plotly_chart(fig_corr, width="stretch")

    with tab4:
        outlier_data = detect_outliers_iqr(df_raw)
        st.dataframe(pd.DataFrame(outlier_data).T, width="stretch")

# 3. Prediction
elif nav_choice == "🌱 Crop Suitability Prediction":
    st.markdown('<div class="main-header">🌱 Crop Suitability Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-Time Multi-Parameter Soil & Climate Inference Engine</div>', unsafe_allow_html=True)

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            n_input = st.number_input("Nitrogen (N) [kg/ha]", min_value=0.0, max_value=250.0, value=90.0, step=1.0)
            p_input = st.number_input("Phosphorus (P) [kg/ha]", min_value=0.0, max_value=250.0, value=42.0, step=1.0)
            k_input = st.number_input("Potassium (K) [kg/ha]", min_value=0.0, max_value=300.0, value=43.0, step=1.0)
        with col2:
            temp_input = st.number_input("Temperature [°C]", min_value=0.0, max_value=60.0, value=20.87, step=0.1)
            hum_input = st.number_input("Relative Humidity [%]", min_value=0.0, max_value=100.0, value=82.0, step=0.1)
            ph_input = st.number_input("Soil pH [0.0 - 14.0]", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
        with col3:
            rain_input = st.number_input("Rainfall [mm]", min_value=0.0, max_value=600.0, value=202.9, step=1.0)
            notes_input = st.text_input("Field ID / Operational Notes", value="Field Zone A-1")
            submit_btn = st.form_submit_button("🌾 PREDICT SUITABLE CROP")

    if submit_btn:
        input_dict = {"N": n_input, "P": p_input, "K": k_input, "temperature": temp_input, "humidity": hum_input, "ph": ph_input, "rainfall": rain_input}
        val_res = validate_input_features(input_dict)
        if not val_res["is_valid"]:
            for err in val_res["errors"]:
                st.error(f"❌ Input Error: {err}")
        else:
            for warning in val_res["warnings"]:
                st.warning(warning)

            pred_res = predict_crop_suitability(input_dict, model=model, metadata=model_metadata)
            soil_res = analyze_soil_nutrients(n_input, p_input, k_input, ph_input)
            env_res = analyze_environmental_conditions(temp_input, hum_input, rain_input, pred_res["predicted_crop"])
            log_id = log_prediction(input_dict, pred_res, notes=notes_input)

            st.markdown(f"""
            <div class="result-hero">
                <div style="font-size: 1.1rem; text-transform: uppercase;">Recommended Crop</div>
                <div style="font-size: 3.5rem; margin: 0.5rem 0;">{pred_res['crop_icon']} {pred_res['predicted_crop'].capitalize()}</div>
                <div style="font-size: 1.25rem;">Confidence: <b>{pred_res['confidence_pct']}%</b> | Engine: <b>{pred_res['model_used']}</b></div>
            </div>
            """, unsafe_allow_html=True)

            top5_df = pd.DataFrame(pred_res["top_recommendations"])
            fig_top5 = px.bar(top5_df, x="probability_pct", y="crop", orientation="h", text="probability_pct", color="probability_pct", color_continuous_scale="greens")
            fig_top5.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_top5, width="stretch")

            pdf_bytes = generate_pdf_report(input_dict, pred_res, soil_res, env_res)
            st.download_button(
                label="📄 Download Diagnostic PDF Dossier",
                data=pdf_bytes,
                file_name=f"AgriSmart_Report_{pred_res['predicted_crop']}_{log_id}.pdf",
                mime="application/pdf",
                width="stretch",
            )

# 4. Soil Analysis
elif nav_choice == "🧪 Soil Nutrient Diagnostic":
    st.markdown('<div class="main-header">🧪 Soil Nutrient & Stoichiometry Diagnostic</div>', unsafe_allow_html=True)
    n_val = st.number_input("Nitrogen (N)", 0.0, 250.0, 90.0)
    p_val = st.number_input("Phosphorus (P)", 0.0, 250.0, 42.0)
    k_val = st.number_input("Potassium (K)", 0.0, 300.0, 43.0)
    ph_val = st.number_input("Soil pH", 0.0, 14.0, 6.5)

    soil_diag = analyze_soil_nutrients(n_val, p_val, k_val, ph_val)
    st.metric("Total NPK Load", f"{soil_diag['npk_total']} kg/ha")
    st.metric("N:P:K Stoichiometry", f"{soil_diag['npk_ratio']}")
    st.info(f"Fertility Score: {soil_diag['fertility_score']}/100 | Reaction: {soil_diag['ph_analysis']['classification']}")

# 5. Environmental Analysis
elif nav_choice == "⛅ Environmental Analysis":
    st.markdown('<div class="main-header">⛅ Environmental & Microclimate Analysis</div>', unsafe_allow_html=True)
    t_val = st.number_input("Ambient Temperature (°C)", 0.0, 50.0, 25.0)
    h_val = st.number_input("Relative Humidity (%)", 0.0, 100.0, 75.0)
    r_val = st.number_input("Precipitation (mm)", 0.0, 500.0, 180.0)
    env_diag = analyze_environmental_conditions(t_val, h_val, r_val)
    st.success(f"Climate Regime: {env_diag['climate_regime']['classification']}")
    st.write(env_diag['climate_regime']['description'])

# 6. XAI
elif nav_choice == "💡 Explainable AI (XAI)":
    st.markdown('<div class="main-header">💡 Explainable AI (XAI) & Attribution</div>', unsafe_allow_html=True)
    importances = get_model_feature_importance(model, model_metadata["features"])
    imp_df = pd.DataFrame(list(importances.items()), columns=["Feature", "Importance Score"])
    fig_imp = px.bar(imp_df, x="Importance Score", y="Feature", orientation="h", color="Importance Score", color_continuous_scale="Viridis")
    fig_imp.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_imp, width="stretch")

# 7. What-If
elif nav_choice == "🔬 What-If Scenario Simulator":
    st.markdown('<div class="main-header">🔬 What-If Scenario Simulator</div>', unsafe_allow_html=True)
    feature_to_perturb = st.selectbox("Feature to Vary", NUMERICAL_FEATURES, index=6)
    base_sample = {"N": 90.0, "P": 42.0, "K": 43.0, "temperature": 22.0, "humidity": 80.0, "ph": 6.5, "rainfall": 150.0}
    bounds = EXPECTED_RANGES[feature_to_perturb]
    sim_steps = simulate_what_if_scenario(base_sample, feature_to_perturb, bounds["min"], bounds["max"], steps=10)
    st.dataframe(pd.DataFrame(sim_steps), width="stretch")

# 8. Model Benchmark
elif nav_choice == "📊 Model Benchmark & Comparison":
    st.markdown('<div class="main-header">📊 Model Benchmarking & Multi-Model Evaluation</div>', unsafe_allow_html=True)
    comp_file = os.path.join(ROOT_DIR, "models", "model_comparison.json")
    if os.path.exists(comp_file):
        with open(comp_file, "r") as f:
            bench_data = json.load(f)
        b_df = pd.DataFrame(bench_data)[["config_name", "model_name", "feature_set", "test_accuracy", "test_f1_weighted", "cv_f1_mean", "cv_f1_std"]]
        st.dataframe(b_df, width="stretch")

# 9. History
elif nav_choice == "📜 Prediction History & Audit":
    st.markdown('<div class="main-header">📜 Prediction History & Audit Trail</div>', unsafe_allow_html=True)
    history_df = export_predictions_to_dataframe()
    if not history_df.empty:
        st.dataframe(history_df, width="stretch")
        csv_data = history_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Export History to CSV", csv_data, "agrismart_history.csv", "text/csv")
    else:
        st.info("No predictions logged yet.")

# 10. Viva
elif nav_choice == "📖 Academic Documentation & Viva":
    st.markdown('<div class="main-header">📖 Academic Documentation & Viva Preparation</div>', unsafe_allow_html=True)
    st.markdown("""
    ### 🎓 Internship Capstone Viva Defense
    - **Champion Model:** Random Forest on 7 Physical Features (Test Accuracy: 99.55%, 5-Fold CV: 99.49%).
    - **Data Leakage Prevention:** Preprocessing steps fitted strictly on training folds inside Pipelines.
    - **Explainability:** Global Gini importance rankings & empirical crop benchmark comparisons.
    """)
