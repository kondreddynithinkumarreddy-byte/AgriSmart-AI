# AgriSmart AI — Internship Capstone Presentation Slide Deck

**Title:** AgriSmart AI: Intelligent Crop Suitability & Soil Analysis System  
**Presenter:** Engineering Internship Candidate  
**Domain:** Machine Learning in Precision Agriculture  

---

## Slide 1: Title Slide
- **Project Name:** AgriSmart AI
- **Subtitle:** Intelligent Crop Suitability & Soil Analysis System
- **Focus:** Data-Driven Crop Recommendation, Soil Stoichiometry & Agro-Climatic Stress Diagnostics
- **Target Audience:** Farmers, Agronomists, Agricultural Extension Officers

---

## Slide 2: Problem Statement & Agricultural Motivation
- **Core Challenge:** Sub-optimal crop selection leads to significant yield deficits and financial losses for farmers.
- **Resource Mismanagement:** Uncalibrated NPK fertilizer application degrades soil health and causes environmental runoff.
- **Climate Vulnerability:** Incompatible cropping patterns exacerbate drought and flood risks.
- **Solution:** A real-time ML decision-support system evaluating 7 chemical and environmental variables.

---

## Slide 3: Dataset Architecture & Empirical Validation
- **Source:** Kaggle Crop Recommendation Dataset (2,200 empirical field observations).
- **Target Space:** 22 balanced crop categories (100 observations each, 0 class imbalance).
- **Input Variables ($D=7$):**
  - Soil Chemistry: Nitrogen ($N$), Phosphorus ($P$), Potassium ($K$), Soil pH.
  - Climate & Weather: Temperature ($^\circ\text{C}$), Relative Humidity ($\%$) and Rainfall ($\text{mm}$).
- **Data Quality:** 0 missing values, 0 duplicate records, rigorously verified distributions.

---

## Slide 4: Multi-Model Machine Learning Benchmarking
- **Validation Protocol:** 5-Fold Stratified Cross-Validation + 20% Held-Out Test Split.
- **Leakage Prevention:** All scalers and transformers fitted strictly within cross-validation folds.

| Model Architecture | 5-Fold CV F1 Score | Test Set Accuracy | Test Macro F1 |
| :--- | :---: | :---: | :---: |
| **Random Forest (Champion)** | **99.49%** | **99.55%** | **99.55%** |
| **K-Nearest Neighbors** | 97.84% | 98.18% | 98.16% |
| **Decision Tree** | 97.78% | 97.95% | 97.94% |
| **Logistic Regression** | 97.04% | 97.27% | 97.25% |

---

## Slide 5: Explainable AI & Feature Attribution
- **Gini Feature Importance Hierarchy:**
  1. Rainfall (22.82%) & Humidity (22.10%): Primary hydrological drivers.
  2. Potassium (17.83%) & Phosphorus (14.79%): Essential osmotic and root drivers.
  3. Nitrogen (10.12%), Temperature (7.16%), Soil pH (5.18%).
- **Crop Benchmark Alignment:** Dynamic z-score comparison between field input telemetry and historical crop optima.

---

## Slide 6: Domain Agronomic Diagnostic Engines
- **1. Soil Stoichiometry & NPK Balancing:**
  - Evaluates Total NPK load (kg/ha) and empirical N:P:K ratio.
  - Soil fertility index scoring (0 - 100).
- **2. Soil Reaction & pH Risk Assessment:**
  - Categorizes into 6 agronomic regimes (Extremely Acidic to Strongly Alkaline).
  - Specific chemical remediation advisories (agricultural lime, elemental sulfur, gypsum).
- **3. Microclimate & Abiotic Stress Engine:**
  - Thermal stress (Heat / Chilling).
  - Hydrological stress (Drought / Waterlogging).
  - Pathological foliar fungal disease alerts.

---

## Slide 7: Software Engineering & System Architecture
- **Frontend / UI:** Streamlit with custom responsive styling and intuitive 7-page workflow.
- **Inference Engine:** Low-latency Scikit-learn Pipeline with automated input range validation.
- **Audit & History:** Persistent SQLite storage tracking field IDs, timestamps, inputs, and predictions.
- **Reporting:** Dynamic multi-page PDF generation via ReportLab.
- **Automated Testing:** Comprehensive smoke test suite ensuring 100% component availability.

---

## Slide 8: Live Demonstration & Workflow
- Step 1: Select or enter field telemetry in **Crop Analysis** (or use 1-click presets).
- Step 2: Receive real-time prediction with calibrated probabilities for top 5 crops.
- Step 3: Inspect detailed nutrient ratios in **Soil Analysis**.
- Step 4: Examine climatic compatibility in **Environmental Analysis**.
- Step 5: Download the diagnostic PDF dossier or review historical audits in **Prediction History**.

---

## Slide 9: Conclusion & Impact
- **Accuracy:** 99.55% test accuracy across 22 crop classes.
- **Practical Utility:** Actionable agronomic diagnostics beyond simple predictions.
- **Production-Ready:** Fully functional, modular, tested, and submission-ready code repository.

---

## Slide 10: Q&A / Discussion
*Thank you! Questions and comments are welcome.*
