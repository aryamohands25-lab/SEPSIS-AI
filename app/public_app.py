import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"

st.set_page_config(page_title="SEPSIS-AI | Research Dashboard", layout="wide")

st.title("SEPSIS-AI")
st.caption("Machine Learning-Based Early Sepsis Risk Research")

st.warning(
    "Academic research prototype only. Not a diagnostic tool "
    "or substitute for professional medical assessment."
)

page = st.sidebar.radio(
    "Explore",
    [
        "Model Comparison",
        "Alert Budget Analysis",
        "Global SHAP Importance",
        "About & Limitations"
    ]
)

if page == "Model Comparison":
    st.header("Model Performance")
    st.caption("Validation-set results")

    df = pd.read_csv(TABLES / "model_comparison_val.csv")
    df = df.rename(columns={"Unnamed: 0": "Model"})

    st.dataframe(
        df[["Model", "AUROC", "AUPRC", "recall", "precision", "Brier"]]
        .round(4),
        use_container_width=True,
        hide_index=True
    )

    fig = px.bar(
        df,
        x="Model",
        y=["AUROC", "AUPRC"],
        barmode="group",
        title="Validation AUROC and AUPRC"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "XGBoost achieved validation AUROC 0.831 and AUPRC 0.101. "
        "These are retrospective development results, not clinical validation."
    )

elif page == "Alert Budget Analysis":
    st.header("Alert Budget Analysis")
    st.caption("Retrospective validation-set analysis")

    df = pd.read_csv(TABLES / "alert_budget_table.csv")

    st.dataframe(df.round(3), use_container_width=True, hide_index=True)

    fig = px.line(
        df,
        x="false_alerts_per_10_patient_days",
        y="alerted_%",
        markers=True,
        title="Alert coverage versus false-alert rate"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Lead time is measured relative to the first positive challenge label, "
        "not a verified clinical onset time."
    )

elif page == "Global SHAP Importance":
    st.header("Global Feature Importance")
    st.caption("Mean absolute SHAP values on the validation sample")

    df = pd.read_csv(TABLES / "shap_importance.csv")
    df = df.rename(columns={"Unnamed: 0": "Feature"})
    df = df.head(15).sort_values("mean_abs_shap")

    fig = px.bar(
        df,
        x="mean_abs_shap",
        y="Feature",
        orientation="h",
        title="Top 15 Features"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "SHAP importance describes model attribution, not causation."
    )

elif page == "About & Limitations":
    st.header("About the Project")

    st.write(
        "SEPSIS-AI explores machine-learning approaches for identifying "
        "patterns associated with early sepsis warning using the PhysioNet "
        "2019 Challenge dataset."
    )

    st.subheader("Methods")
    st.markdown("""
    - Logistic Regression, Random Forest and XGBoost
    - Patient-level training, validation and test splits
    - Temporal feature engineering
    - Probability calibration
    - SHAP-based model explanations
    - Alert-budget and lead-time analysis
    """)

    st.subheader("Important Limitations")
    st.markdown("""
    - Retrospective dataset; no prospective clinical validation.
    - Validation metrics do not establish clinical usefulness.
    - Low precision means false alerts remain a major concern.
    - Challenge labels are not equivalent to verified clinical onset.
    - SHAP explanations do not establish causation.
    - This system must not be used to make patient-care decisions.
    """)

    st.subheader("Research Disclaimer")
    st.write(
        "This project is intended for academic and educational purposes only."
    )

st.divider()
st.caption("SEPSIS-AI | Academic Portfolio Project")
