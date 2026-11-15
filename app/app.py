
"""SEPSIS-AI dashboard.
Run using: python -m streamlit run app/app.py

Academic prototype - not a substitute for professional medical assessment.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config as C
from src.synthetic_demo import explain, what_changed, THRESHOLD


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="SEPSIS-AI",
    layout="wide"
)

st.warning(
    "Academic prototype. Not a substitute for professional medical assessment."
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

from pathlib import Path

REAL_DATA_ROOT = C.ROOT

@st.cache_data
def load_tables(dataset_mode):
    if dataset_mode == "Synthetic demo":
        rs = C.read_table(C.TABLES / "risk_series_demo.parquet")
        vit = C.read_table(C.DATA_PROC / "demo.parquet")
        return rs, None, vit

    if dataset_mode == "PhysioNet research":
        risk_path = C.TABLES / "risk_series.parquet"
        test_path = C.ROOT / "demo_data" / "test.pkl"
        ess_path = C.TABLES / "ess_scores.parquet"

        if not risk_path.exists() or not test_path.exists():
            raise FileNotFoundError("PhysioNet research files are missing.")

        rs = C.read_table(risk_path)
        vit = pd.read_pickle(test_path)
        ess = C.read_table(ess_path) if ess_path.exists() else None

        test_ids = set(vit["patient_id"].unique())
        rs = rs[rs["patient_id"].isin(test_ids)].copy()

        if ess is not None:
            ess = ess[ess["patient_id"].isin(test_ids)].copy()

        return rs, ess, vit

    raise ValueError("Unknown dataset mode")


available_modes = ["Synthetic demo"]

if (
    (C.TABLES / "risk_series.parquet").exists()
    and (C.ROOT / "demo_data" / "test.pkl").exists()
):
    available_modes.append("PhysioNet research")

dataset_mode = st.sidebar.selectbox(
    "Dataset",
    available_modes,
    index=available_modes.index("PhysioNet research") if "PhysioNet research" in available_modes else 0,
    key="dataset_mode"
)

try:
    rs, ess, vit = load_tables(dataset_mode)
except Exception as exc:
    st.error(f"Could not load selected dataset: {exc}")
    st.stop()

if dataset_mode == "Synthetic demo":
    thr = THRESHOLD
else:
    from src.models import load_thresholds
    thr = load_thresholds()["xgb"]

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

page = st.sidebar.radio(
    "Page",
    [
        "Patient view",
        "Risk trajectory (RTV)",
        "Synthetic explanations",
        "Vitals",
        "ESS Stability",
        "SHAP Explainability",
        "Model comparison",
        "About / limitations"
    ]
)


# --------------------------------------------------
# PATIENT SELECTION
# --------------------------------------------------

septic = rs[
    rs.patient_id.isin(
        vit[vit.label == 1].patient_id.unique()
    )
].patient_id.unique()

pool = st.sidebar.radio(
    "Patient pool",
    ["Septic", "Non-septic"]
)

ids = (
    sorted(septic)
    if pool == "Septic"
    else sorted(set(rs.patient_id) - set(septic))
)

if len(ids) == 0:
    st.error("No patients available in the selected patient pool.")
    st.stop()

pid = st.sidebar.selectbox("Select Patient ID", options=ids, format_func=lambda x: str(x), key=f"selected_patient_{dataset_mode}")

p = rs[rs.patient_id == pid].reset_index(drop=True)

if p.empty:
    st.error("No risk data available for this patient.")
    st.stop()

hour = st.sidebar.slider(
    "ICU hour",
    int(p.hour.min()),
    int(p.hour.max()),
    int(p.hour.max())
)

cur = p[p.hour == hour].iloc[0]


# --------------------------------------------------
# RISK CLASSIFICATION
# --------------------------------------------------

def label(r):
    if r >= thr:
        return "High"
    elif r >= thr / 2:
        return "Medium"
    else:
        return "Low"


# --------------------------------------------------
# PAGE 1: PATIENT VIEW
# --------------------------------------------------

if page == "Patient view":

    st.title("SEPSIS-AI | Patient Risk Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Current risk",
        f"{cur.risk:.1%}"
    )

    c2.metric(
        "Risk level",
        label(cur.risk)
    )

    c3.metric(
        "Alert status",
        "ALERT" if cur.risk >= thr else "No alert"
    )

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=cur.risk * 100,
            number={"suffix": "%"},
            gauge={
                "axis": {
                    "range": [
                        0,
                        max(10, 100 * p.risk.max() * 1.2)
                    ]
                },
                "threshold": {
                    "line": {
                        "color": "red",
                        "width": 3
                    },
                    "value": thr * 100
                }
            }
        )
    )

    fig.update_layout(height=280)

    st.plotly_chart(
        fig,
        width="stretch"
    )


# --------------------------------------------------
# PAGE 2: RISK TRAJECTORY
# --------------------------------------------------

elif page == "Risk trajectory (RTV)":

    st.title("Risk Trajectory Visualization")

    fl = (p.risk >= thr).to_numpy()

    fig = go.Figure(
        go.Scatter(
            x=p.hour,
            y=p.risk,
            name="Risk"
        )
    )

    fig.add_hline(
        y=thr,
        line_dash="dash",
        annotation_text="Alert threshold"
    )

    onset = vit[
        (vit.patient_id == pid) &
        (vit.label == 1)
    ].hour.min()

    if fl.any():

        ta = p.hour[fl].min()

        fig.add_vrect(
            x0=ta,
            x1=p.hour.max(),
            fillcolor="red",
            opacity=0.08,
            line_width=0
        )

        fig.add_vline(
            x=ta,
            line_color="red",
            annotation_text="First alert"
        )

    if not np.isnan(onset):

        fig.add_vline(
            x=onset,
            line_color="green",
            line_dash="dot",
            annotation_text="Onset (label=1)"
        )

        if fl.any():

            st.metric(
                "Lead time (onset - first alert)",
                f"{onset - ta:.0f} h"
            )

    fig.update_layout(
        xaxis_title="ICU hour",
        yaxis_title="Illustrative index",
        height=380
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    if dataset_mode == "Synthetic demo":
        st.info("Synthetic illustrative index only; not a clinical risk prediction.")
        st.subheader(
            f"Descriptive changes over the previous 6 hours (hour {hour})"
        )

        w = what_changed(pid, hour, split="demo")

        c1, c2 = st.columns(2)

        c1.markdown(
            "**Pushed risk up**\n\n" +
            "\n\n".join(
                f"- {l}"
                for l in w["text"]
                if "+" in l.split(":")[-1]
            ) or "-"
        )

        c2.markdown(
            "**Pushed risk down**\n\n" +
            "\n\n".join(
                f"- {l}"
                for l in w["text"]
                if "-" in l.split(":")[-1].split("(")[0]
            ) or "-"
        )


# --------------------------------------------------
# PAGE 3: EXPLANATION (SHAP + ESS)
# --------------------------------------------------

elif page == "Synthetic explanations":

    if dataset_mode != "Synthetic demo":
        st.info("This illustrative explanation is only available in Synthetic demo mode.")
        st.stop()

    st.title("Synthetic Feature Explanation")
    e = explain(pid, hour, split="demo")



    top = e["components"].head(12)[::-1]

    fig = go.Figure(
        go.Bar(
            x=top.values,
            y=list(top.index),
            orientation="h",
            marker_color=[
                "#c1121f" if v > 0 else "#2a6f97"
                for v in top.values
            ]
        )
    )

    fig.update_layout(
        title="Illustrative rule components (not model contributions)",
        height=450
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    st.info("Illustrative synthetic rule components only. Not SHAP, not model explanations, and not clinical evidence.")

    if ess is not None:

        row = ess[
            (ess.patient_id == pid) &
            (ess.hour == hour)
        ]

        if len(row):

            r = row.iloc[0]

            st.success(
                f"Explanation confidence: **{r.level}** "
                f"(ESS = {r.ess:.2f}; "
                f"seed {r.ess_seed:.2f}, "
                f"noise {r.ess_noise:.2f}, "
                f"cross-model {r.ess_model:.2f})"
            )

        else:

            st.info(
                "ESS is computed on a sample of test rows; "
                "this patient-hour is not in the sample. "
                "Pick a row from the list below."
            )

            st.dataframe(
                ess[
                    ["patient_id", "hour", "ess", "level"]
                ].head(30)
            )

    else:
        st.info(
            "ESS stability is not available. These synthetic rule components are illustrative only."
        )


# --------------------------------------------------
# PAGE 4: VITALS
# --------------------------------------------------

elif page == "Vitals":

    st.title("Patient Vital Signs")

    v = vit[vit.patient_id == pid]

    for col in [
        "HR",
        "Temp",
        "Resp",
        "SBP",
        "MAP",
        "O2Sat"
    ]:

        st.subheader(col)

        st.line_chart(
            v.set_index("hour")[col],
            height=140
        )

    st.caption(
        "Values are forward-filled (last known measurement)."
    )


# --------------------------------------------------
# PAGE 5: MODEL COMPARISON
# --------------------------------------------------

elif page == "Model comparison":

    st.title("Model Performance Comparison")

    st.info(
        "This public demonstration uses fictional synthetic patient records. "
        "The original model evaluation metrics, performance curves, and "
        "alert-budget results are not displayed here because they were "
        "generated from a separate evaluation dataset."
    )

    st.warning(
        "The synthetic demonstration data is not a benchmark dataset. "
        "No model accuracy, clinical performance, or generalisation claim "
        "should be inferred from the displayed synthetic patient examples."
    )

    st.subheader("Models included in the research project")
    st.write(
        "- Logistic Regression (LR)\n"
        "- Random Forest (RF)\n"
        "- XGBoost (XGB)"
    )

    st.caption(
        "A separately documented and reviewed research evaluation "
        "may be presented in a future version."
    )


# --------------------------------------------------
# PAGE: ESS STABILITY
# --------------------------------------------------

elif page == "ESS Stability":

    st.title("Explanation Stability Score (ESS)")

    if dataset_mode != "PhysioNet research" or ess is None:
        st.info("ESS research results are available in PhysioNet research mode.")
    else:
        st.caption(
            "Original research ESS sample. These 100 sampled patient-hour "
            "observations are not the full test cohort."
        )

        st.metric("ESS observations", len(ess))
        st.metric("Mean ESS", f"{ess['ess'].mean():.3f}")
        st.metric("Median ESS", f"{ess['ess'].median():.3f}")

        st.subheader("ESS distribution")
        fig = go.Figure(go.Histogram(x=ess["ess"], nbinsx=15))
        fig.update_layout(xaxis_title="ESS", yaxis_title="Observations")
        st.plotly_chart(fig, width="stretch")

        st.subheader("Stability components")
        components = [
            c for c in [
                "ess_spearman", "ess_jaccard",
                "ess_seed", "ess_noise", "ess_model"
            ] if c in ess.columns
        ]
        if components:
            means = ess[components].mean().sort_values()
            fig = go.Figure(go.Bar(
                x=means.values, y=means.index, orientation="h"
            ))
            fig.update_layout(
                title="Mean ESS component scores",
                xaxis_title="Mean score"
            )
            st.plotly_chart(fig, width="stretch")

        st.subheader("ESS by stability level")
        st.dataframe(
            ess.groupby("level")["ess"].agg(
                ["count", "mean", "median", "min", "max"]
            ).round(4),
            width="stretch"
        )

        tables = C.TABLES
        validation = tables / "ess_validation.csv"
        unstable = tables / "ess_unstable_features.csv"

        if validation.exists():
            st.subheader("ESS validation summary")
            st.dataframe(pd.read_csv(validation), width="stretch")

        if unstable.exists():
            st.subheader("Features with highest instability")
            u = pd.read_csv(unstable)
            u = u.rename(columns={"Unnamed: 0": "feature"})
            st.dataframe(u, width="stretch")

        st.subheader("Sampled patient-hour ESS records")
        st.dataframe(ess, width="stretch")


# --------------------------------------------------
# PAGE: SHAP EXPLAINABILITY
# --------------------------------------------------

elif page == "SHAP Explainability":

    st.title("SHAP Explainability")

    if dataset_mode != "PhysioNet research":
        st.info("Original SHAP research outputs are available in PhysioNet mode.")
    else:
        tables = C.TABLES
        figures = C.FIGURES
        importance_path = tables / "shap_importance.csv"

        if importance_path.exists():
            shap_df = pd.read_csv(importance_path)
            shap_df = shap_df.rename(
                columns={"Unnamed: 0": "feature"}
            ).sort_values("mean_abs_shap", ascending=False)

            st.caption(
                "Global mean absolute SHAP importance from the original "
                "research evaluation. This is not a patient-specific explanation."
            )

            st.subheader("Top 20 influential features")
            top = shap_df.head(20).sort_values("mean_abs_shap")
            fig = go.Figure(go.Bar(
                x=top["mean_abs_shap"],
                y=top["feature"],
                orientation="h"
            ))
            fig.update_layout(
                height=650,
                xaxis_title="Mean absolute SHAP value",
                yaxis_title="Feature"
            )
            st.plotly_chart(fig, width="stretch")

            with st.expander("View all 115 SHAP features"):
                st.dataframe(shap_df, width="stretch")
        else:
            st.error("SHAP importance CSV not found.")

        st.subheader("Original SHAP research figures")

        image_files = [
            ("SHAP bar plot", "shap_bar.png"),
            ("SHAP beeswarm plot", "shap_beeswarm.png"),
            ("False-negative waterfall", "shap_waterfall_false_negative.png"),
            ("High-risk true-positive waterfall",
             "shap_waterfall_high_risk_true_positive.png"),
            ("Low-risk true-negative waterfall",
             "shap_waterfall_low_risk_true_negative.png"),
        ]

        for title, filename in image_files:
            image_path = figures / filename
            if image_path.exists():
                st.markdown(f"**{title}**")
                st.image(str(image_path), width="stretch")


# --------------------------------------------------
# PAGE: MODEL COMPARISON AND RESEARCH EVALUATION
# --------------------------------------------------

elif page == "Model comparison":

    st.title("Model Performance Comparison")

    if dataset_mode != "PhysioNet research":
        st.info("Original model evaluation results are available in PhysioNet mode.")
    else:
        tables = C.TABLES

        for title, filename in [
            ("Test-set model comparison", "model_comparison_test.csv"),
            ("Validation-set model comparison", "model_comparison_val.csv"),
            ("Subgroup evaluation (test)", "subgroups_xgb_test.csv"),
            ("Subgroup evaluation (validation)", "subgroups_xgb_val.csv"),
            ("Alert-budget evaluation", "alert_budget_table.csv"),
            ("RTV lead-time comparison", "rtv_lead_time.csv"),
        ]:
            path = tables / filename
            if path.exists():
                st.subheader(title)
                st.dataframe(pd.read_csv(path), width="stretch")

# PAGE 6: ABOUT AND LIMITATIONS
# --------------------------------------------------

else:

    st.title("About SEPSIS-AI")

    st.markdown("""
    **Public demonstration data**

    This dashboard uses 20 fictional synthetic patient profiles,
    with 24 hourly records per profile. The records were generated
    specifically for demonstrating the dashboard interface.

    **Synthetic demonstration index**

    The displayed index is generated from transparent, hand-written
    illustrative rules. It is not produced by a trained model and is
    not a probability of sepsis.

    **Descriptive explanations**

    Feature changes are descriptive synthetic examples, not SHAP values,
    model explanations, or clinical causes.

    **ESS**

    Explanation stability has not yet been evaluated on this
    synthetic demonstration dataset.

    **Important limitations**

    - Synthetic records are fictional and are not real patient cases.
    - Synthetic examples are not a model performance benchmark.
    - No clinical accuracy or generalisation claim should be inferred.
    - Synthetic rule components are not SHAP explanations.
    - This academic prototype is not validated for clinical use.
    - It must not be used for diagnosis or treatment decisions.
    """)

    st.caption(
        "The public demonstration is separate from the original "
        "research evaluation dataset and its patient-level outputs."
    )


