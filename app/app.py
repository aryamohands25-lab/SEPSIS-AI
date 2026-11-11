
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
from src.explain import explain
from src.models import load_thresholds
from src.rtv import what_changed


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

@st.cache_data
def load_tables():
    rs = C.read_table(C.TABLES / "risk_series.parquet")

    ess = (
        C.read_table(C.TABLES / "ess_scores.parquet")
        if (C.TABLES / "ess_scores.parquet").exists()
        else None
    )

    test = C.read_table(
        C.DATA_PROC / "test.parquet",
        columns=[
            "patient_id",
            "hour",
            "label",
            "HR",
            "Temp",
            "Resp",
            "SBP",
            "MAP",
            "O2Sat"
        ]
    )

    return rs, ess, test


@st.cache_resource
def thresholds():
    return load_thresholds()


rs, ess, vit = load_tables()
thr = thresholds()["xgb"]


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

page = st.sidebar.radio(
    "Page",
    [
        "Patient view",
        "Risk trajectory (RTV)",
        "Explanation (SHAP + ESS)",
        "Vitals",
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

pid = st.sidebar.selectbox("Select Patient ID", options=ids, format_func=lambda x: str(x), key="selected_patient")

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
        yaxis_title="Calibrated risk",
        height=380
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    st.subheader(
        f"What changed in the last 6 hours (hour {hour})"
    )

    w = what_changed(pid, hour)

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

elif page == "Explanation (SHAP + ESS)":

    st.title("Model Explanation")

    e = explain(pid, hour)

    top = (
        e["shap"]
        .reindex(
            e["shap"].abs()
            .sort_values(ascending=False)
            .index
        )
        .head(12)[::-1]
    )

    fig = go.Figure(
        go.Bar(
            x=top.values,
            y=[
                f"{k} = {e['features'][k]:.3g}"
                for k in top.index
            ],
            orientation="h",
            marker_color=[
                "#c1121f" if v > 0 else "#2a6f97"
                for v in top.values
            ]
        )
    )

    fig.update_layout(
        title="Top contributions (log-odds; red raises risk)",
        height=450
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

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

    f = C.TABLES / "model_comparison_test.csv"

    if f.exists():

        st.subheader("Model evaluation metrics")

        st.dataframe(
            pd.read_csv(f, index_col=0).round(4)
        )

    else:

        st.info(
            "Run `python -m src.evaluate --split test --final` first."
        )

    # Corrected image display block
    curves_file = C.FIGURES / "curves_test.png"

    if curves_file.exists():

        st.subheader("Model Performance Curves")

        st.image(
            str(curves_file)
        )

    # Alert budget analysis
    bt = pd.read_csv(
        C.TABLES / "alert_budget_table.csv"
    )

    st.subheader("Alert Budget Analysis")

    b = st.select_slider(
        "Allowed false alerts per 10 patient-days",
        bt.budget_false_alerts_per_10pd.tolist(),
        5
    )

    r = bt[
        bt.budget_false_alerts_per_10pd == b
    ].iloc[0]

    a, b2, c = st.columns(3)

    a.metric(
        "Threshold (from validation)",
        f"{r.threshold:.3f}"
    )

    b2.metric(
        "Septic patients alerted",
        f"{r['alerted_%']:.0f}%"
    )

    c.metric(
        "Median lead time",
        f"{r.median_lead_h:.0f} h"
    )

    st.subheader("Complete Alert Budget Table")

    st.dataframe(
        bt.round(3)
    )


# --------------------------------------------------
# PAGE 6: ABOUT AND LIMITATIONS
# --------------------------------------------------

else:

    st.title("About SEPSIS-AI")

    st.markdown("""
    **Data**

    PhysioNet/CinC Challenge 2019 (training sets A and B),
    hourly ICU data, vitals + labs + demographics.

    **Onset**

    First hour with `SepsisLabel = 1`.

    **ESS**

    Stability of the SHAP explanation across seeds,
    input noise and a second model (0-1).

    **RTV**

    Risk over time with first-alert hour and lead time.

    **Limitations**

    - Single retrospective dataset.
    - ICU data only.
    - Imperfect sepsis labels.
    - Possible subgroup performance gaps.
    - SHAP shows model associations, not causes.
    - ESS measures explanation stability, not correctness.
    - Not validated for clinical use.
    """)

    subgroup_files = sorted(
        C.TABLES.glob("subgroups_xgb_*.csv")
    )

    if subgroup_files:

        latest_file = subgroup_files[-1]

        st.subheader(
            "Subgroup Results (XGBoost)"
        )

        st.dataframe(
            pd.read_csv(latest_file).round(3)
        )

