"""Stage 3: SHAP for the final XGBoost model.

    python -m src.explain          # global + local plots on the test sample
Reusable:  explain(patient_id, hour)  -> dict with SHAP values, risk, feature values
"""
import functools
import json
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from . import config as C
from .models import load_model, load_thresholds


@functools.lru_cache(maxsize=1)
def _artifacts():
    model = load_model("xgb")
    feats = json.loads((C.MODELS / "features.json").read_text())
    return model, shap.TreeExplainer(model.base), feats


@functools.lru_cache(maxsize=2)
def _frame(split="test"):
    if split == "demo":
        return pd.read_pickle(C.ROOT / "demo_data" / "test.pkl")
    return C.read_table(C.DATA_PROC / f"{split}.parquet")


def shap_values(X: pd.DataFrame) -> np.ndarray:
    """SHAP on the raw XGBoost margin (log-odds). Values add up to margin - base."""
    _, expl, feats = _artifacts()
    return expl.shap_values(X[feats])


def explain(patient_id: str, hour: int, split: str = "test") -> dict:
    model, expl, feats = _artifacts()
    df = _frame(split)
    row = df[(df.patient_id == patient_id) & (df.hour == hour)]
    if row.empty:
        raise KeyError(f"{patient_id} hour {hour} not in {split}")
    X = row[feats]
    sv = expl.shap_values(X)[0]
    return {"patient_id": patient_id, "hour": hour,
            "risk": float(model.predict_proba(X)[0]),          # calibrated probability
            "base_value": float(np.ravel(expl.expected_value)[0]),
            "shap": pd.Series(sv, index=feats), "features": X.iloc[0],
            "label": int(row["label"].iloc[0])}


def sample_rows(df, n, seed=C.SEED):
    """All-purpose sample: ~30 % septic rows, the rest non-septic."""
    rng = np.random.RandomState(seed)
    pos = np.flatnonzero(df.label.to_numpy() == 1)
    neg = np.flatnonzero(df.label.to_numpy() == 0)
    idx = np.r_[rng.choice(pos, min(int(n * .3), len(pos)), replace=False),
                rng.choice(neg, n - min(int(n * .3), len(pos)), replace=False)]
    return df.iloc[np.sort(idx)].reset_index(drop=True)


def _waterfall(e, title, path):
    ex = shap.Explanation(values=e["shap"].to_numpy(), base_values=e["base_value"],
                          data=e["features"].to_numpy(), feature_names=list(e["shap"].index))
    shap.plots.waterfall(ex, max_display=12, show=False)
    plt.title(f"{title}\n{e['patient_id']} hour {e['hour']}  risk={e['risk']:.1%}  label={e['label']}", fontsize=9)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


def main():
    model, expl, feats = _artifacts()
    thr = load_thresholds()["xgb"]
    df = _frame("test")
    s = sample_rows(df, C.SHAP_SAMPLE)
    X = s[feats]
    sv = expl.shap_values(X)
    np.savez_compressed(C.TABLES / "shap_sample.npz", shap=sv, patient_id=s.patient_id.values, hour=s.hour.values)

    shap.summary_plot(sv, X, max_display=20, show=False)
    plt.tight_layout(); plt.savefig(C.FIGURES / "shap_beeswarm.png", dpi=150); plt.close()
    shap.summary_plot(sv, X, plot_type="bar", max_display=20, show=False)
    plt.tight_layout(); plt.savefig(C.FIGURES / "shap_bar.png", dpi=150); plt.close()
    imp = pd.Series(np.abs(sv).mean(0), index=feats).sort_values(ascending=False)
    imp.to_csv(C.TABLES / "shap_importance.csv", header=["mean_abs_shap"])
    print("Top-15 features by mean |SHAP| (sanity-check these clinically; patient-level"
          " identifiers or ICULOS/HospAdmTime dominating => investigate leakage):\n", imp.head(15))

    # local examples: highest-risk septic row, low-risk non-septic row, false negative
    risk = model.predict_proba(X)
    lab = s.label.to_numpy()
    picks = {"high_risk_true_positive": np.flatnonzero((lab == 1) & (risk >= thr)),
             "low_risk_true_negative": np.flatnonzero((lab == 0) & (risk < thr)),
             "false_negative": np.flatnonzero((lab == 1) & (risk < thr))}
    for name, idx in picks.items():
        if len(idx) == 0:
            continue
        i = idx[np.argmax(risk[idx])] if name.startswith("high") else idx[np.argmin(risk[idx])] if name.startswith("low") else idx[np.argmax(risk[idx])]
        _waterfall(explain(s.patient_id[i], int(s.hour[i])), name.replace("_", " "), C.FIGURES / f"shap_waterfall_{name}.png")
    print("saved SHAP figures to", C.FIGURES)


if __name__ == "__main__":
    main()
