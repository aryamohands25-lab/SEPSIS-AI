"""Shared evaluation script used by everybody.

    python -m src.evaluate --split val                 # as often as you like
    python -m src.evaluate --split test --final        # ONCE: locks the test set
"""
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve, precision_recall_curve)
from . import config as C
from .models import load_model, load_split, load_thresholds

MODELS = {"lr": "Logistic Regression", "rf": "Random Forest", "xgb": "XGBoost"}
LOCK = C.TABLES / ".test_locked"


def compute_metrics(y, score, thr=None, binary=None):
    """Full metric row. `score` ranks patients; `binary` overrides thresholding (rules)."""
    pred = binary if binary is not None else (score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred, zero_division=0),
            "AUROC": roc_auc_score(y, score), "AUPRC": average_precision_score(y, score),
            "Brier": brier_score_loss(y, np.clip(score, 0, 1)) if binary is None else np.nan,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn}


def all_scores(X):
    out = {n: load_model(n).predict_proba(X) for n in MODELS}
    return out


def evaluate(split: str):
    meta, X, y = load_split(split)
    thr = load_thresholds()
    scores = all_scores(X)
    rows = {}
    for n, label in MODELS.items():
        rows[label] = compute_metrics(y, scores[n], thr[n])
    # clinical rule benchmarks (not trained models). qSOFA lacks GCS here -> resp + SBP only
    rows["qSOFA (>=2 of 2)"] = compute_metrics(y, X["qsofa_score"].to_numpy(), binary=(X["qsofa_score"] >= 2).astype(int))
    rows["SIRS (>=2 of 4)"] = compute_metrics(y, X["sirs_score"].to_numpy(), binary=(X["sirs_score"] >= 2).astype(int))
    table = pd.DataFrame(rows).T
    table.to_csv(C.TABLES / f"model_comparison_{split}.csv")
    print(f"\n=== {split} ({len(y)} rows, {y.mean():.2%} positive) ===")
    print(table.drop(columns=["TP", "FP", "FN", "TN"]).round(4).to_string())

    curves(y, scores, split)
    subgroups(meta, X, y, scores["xgb"], thr["xgb"], split)
    return meta, X, y, scores


def curves(y, scores, split):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    for n, label in MODELS.items():
        fpr, tpr, _ = roc_curve(y, scores[n]); ax[0].plot(fpr, tpr, label=label)
        p, r, _ = precision_recall_curve(y, scores[n]); ax[1].plot(r, p, label=label)
        fp_, mp_ = calibration_curve(y, scores[n], n_bins=10, strategy="quantile")
        ax[2].plot(mp_, fp_, "o-", label=label)
    ax[0].plot([0, 1], [0, 1], "k--"); ax[0].set(title="ROC", xlabel="FPR", ylabel="TPR")
    ax[1].axhline(y.mean(), color="k", ls="--"); ax[1].set(title="Precision-Recall", xlabel="Recall", ylabel="Precision")
    m = max(ax[2].get_xlim()[1], ax[2].get_ylim()[1]); ax[2].plot([0, m], [0, m], "k--")
    ax[2].set(title="Calibration (reliability)", xlabel="Predicted risk", ylabel="Observed rate")
    for a in ax: a.legend(); a.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(C.FIGURES / f"curves_{split}.png", dpi=150); plt.close()


def subgroups(meta, X, y, score, thr, split):
    age_bins = pd.cut(X["Age"], [0, 40, 60, 75, 120], labels=["<40", "40-59", "60-74", "75+"], right=False)
    groups = {"Age": age_bins, "Gender": X["Gender"].map({0: "female(0)", 1: "male(1)"})}
    rows = []
    for gname, g in groups.items():
        for lvl in g.dropna().unique():
            m = (g == lvl).to_numpy()
            if y[m].sum() < 5:
                continue
            rows.append({"group": gname, "level": str(lvl), "rows": int(m.sum()),
                         "positive_rate": y[m].mean(), "AUROC": roc_auc_score(y[m], score[m]),
                         "AUPRC": average_precision_score(y[m], score[m]),
                         "recall": recall_score(y[m], score[m] >= thr)})
    t = pd.DataFrame(rows).sort_values(["group", "level"])
    t.to_csv(C.TABLES / f"subgroups_xgb_{split}.csv", index=False)
    print("\nSubgroups (XGBoost):\n", t.round(3).to_string(index=False))


def ablation_trends(split="val"):
    """XGBoost with vs without rolling/trend features. Run on VALIDATION only."""
    from .train import make_xgb
    mtr, Xtr, ytr = load_split("train"); mva, Xva, yva = load_split(split)
    info = json.loads((C.MODELS / "train_info.json").read_text())["xgb"]
    trend_cols = [c for c in Xtr.columns if any(s in c for s in ("_mean", "_min", "_max", "_std", "_slope", "_delta"))]
    out = {}
    for name, cols in {"with trend features": list(Xtr.columns),
                       "without trend features": [c for c in Xtr.columns if c not in trend_cols]}.items():
        m = make_xgb(info["params"], info["scale_pos_weight"], n_estimators=info["best_rounds"], early=False)
        m.fit(Xtr[cols], ytr)
        p = m.predict_proba(Xva[cols])[:, 1]
        out[name] = {"n_features": len(cols), "AUROC": roc_auc_score(yva, p), "AUPRC": average_precision_score(yva, p)}
    t = pd.DataFrame(out).T; t.to_csv(C.TABLES / "ablation_trend_features.csv")
    print("\nAblation (validation):\n", t.round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--final", action="store_true", help="required for the test split")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--ablation", action="store_true")
    a = ap.parse_args()
    if a.split == "test":
        if not a.final:
            raise SystemExit("The test set is locked. Use --final when all models are frozen.")
        if LOCK.exists() and not a.force:
            raise SystemExit("Test set already evaluated once. Re-running = peeking (use --force to override).")
        # Mark the test as consumed only after evaluation completes successfully.
        evaluate(a.split)
        LOCK.write_text("evaluated")
    else:
        evaluate(a.split)
    if a.ablation:
        ablation_trends()
