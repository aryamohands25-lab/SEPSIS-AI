"""Shared model helpers: data loading, Platt-calibrated wrapper, threshold choice."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from . import config as C


def load_split(name: str):
    """-> meta (patient_id, hour, label), X (features), y."""
    df = C.read_table(C.DATA_PROC / f"{name}.parquet")
    feats = json.loads((C.MODELS / "features.json").read_text())
    return df[["patient_id", "hour", "label"]], df[feats], df["label"].to_numpy()


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


class CalibratedModel:
    """Base model + Platt scaling fitted on VALIDATION data.
    Platt is strictly monotone, so AUROC/AUPRC are unchanged; it only makes
    '30 % risk' mean about 30 % (important because class weights distort raw scores)."""

    def __init__(self, base, name):
        self.base, self.name, self.platt = base, name, None

    def raw_proba(self, X):
        return self.base.predict_proba(X)[:, 1]

    def fit_calibration(self, X_val, y_val):
        self.platt = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(self.raw_proba(X_val)), y_val)
        return self

    def predict_proba(self, X):
        p = self.raw_proba(X)
        return self.platt.predict_proba(_logit(p))[:, 1] if self.platt is not None else p


def choose_threshold(y, p, recall_target=C.RECALL_TARGET) -> float:
    """Highest threshold whose recall on VALIDATION data is >= target (PR curve)."""
    prec, rec, thr = precision_recall_curve(y, p)
    ok = np.where(rec[:-1] >= recall_target)[0]
    return float(thr[ok[-1]]) if len(ok) else float(thr[0])


def load_model(name: str) -> CalibratedModel:
    return joblib.load(C.MODELS / f"{name}.joblib")


def load_thresholds() -> dict:
    return json.loads((C.MODELS / "threshold.json").read_text())
