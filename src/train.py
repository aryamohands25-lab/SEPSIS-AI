"""Stage 2: train Logistic Regression, Random Forest, XGBoost. Only TRAIN data is
used for fitting and VALIDATION for tuning / calibration / threshold.  The test
set is never touched here.

    python -m src.train            (SEPSIS_FAST=1 for a quick smoke run)
"""
import json
import time
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from . import config as C
from .models import CalibratedModel, choose_threshold, load_split


def _subsample_patients(meta, X, y, frac, seed=C.SEED):
    ids = meta["patient_id"].unique()
    keep = np.random.RandomState(seed).choice(ids, int(len(ids) * frac), replace=False)
    m = meta["patient_id"].isin(keep).to_numpy()
    return X[m], y[m]


def auprc(y, p):
    return average_precision_score(y, p)


def train_lr(Xtr, ytr, Xva, yva):
    best = None
    for c in C.LR_C_GRID:
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(C=c, class_weight="balanced", max_iter=1000))
        m.fit(Xtr, ytr)
        s = auprc(yva, m.predict_proba(Xva)[:, 1])
        print(f"  LR C={c}: val AUPRC={s:.4f}")
        if best is None or s > best[0]:
            best = (s, c, m)
    return best[2], {"C": best[1], "val_auprc": best[0]}


def train_rf(Xtr, ytr, Xva, yva):
    best = None
    for g in C.RF_GRID:
        m = RandomForestClassifier(n_estimators=C.RF_TREES, class_weight="balanced_subsample",
                                   max_samples=C.RF_MAX_SAMPLES, n_jobs=C.N_JOBS,
                                   random_state=C.SEED, **g)
        m.fit(Xtr, ytr)
        s = auprc(yva, m.predict_proba(Xva)[:, 1])
        print(f"  RF {g}: val AUPRC={s:.4f}")
        if best is None or s > best[0]:
            best = (s, g, m)
    return best[2], {**best[1], "val_auprc": best[0]}


def xgb_params_random(rng):
    return dict(learning_rate=float(rng.choice([0.05, 0.08, 0.1])),
                max_depth=int(rng.randint(4, 9)),
                min_child_weight=int(rng.choice([1, 5, 10, 20])),
                subsample=0.8, colsample_bytree=float(rng.choice([0.6, 0.8, 1.0])),
                reg_lambda=float(rng.choice([1, 5, 10])),
                gamma=float(rng.choice([0, 0.5, 1])))


def make_xgb(params, spw, seed=C.SEED, n_estimators=None, early=True):
    kw = dict(n_estimators=n_estimators or C.XGB_MAX_ROUNDS, tree_method="hist",
              scale_pos_weight=spw, eval_metric="aucpr", random_state=seed,
              n_jobs=C.N_JOBS, **params)
    if early:
        kw["early_stopping_rounds"] = C.XGB_EARLY_STOP
    return XGBClassifier(**kw)


def train_xgb(Xtr, ytr, Xva, yva):
    spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    rng, best = np.random.RandomState(C.SEED), None
    for t in range(C.XGB_TRIALS):
        params = xgb_params_random(rng)
        t0 = time.time()
        m = make_xgb(params, spw)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        s = auprc(yva, m.predict_proba(Xva)[:, 1])
        print(f"  XGB trial {t + 1}/{C.XGB_TRIALS}: val AUPRC={s:.4f} "
              f"rounds={m.best_iteration + 1} ({time.time() - t0:.0f}s) {params}")
        if best is None or s > best[0]:
            best = (s, params, m)
    info = {"params": best[1], "best_rounds": int(best[2].best_iteration + 1),
            "scale_pos_weight": spw, "val_auprc": best[0]}
    return best[2], info


def main():
    mtr, Xtr, ytr = load_split("train")
    mva, Xva, yva = load_split("val")
    train_patients = int(mtr["patient_id"].nunique())
    if C.FAST:
        Xtr, ytr = _subsample_patients(mtr, Xtr, ytr, C.FAST_TRAIN_PATIENT_FRAC)
        train_patients = int(mtr["patient_id"].nunique() * C.FAST_TRAIN_PATIENT_FRAC)
        print(f"FAST mode: training on a patient subset; {len(Xtr)} rows")
    else:
        expected_patients = len((C.DATA_SPLITS / "train_ids.txt").read_text().split())
        if train_patients != expected_patients:
            raise ValueError(
                f"Training data contains {train_patients} patients; "
                f"expected {expected_patients} from train_ids.txt."
            )
    print(f"train {Xtr.shape}, positives {ytr.mean():.3%}")

    trainers = {"lr": train_lr, "rf": train_rf, "xgb": train_xgb}
    thresholds, info = {}, {}
    for name, fn in trainers.items():
        print(f"== {name}")
        t0 = time.time()
        base, meta_info = fn(Xtr, ytr, Xva, yva)
        model = CalibratedModel(base, name).fit_calibration(Xva, yva)
        thr = choose_threshold(yva, model.predict_proba(Xva))
        thresholds[name] = thr
        info[name] = {
            **meta_info,
            "threshold": thr,
            "train_seconds": round(time.time() - t0),
            "train_patients": train_patients,
            "train_rows": len(Xtr),
            "fast_mode": C.FAST,
            "scikit_learn_version": sklearn.__version__,
        }
        joblib.dump(model, C.MODELS / f"{name}.joblib")
        print(f"  saved {name}; threshold (recall>={C.RECALL_TARGET}) = {thr:.4f}")

    (C.MODELS / "threshold.json").write_text(json.dumps(thresholds, indent=2))
    (C.MODELS / "train_info.json").write_text(json.dumps(info, indent=2, default=float))
    # LR coefficients = extra interpretable view
    lr = joblib.load(C.MODELS / "lr.joblib").base
    coefs = pd.Series(lr[-1].coef_[0], index=Xtr.columns).sort_values(key=abs, ascending=False)
    coefs.to_csv(C.TABLES / "lr_coefficients.csv", header=["coef_standardised"])
    print("top LR coefficients:\n", coefs.head(10))


if __name__ == "__main__":
    main()
