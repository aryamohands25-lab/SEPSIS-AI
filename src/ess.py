"""Novelty 1 - Explanation Stability Score (ESS).

    python -m src.ess
For each sampled test row compare the reference SHAP vector (final XGBoost) with
variants: 5 retrained seeds, N noise-perturbed copies of the input, and a Random
Forest.  ESS = 0.5 * mean clipped Spearman(|SHAP|) + 0.5 * mean top-10 Jaccard.
"""
import json
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from . import config as C
from .explain import _artifacts, _frame
from .models import load_thresholds
from .train import make_xgb
from .models import load_split

TOPK = 10


def _spearman_rows(a, b):
    """Row-wise Spearman between |a| and |b| (ties -> average ranks)."""
    ra, rb = rankdata(np.abs(a), axis=1), rankdata(np.abs(b), axis=1)
    ra, rb = ra - ra.mean(1, keepdims=True), rb - rb.mean(1, keepdims=True)
    den = np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (ra * rb).sum(1) / den
    return np.nan_to_num(r)                       # constant vector -> 0


def _topk_mask(s, k=TOPK):
    idx = np.argsort(-np.abs(s), axis=1)[:, :k]
    m = np.zeros_like(s, dtype=bool)
    np.put_along_axis(m, idx, True, axis=1)
    return m


def _jaccard_rows(a, b):
    ma, mb = _topk_mask(a), _topk_mask(b)
    return (ma & mb).sum(1) / (ma | mb).sum(1)


def choose_sample(df, model, feats, n, seed=C.SEED):
    """Mixed sample: septic rows + high-risk non-septic + random non-septic."""
    rng = np.random.RandomState(seed)
    risk = model.predict_proba(df[feats])
    lab = df.label.to_numpy()
    pos = np.flatnonzero(lab == 1)
    neg = np.flatnonzero(lab == 0)
    hi = neg[risk[neg] >= np.quantile(risk[neg], .75)]
    a = rng.choice(pos, min(int(n * .3), len(pos)), replace=False)
    b = rng.choice(hi, int(n * .35), replace=False)
    c = rng.choice(np.setdiff1d(neg, b), n - len(a) - len(b), replace=False)
    return df.iloc[np.sort(np.r_[a, b, c])].reset_index(drop=True)


def main():
    model, expl, feats = _artifacts()
    thr = load_thresholds()["xgb"]
    info = json.loads((C.MODELS / "train_info.json").read_text())["xgb"]
    test = _frame("val")
    s = choose_sample(test, model, feats, C.ESS_SAMPLE)
    X = s[feats]
    ref = expl.shap_values(X)
    print(f"ESS sample: {len(s)} rows ({s.label.mean():.0%} septic)")

    # ---- variants -------------------------------------------------------
    mtr, Xtr, ytr = load_split("train")
    if C.FAST:
        keep = np.random.RandomState(C.SEED).rand(len(Xtr)) < C.FAST_TRAIN_PATIENT_FRAC
        Xtr, ytr = Xtr[keep], ytr[keep]
    variants = {"seed": [], "noise": [], "model": []}
    for sd in C.ESS_SEEDS:                                   # (3) seed variation
        m = make_xgb(info["params"], info["scale_pos_weight"], seed=sd,
                     n_estimators=info["best_rounds"], early=False).fit(Xtr, ytr)
        variants["seed"].append(shap.TreeExplainer(m).shap_values(X))
        print(f"  seed {sd} done")

    rng = np.random.RandomState(C.SEED)                      # (4) input noise
    sd_ = Xtr.sample(n=min(50000, len(Xtr)), random_state=C.SEED).std().to_numpy()
    lo, hi = Xtr.min().to_numpy(), Xtr.max().to_numpy()
    continuous = np.array([Xtr[c].nunique() > 2 for c in feats])   # skip binary flags
    for _ in range(C.ESS_NOISE_REPS):
        noise = rng.normal(0, 1, X.shape) * sd_ * C.ESS_NOISE_FRAC * continuous
        Xn = pd.DataFrame(np.clip(X.to_numpy() + noise, lo, hi), columns=feats)
        variants["noise"].append(expl.shap_values(Xn))

    rf = RandomForestClassifier(n_estimators=C.ESS_RF_TREES, max_depth=10, min_samples_leaf=20,
                                class_weight="balanced_subsample", max_samples=C.RF_MAX_SAMPLES,
                                n_jobs=C.N_JOBS, random_state=C.SEED).fit(Xtr, ytr)   # (5) cross-model
    rf_sv = shap.TreeExplainer(rf).shap_values(X, check_additivity=False)
    variants["model"].append(rf_sv[..., 1] if np.ndim(rf_sv) == 3 else rf_sv)
    print("  RF SHAP done")

    # ---- compare + combine ---------------------------------------------
    allv = [v for k in variants for v in variants[k]]
    rho = np.mean([np.clip(_spearman_rows(ref, v), 0, 1) for v in allv], axis=0)
    jac = np.mean([_jaccard_rows(ref, v) for v in allv], axis=0)
    ess = 0.5 * rho + 0.5 * jac
    parts = {k: 0.5 * np.mean([np.clip(_spearman_rows(ref, v), 0, 1) for v in vs], 0)
                + 0.5 * np.mean([_jaccard_rows(ref, v) for v in vs], 0) for k, vs in variants.items()}

    level = np.where(ess >= C.ESS_HIGH, "High", np.where(ess >= C.ESS_MED, "Medium", "Low"))
    out = pd.DataFrame({"patient_id": s.patient_id, "hour": s.hour, "label": s.label,
                        "risk": model.predict_proba(X), "ess": ess, "ess_spearman": rho,
                        "ess_jaccard": jac, "ess_seed": parts["seed"], "ess_noise": parts["noise"],
                        "ess_model": parts["model"], "level": level})
    out["pred"] = (out.risk >= thr).astype(int)
    out["error"] = (out.pred != out.label).astype(int)
    C.write_table(out, C.TABLES / "ess_scores.parquet", index=False)
    q = np.quantile(ess, [1 / 3, 2 / 3])
    print(f"\nESS mean={ess.mean():.3f}  tertile cut-offs={q.round(3)}  (fixed cut-offs 0.75/0.5 used for labels)")

    # ---- validation of ESS ---------------------------------------------
    rows = []
    for lv, d in out.groupby("level"):
        pos = d[d.label == 1]
        rows.append({"level": lv, "n": len(d), "error_rate": d.error.mean(),
                     "miss_rate_among_septic": (pos.pred == 0).mean() if len(pos) else np.nan,
                     "brier": ((d.risk - d.label) ** 2).mean()})
    val = pd.DataFrame(rows).set_index("level").reindex(["High", "Medium", "Low"]).dropna(how="all")
    val.to_csv(C.TABLES / "ess_validation.csv")
    from scipy.stats import spearmanr
    r, p = spearmanr(out.ess, out.error)
    print("\nError rate per ESS level:\n", val.round(4).to_string())
    print(f"Spearman(ESS, error) = {r:.3f} (p={p:.3g})  [negative = low ESS goes with more errors]")
    print("Breakdown (mean):", {k: round(float(v.mean()), 3) for k, v in parts.items()})

    # most unstable features: mean |rank change| vs reference, counted only where the
    # feature is in the reference top-10 (otherwise ties among ~0 SHAP values dominate)
    rref = rankdata(-np.abs(ref), axis=1)
    top = _topk_mask(ref)
    shift = np.zeros(len(feats))
    for v in allv:
        d = np.abs(rref - rankdata(-np.abs(v), axis=1))
        shift += np.where(top, d, 0).sum(0)
    shift = shift / (len(allv) * np.maximum(top.sum(0), 1))
    shift[top.sum(0) < 5] = np.nan                    # too rarely important to judge
    uns = pd.Series(shift, index=feats).dropna().sort_values(ascending=False)
    uns.to_csv(C.TABLES / "ess_unstable_features.csv", header=["mean_rank_shift"])
    print("\nMost unstable features (mean rank shift):\n", uns.head(10).round(2).to_string())

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(ess, bins=30, color="#2a6f97"); ax[0].set(title="ESS distribution", xlabel="ESS", ylabel="rows")
    ax[0].axvline(C.ESS_MED, ls="--", c="k"); ax[0].axvline(C.ESS_HIGH, ls="--", c="k")
    if len(val): val["error_rate"].plot.bar(ax=ax[1], color="#c1121f"); ax[1].set(title="Error rate by ESS level", ylabel="error rate")
    plt.tight_layout(); plt.savefig(C.FIGURES / "ess_overview.png", dpi=150); plt.close()


if __name__ == "__main__":
    main()


