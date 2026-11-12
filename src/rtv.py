"""Novelty 2 - Risk Trajectory View (RTV).

    python -m src.rtv
Hour-by-hour risk for every patient, first-alert hour, lead time vs. first positive challenge label,
comparison with qSOFA/SIRS, alert-budget table and the 'what changed in the
last 6 hours' explanation (SHAP difference).

'First positive label hour' = first hour with SepsisLabel = 1 (the Challenge label turns
positive approximately 6 h before clinical sepsis onset; this is not the actual clinical onset).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from . import config as C
from .explain import _artifacts, _frame, shap_values
from .features import lag_stack
from .models import load_split, load_thresholds


# ------------------------------------------------------------ risk series
def risk_series(split, smooth=1):
    """patient_id, hour, risk, label (+qSOFA/SIRS scores).  smooth>1 = TRAILING moving
    average (uses only past hours)."""
    model, _, feats = _artifacts()
    df = _frame(split)
    out = df[["patient_id", "hour", "label", "qsofa_score", "sirs_score"]].copy()
    out["risk"] = model.predict_proba(df[feats]).astype(np.float32)
    if smooth > 1:
        out["risk"] = out.groupby("patient_id")["risk"].transform(lambda s: s.rolling(smooth, min_periods=1).mean())
    return out


def add_alerts(rs, thr, consecutive=C.ALERT_CONSECUTIVE, col="risk", rule_thr=None):
    """alert_flag = value >= thr for `consecutive` hours in a row (current + past only)."""
    codes = pd.factorize(rs.patient_id)[0]
    above = (rs[col].to_numpy() >= thr).astype(np.float32)
    flag = lag_stack(above, codes, consecutive).min(0) if consecutive > 1 else above
    return np.nan_to_num(flag).astype(bool)


# ------------------------------------------------------------ metrics
def first_alert_hours(rs, flag):
    return rs[flag].groupby("patient_id")["hour"].min()


def first_positive_label_hours(rs):
    return rs[rs.label == 1].groupby("patient_id")["hour"].min()


def alert_metrics(rs, flag, name=""):
    first_label, t_alert = first_positive_label_hours(rs), first_alert_hours(rs, flag)
    sepsis_ids = first_label.index
    lead = (first_label - t_alert.reindex(sepsis_ids))           # NaN = never alerted
    n = len(sepsis_ids)
    # false alerts: alert EPISODES (rising edges) on non-septic patients per 10 patient-days
    prev = pd.Series(flag, index=rs.index).groupby(rs.patient_id).shift(1, fill_value=False)
    edges = flag & ~prev.to_numpy()
    ns = ~rs.patient_id.isin(sepsis_ids).to_numpy()
    patient_days = ns.sum() / 24
    m = {"rule": name, "septic_patients": n,
         "alerted_%": 100 * lead.notna().mean(),
         "alerted_>=1h_before_%": 100 * (lead >= 1).mean(),
         "alerted_>=3h_before_%": 100 * (lead >= 3).mean(),
         "alerted_>=6h_before_%": 100 * (lead >= 6).mean(),
         "median_hours_before_first_positive_label": lead.median(),
         "false_alerts_per_10_patient_days": 10 * edges[ns].sum() / patient_days}
    return m, lead


def threshold_for_budget(rs_val, budget, consecutive=C.ALERT_CONSECUTIVE):
    """Smallest threshold keeping false alerts per 10 patient-days <= budget (validation)."""
    lo, hi = 0.0, 1.0
    for _ in range(18):
        mid = (lo + hi) / 2
        fa = alert_metrics(rs_val, add_alerts(rs_val, mid, consecutive))[0]["false_alerts_per_10_patient_days"]
        lo, hi = (mid, hi) if fa > budget else (lo, mid)
    return hi


# ------------------------------------------------------------ what changed
def what_changed(patient_id, hour, lookback=6, split="val", top=5):
    """SHAP is additive, so SHAP(t) - SHAP(t-6) tells which features moved the risk."""
    df = _frame(split)
    p = df[df.patient_id == patient_id].set_index("hour")
    h0 = max(int(p.index.min()), hour - lookback)
    _, _, feats = _artifacts()
    sv = shap_values(p.loc[[h0, hour], feats])
    d = pd.DataFrame({"feature": feats, "value_then": p.loc[h0, feats].to_numpy(),
                      "value_now": p.loc[hour, feats].to_numpy(),
                      "shap_change": sv[1] - sv[0]})
    up = d[d.shap_change > 1e-6].sort_values("shap_change", ascending=False).head(top)
    down = d[d.shap_change < -1e-6].sort_values("shap_change").head(top)
    lines = [f"{r.feature} {r.value_then:.3g} -> {r.value_now:.3g}: {r.shap_change:+.2f} (log-odds)"
             for r in pd.concat([up, down]).itertuples()]
    if h0 == hour:
        lines = ["No earlier hour available to compare with."]
    return {"from_hour": h0, "to_hour": hour, "increases": up, "decreases": down, "text": lines}


# ------------------------------------------------------------ plotting
def plot_patient(rs, pid, thr, flag, path, title=""):
    p = rs[rs.patient_id == pid]
    fl = flag[(rs.patient_id == pid).to_numpy()]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(p.hour, p.risk, lw=2, label="risk")
    ax.axhline(thr, color="k", ls="--", label=f"threshold {thr:.3f}")
    if fl.any():
        ta = p.hour[fl].min(); ax.axvline(ta, color="r"); ax.axvspan(ta, p.hour.max(), color="r", alpha=.08)
        ax.text(ta, ax.get_ylim()[1] * .92, " first alert", color="r")
    if (p.label == 1).any():
        ax.axvline(p.hour[p.label == 1].min(), color="g", ls=":", label="first positive label")
    ax.set(title=f"{title} {pid}", xlabel="ICU hour", ylabel="calibrated risk"); ax.legend()
    plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()


# ------------------------------------------------------------ main
def main():
    thr = load_thresholds()["xgb"]
    val, test = risk_series("val"), risk_series("test")
    flag_t = add_alerts(test, thr)
    test["alert_flag"] = flag_t
    C.write_table(test[["patient_id", "hour", "risk", "alert_flag"]], C.TABLES / "risk_series.parquet", index=False)
    C.write_table(val[["patient_id", "hour", "risk"]], C.TABLES / "risk_series_val.parquet", index=False)

    rows, leads = [], {}
    for name, fl in {"XGBoost RTV": flag_t,
                     "qSOFA (>=2)": (test.qsofa_score >= 2).to_numpy(),
                     "SIRS (>=2)": (test.sirs_score >= 2).to_numpy()}.items():
        m, lead = alert_metrics(test, fl, name); rows.append(m); leads[name] = lead
    t = pd.DataFrame(rows).set_index("rule")
    both = {k: (leads[k] - leads["XGBoost RTV"]) for k in ("qSOFA (>=2)", "SIRS (>=2)")}
    t.to_csv(C.TABLES / "rtv_lead_time.csv")
    print("\nLead-time / alert metrics (test, threshold from validation):\n", t.round(2).T.to_string())
    for k, d in both.items():
        d = (-d).dropna()                                     # hours the model alerts EARLIER than the rule
        print(f"  vs {k}: patients alerted by both = {len(d)}, median hours gained = {d.median():.1f}")

    # alert budget: threshold chosen on VALIDATION, performance reported on TEST
    rows = []
    for b in C.ALERT_BUDGETS:
        th = threshold_for_budget(val, b)
        m, _ = alert_metrics(test, add_alerts(test, th))
        rows.append({"budget_false_alerts_per_10pd": b, "threshold": th, **{k: m[k] for k in m if k not in ("rule", "septic_patients")}})
    bt = pd.DataFrame(rows); bt.to_csv(C.TABLES / "alert_budget_table.csv", index=False)
    print("\nAlert-budget table:\n", bt.round(3).to_string(index=False))

    # four example patients (early catch, late catch, missed, false alarm)
    first_label = first_positive_label_hours(test); ta = first_alert_hours(test, flag_t)
    lead = (first_label - ta.reindex(first_label.index))
    first_hour = test.groupby("patient_id")["hour"].min()
    late_start = ta.index[(ta - first_hour.reindex(ta.index)) >= 6]      # alert after >=6 h of data
    ex = {"early_catch": lead[(lead >= 6) & lead.index.isin(late_start)].index[:1], "late_catch": lead[lead <= 0].index[:1],
          "missed_case": lead[lead.isna()].index[:1],
          "false_alarm": ta.index.difference(first_label.index)[:1]}
    for k, ids in ex.items():
        if len(ids):
            plot_patient(test, ids[0], thr, flag_t, C.FIGURES / f"rtv_{k}.png", k.replace("_", " "))
            if k == "early_catch":
                w = what_changed(ids[0], int(ta[ids[0]]))
                print(f"\n'What changed' for {ids[0]} at first alert (hour {ta[ids[0]]}):")
                print("\n".join("  " + l for l in w["text"]))


if __name__ == "__main__":
    main()
