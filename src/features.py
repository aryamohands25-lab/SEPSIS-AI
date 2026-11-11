"""Cleaning + feature engineering.  EVERY feature uses past data only
(current hour and earlier hours of the same patient), never future hours."""
import warnings
import numpy as np
import pandas as pd
from . import config as C


# --------------------------------------------------------------- cleaning
def clean(df: pd.DataFrame):
    """Set impossible values to NaN, drop duplicates / 1-hour patients. Returns (df, log)."""
    log = {"rows_in": int(len(df)), "patients_in": int(df[C.ID].nunique())}
    n = len(df)
    df = df.drop_duplicates([C.ID, C.TIME])
    log["duplicate_rows_removed"] = int(n - len(df))
    log["values_set_to_nan"] = {}
    for col, (lo, hi) in C.PLAUSIBLE.items():
        bad = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
        log["values_set_to_nan"][col] = int(bad.sum())
        df.loc[bad, col] = np.nan
    sizes = df.groupby(C.ID)[C.TIME].transform("size")
    log["one_hour_patients_removed"] = int((df[sizes < 2][C.ID].nunique()))
    df = df[sizes >= 2]
    df = df.sort_values([C.ID, C.TIME]).reset_index(drop=True)
    log["rows_out"], log["patients_out"] = int(len(df)), int(df[C.ID].nunique())
    return df, log


# --------------------------------------------------------------- helpers
def lag_stack(x: np.ndarray, codes: np.ndarray, w: int) -> np.ndarray:
    """(w, n) array; row k holds x shifted k rows back WITHIN the same patient
    (NaN across patient boundaries).  Row 0 is the current hour."""
    out = np.full((w, len(x)), np.nan, dtype=np.float32)
    out[0] = x
    for k in range(1, w):
        same = codes[k:] == codes[:-k]
        out[k, k:] = np.where(same, x[:-k], np.nan)
    return out


def _slope(stack: np.ndarray) -> np.ndarray:
    """Least-squares slope per hour over the window (NaN-safe, >=3 points)."""
    w = stack.shape[0]
    t = -np.arange(w, dtype=np.float32)[:, None]          # 0, -1, -2 ... (hours ago)
    valid = ~np.isnan(stack)
    n = valid.sum(0).astype(np.float32)
    tv = np.where(valid, t, 0)
    x0 = np.where(valid, stack, 0)
    st, sx = tv.sum(0), x0.sum(0)
    stx, stt = (tv * x0).sum(0), (tv * tv).sum(0)
    den = n * stt - st ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (n * stx - st * sx) / den
    s[(n < 3) | (den == 0)] = np.nan
    return s


# --------------------------------------------------------------- features
def build_features(df: pd.DataFrame):
    """df must be cleaned and sorted by (Patient_ID, ICULOS).
    Returns (meta, X) with X un-imputed float32 (NaN where unknown)."""
    df = df.sort_values([C.ID, C.TIME]).reset_index(drop=True)
    codes = pd.factorize(df[C.ID])[0]
    vl = C.VITALS + C.LABS
    obs = df[vl].notna()
    ff = df.groupby(C.ID)[vl].ffill()                       # carry last value forward, never across patients
    cols = {}

    # 1) current values (forward-filled) + demographics / time
    for c in vl:
        cols[c] = ff[c].to_numpy(np.float32)
    for c in ["Age", "Gender", "HospAdmTime", C.TIME]:
        cols[c] = df[c].to_numpy(np.float32)
    for c in ["Unit1", "Unit2"]:                            # missing unit -> -1 ("unknown")
        cols[c] = df[c].fillna(-1).to_numpy(np.float32)

    # 2) rolling statistics + 3) trends over the last WINDOW hours (incl. now)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for v in C.ROLL_VARS:
            x = cols[v]
            st = lag_stack(x, codes, C.WINDOW)
            cols[f"{v}_mean{C.WINDOW}h"] = np.nanmean(st, 0)
            cols[f"{v}_min{C.WINDOW}h"] = np.nanmin(st, 0)
            cols[f"{v}_max{C.WINDOW}h"] = np.nanmax(st, 0)
            cols[f"{v}_std{C.WINDOW}h"] = np.nanstd(st, 0)
            cols[f"{v}_slope{C.WINDOW}h"] = _slope(st)
            st7 = lag_stack(x, codes, C.WINDOW + 1)
            cols[f"{v}_delta{C.WINDOW}h"] = st7[0] - st7[C.WINDOW]

    # 4) clinical flags (qSOFA-style / SIRS-style; GCS is not in the data)
    resp, sbp, hr, temp, wbc = (ff[c] for c in ["Resp", "SBP", "HR", "Temp", "WBC"])
    cols["qsofa_resp"] = (resp >= 22).to_numpy(np.float32)
    cols["qsofa_sbp"] = (sbp <= 100).to_numpy(np.float32)
    cols["qsofa_score"] = cols["qsofa_resp"] + cols["qsofa_sbp"]
    cols["sirs_hr"] = (hr > 90).to_numpy(np.float32)
    cols["sirs_temp"] = ((temp > 38) | (temp < 36)).to_numpy(np.float32)
    cols["sirs_resp"] = (resp > 20).to_numpy(np.float32)
    cols["sirs_wbc"] = ((wbc > 12) | (wbc < 4)).to_numpy(np.float32)
    cols["sirs_score"] = cols["sirs_hr"] + cols["sirs_temp"] + cols["sirs_resp"] + cols["sirs_wbc"]

    # 5) missingness: hours since last measured (+ 'measured in last 24h' for labs)
    for c in C.LABS:
        lo = df[C.TIME].where(obs[c]).groupby(df[C.ID]).ffill()   # hour of last measurement
        hs = (df[C.TIME] - lo).fillna(C.HS_CAP).clip(upper=C.HS_CAP).to_numpy(np.float32)
        if c in C.KEY_LABS:
            cols[f"{c}_hours_since"] = hs
        cols[f"{c}_measured24h"] = (hs <= C.MEASURED_WINDOW).astype(np.float32)

    X = pd.DataFrame(cols)
    meta = df[[C.ID, C.TIME, C.LABEL]].copy()
    return meta, X


# --------------------------------------------------------------- saved pipeline
def fit_preprocess(X_train: pd.DataFrame, raw_train: pd.DataFrame) -> dict:
    """Learn everything from TRAIN only: columns to drop + medians."""
    miss = raw_train[C.VITALS + C.LABS].isna().mean()
    drop = [c for c in miss.index if miss[c] > C.RAW_MISSING_DROP]
    X = X_train.drop(columns=drop)
    return {"drop_cols": drop, "features": list(X.columns),
            "medians": X.median(numeric_only=True).astype("float32").to_dict()}


def apply_preprocess(X: pd.DataFrame, pipe: dict) -> pd.DataFrame:
    X = X.drop(columns=pipe["drop_cols"])[pipe["features"]]
    return X.fillna(pipe["medians"]).astype(np.float32)


def transform_raw(df_raw: pd.DataFrame, pipe: dict):
    """Raw patient rows -> (meta, model-ready X). Used by the dashboard for new data."""
    df, _ = clean(df_raw.copy())
    meta, X = build_features(df)
    return meta, apply_preprocess(X, pipe)
