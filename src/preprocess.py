"""Stage 1: raw CSV -> cleaned data -> patient-level splits -> features -> parquet.

    python -m src.preprocess
Outputs: data/splits/*.txt, data/processed/{train,val,test}.parquet,
         models/preprocess.joblib, models/features.json, results/tables/*
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from . import config as C
from .features import clean, build_features, fit_preprocess, apply_preprocess


def load_raw() -> pd.DataFrame:
    frames = []
    for f in C.RAW_FILES:
        path = C.DATA_RAW / f
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - copy the CSVs into data/raw/ "
                                    f"or set SEPSIS_RAW_DIR")
        # Compact numeric storage is important for the 1.5M-row training files.
        d = pd.read_csv(path, low_memory=False)
        for col in d.select_dtypes(include=["float64", "int64"]).columns:
            d[col] = pd.to_numeric(d[col], downcast="float" if d[col].dtype.kind == "f" else "integer")
        d["Patient_ID"] = d["Patient_ID"].astype("category")
        d["source_set"] = f[-5]                       # 'A' or 'B'
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def make_splits(df: pd.DataFrame) -> dict:
    """70/15/15 by PATIENT, stratified by 'ever septic'. Seed 42."""
    per = df.groupby(C.ID)[C.LABEL].max()
    ids, y = per.index.to_numpy(), per.to_numpy()
    tr, rest, _, y_rest = train_test_split(ids, y, train_size=C.SPLIT_FRACS[0],
                                           stratify=y, random_state=C.SEED)
    va, te = train_test_split(rest, train_size=0.5, stratify=y_rest, random_state=C.SEED)
    splits = {"train": sorted(tr), "val": sorted(va), "test": sorted(te)}
    for k, v in splits.items():
        (C.DATA_SPLITS / f"{k}_ids.txt").write_text("\n".join(v))
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
    return splits


def _features_by_chunks(df, chunk_patients=500):
    ids = df[C.ID].unique()
    metas, Xs = [], []
    for i in range(0, len(ids), chunk_patients):
        part = df[df[C.ID].isin(ids[i:i + chunk_patients])]
        m, X = build_features(part)
        metas.append(m), Xs.append(X)
    return pd.concat(metas, ignore_index=True), pd.concat(Xs, ignore_index=True)


def dataset_summary(df, log):
    per = df.groupby(C.ID)[C.LABEL].max()
    s = {"patients": int(df[C.ID].nunique()), "rows": int(len(df)),
         "pct_septic_patients": round(float(per.mean() * 100), 2),
         "pct_positive_rows": round(float(df[C.LABEL].mean() * 100), 2),
         "mean_stay_hours": round(float(df.groupby(C.ID).size().mean()), 1)}
    s["cleaning_log"] = log
    (C.TABLES / "dataset_summary.json").write_text(json.dumps(s, indent=2))
    (df[C.VITALS + C.LABS + ["Unit1", "Unit2"]].isna().mean() * 100).round(2) \
        .rename("pct_missing").to_csv(C.TABLES / "missing_per_column.csv")
    return s


def main():
    print("loading raw ...", flush=True)
    df = load_raw()
    df, log = clean(df)
    print("summary:", json.dumps(dataset_summary(df, log), indent=1)[:600], "...", flush=True)
    splits = make_splits(df)

    # Process one split at a time; retaining all three feature matrices at once
    # can exhaust memory on the full PhysioNet training files.
    pipe = None
    for name in ("train", "val", "test"):
        ids = set(splits[name])
        part = df[df[C.ID].isin(ids)].copy()
        meta, X = _features_by_chunks(part)
        if name == "train":
            pipe = fit_preprocess(X, part)
            joblib.dump(pipe, C.MODELS / "preprocess.joblib")
            (C.MODELS / "features.json").write_text(json.dumps(pipe["features"], indent=1))
            print(f"kept {len(pipe['features'])} features; dropped raw columns: {pipe['drop_cols']}", flush=True)
        Xp = apply_preprocess(X, pipe)
        out = pd.concat([meta.rename(columns={C.ID: "patient_id", C.TIME: "hour", C.LABEL: "label"}), Xp], axis=1)
        C.write_table(out, C.DATA_PROC / f"{name}.parquet", index=False)
        print(f"{name}: {meta[C.ID].nunique()} patients, {len(meta)} rows, {Xp.shape[1]} features", flush=True)
        del part, meta, X, Xp, out
    print("done.", flush=True)


if __name__ == "__main__":
    main()
