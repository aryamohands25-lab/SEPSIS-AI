import numpy as np
import pandas as pd
from . import config as C

THRESHOLD = 0.35

def score_frame(df):
    def col(name, default):
        return pd.to_numeric(df.get(name, pd.Series(default, index=df.index)), errors="coerce").fillna(default)

    components = pd.DataFrame(index=df.index)
    components["Heart rate elevation"] = ((col("HR", 80)-90)/50).clip(0,1)
    components["Oxygen saturation reduction"] = ((95-col("O2Sat", 97))/15).clip(0,1)
    components["Respiratory rate elevation"] = ((col("Resp", 18)-20)/15).clip(0,1)
    components["Low systolic pressure"] = ((100-col("SBP", 120))/40).clip(0,1)
    components["Temperature deviation"] = (np.maximum(col("Temp", 37)-38, 36-col("Temp", 37))/2).clip(0,1)
    components["Lactate elevation"] = ((col("Lactate", 1.1)-2)/4).clip(0,1)

    weights = pd.Series({
        "Heart rate elevation": .18,
        "Oxygen saturation reduction": .22,
        "Respiratory rate elevation": .18,
        "Low systolic pressure": .20,
        "Temperature deviation": .12,
        "Lactate elevation": .10
    })
    return (components * weights).sum(axis=1).clip(0,1), components

def explain(patient_id, hour, split="demo"):
    df = C.read_table(C.DATA_PROC / "demo.parquet")
    row = df[(df.patient_id == patient_id) & (df.hour == hour)]
    if row.empty:
        raise KeyError(f"{patient_id}, hour {hour} not found")
    risk, components = score_frame(row)
    values = components.iloc[0].sort_values(ascending=False)
    return {
        "patient_id": patient_id,
        "hour": hour,
        "risk": float(risk.iloc[0]),
        "components": values,
        "features": row.iloc[0]
    }

def what_changed(patient_id, hour, lookback=6, split="demo", top=5):
    df = C.read_table(C.DATA_PROC / "demo.parquet")
    p = df[df.patient_id == patient_id].sort_values("hour")
    current = p[p.hour == hour]
    previous = p[p.hour <= hour-lookback].tail(1)
    if current.empty:
        raise KeyError(f"{patient_id}, hour {hour} not found")
    if previous.empty:
        return {"text": ["No earlier observation available for comparison."]}
    _, now = score_frame(current)
    _, before = score_frame(previous)
    delta = (now.iloc[0]-before.iloc[0]).sort_values()
    up = delta[delta > 0].tail(top)
    down = delta[delta < 0].head(top)
    lines = [f"{k}: {before.iloc[0][k]:.2f} -> {now.iloc[0][k]:.2f} ({v:+.2f})"
             for k,v in pd.concat([up,down]).items()]
    return {"text": lines or ["No change in the illustrative components."]}
