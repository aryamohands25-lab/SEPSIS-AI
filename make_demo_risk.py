import pandas as pd
from src import config as C
from src.synthetic_demo import score_frame

demo = C.read_table(C.DATA_PROC / "demo.parquet")
risk, _ = score_frame(demo)
out = demo[["patient_id","hour","label","qsofa_score","sirs_score"]].copy()
out["risk"] = risk.astype("float32")
C.write_table(out, C.TABLES / "risk_series_demo.parquet")
print("Synthetic illustrative index generated:", len(out), "rows")
print(out.groupby("label").risk.agg(["count","mean","min","max"]))
