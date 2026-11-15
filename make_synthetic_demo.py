import numpy as np
import pandas as pd
from src import config as C
from src import features

rng = np.random.default_rng(20261115)
rows = []
lab_ranges = {
    "BaseExcess":(0,2),"HCO3":(24,2),"FiO2":(.21,.03),"pH":(7.4,.04),
    "PaCO2":(40,3),"SaO2":(97,1),"AST":(25,5),"BUN":(14,3),
    "Alkalinephos":(80,10),"Calcium":(9.2,.3),"Chloride":(103,2),
    "Creatinine":(.9,.15),"Bilirubin_direct":(.2,.05),"Glucose":(100,10),
    "Lactate":(1.1,.2),"Magnesium":(2,.1),"Phosphate":(3.5,.3),
    "Potassium":(4,.2),"Bilirubin_total":(.7,.1),"TroponinI":(.01,.003),
    "Hct":(40,2),"Hgb":(13,.7),"PTT":(30,2),"WBC":(7,1),
    "Fibrinogen":(300,25),"Platelets":(250,20)
}
for n in range(1,21):
    pid=f"DEMO-{n:03d}"
    septic=n<=10
    onset=13 if septic else 999
    for hour in range(1,25):
        severity=min(1,max(0,(hour-onset)/8)) if septic else 0
        row={
            C.ID:pid,C.TIME:hour,C.LABEL:int(hour>=onset),
            "Age":int(rng.integers(25,86)),"Gender":int(rng.integers(0,2)),
            "Unit1":int(rng.integers(0,2)),"Unit2":int(rng.integers(0,2)),
            "HospAdmTime":-float(rng.integers(1,120)),
            "HR":np.clip(82+18*severity+rng.normal(0,5),55,140),
            "O2Sat":np.clip(97-4*severity+rng.normal(0,1),85,100),
            "Temp":np.clip(36.8+severity+rng.normal(0,.2),35,40),
            "SBP":np.clip(122-18*severity+rng.normal(0,6),75,160),
            "MAP":np.clip(88-12*severity+rng.normal(0,4),55,115),
            "DBP":np.clip(72-10*severity+rng.normal(0,4),40,100),
            "Resp":np.clip(17+8*severity+rng.normal(0,2),10,35),
            "EtCO2":np.clip(36-4*severity+rng.normal(0,2),20,50)
        }
        for lab,(mean,sd) in lab_ranges.items():
            row[lab]=max(.01,rng.normal(mean+severity*(.5 if lab in ("Lactate","WBC","BUN","Creatinine") else 0),sd)) if hour in (1,7,13,19) else np.nan
        rows.append(row)

raw=pd.DataFrame(rows)
cleaned,log=features.clean(raw)
meta,X=features.build_features(cleaned)
# Synthetic-only imputation: medians learned from these fictional rows only.
X=X.replace([np.inf,-np.inf],np.nan)
X=X.fillna(X.median()).fillna(0).astype("float32")
meta=meta.rename(columns={C.ID:"patient_id",C.TIME:"hour",C.LABEL:"label"}).reset_index(drop=True)
demo=pd.concat([meta,X.reset_index(drop=True)],axis=1)
C.write_table(demo,C.DATA_PROC/"demo.parquet")
print("Synthetic demo:",len(demo),"rows;",demo.patient_id.nunique(),"fictional profiles")
