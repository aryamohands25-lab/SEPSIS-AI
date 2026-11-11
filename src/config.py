"""Central configuration for SEPSIS-AI. Every other module imports from here."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = Path(os.environ.get("SEPSIS_RAW_DIR", ROOT / "data" / "raw"))
DATA_PROC = ROOT / "data" / "processed"
DATA_SPLITS = ROOT / "data" / "splits"
MODELS = ROOT / "models"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
for _p in (DATA_PROC, DATA_SPLITS, MODELS, TABLES, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

# Parquet is preferred. A pickle fallback keeps the project runnable in
# restricted/offline Python environments where pyarrow/fastparquet is absent.
def write_table(df, path, index=False):
    path = Path(path)
    try:
        df.to_parquet(path, index=index)
    except (ImportError, ModuleNotFoundError):
        df.to_pickle(path)

def read_table(path, columns=None):
    path = Path(path)
    try:
        return __import__("pandas").read_parquet(path, columns=columns)
    except Exception:
        # Also handles files previously written with the pickle fallback on a
        # machine that later gains a Parquet engine (ArrowInvalid, etc.).
        df = __import__("pandas").read_pickle(path)
        return df.loc[:, columns] if columns is not None else df

RAW_FILES = ["training_setA.csv", "training_setB.csv"]
SEED = 42

# SEPSIS_FAST=1 -> small/quick settings for smoke tests on a weak machine.
FAST = os.environ.get("SEPSIS_FAST") == "1"

# ---------------------------------------------------------------- columns
ID, TIME, LABEL = "Patient_ID", "ICULOS", "SepsisLabel"
VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
LABS = ["BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
        "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
        "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
        "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
        "Fibrinogen", "Platelets"]
DEMO = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]

# ---------------------------------------------------------------- cleaning
# (low, high) inclusive-valid ranges; anything outside becomes NaN (never deleted)
PLAUSIBLE = {
    "HR": (20, 300), "O2Sat": (20, 100), "Temp": (25, 45), "SBP": (30, 300),
    "MAP": (20, 250), "DBP": (10, 200), "Resp": (3, 80), "EtCO2": (5, 100),
    "pH": (6.5, 8.0), "Lactate": (0.1, 40), "Glucose": (10, 1500),
}
# Raw columns missing in more than this share of *training* rows lose their
# raw value column but keep their "measured" indicator.  The plan suggests
# ~90 %, but with this dataset that would delete almost every lab (lactate is
# 97 % missing) so we use 99 %.
RAW_MISSING_DROP = 0.99

# ---------------------------------------------------------------- features
WINDOW = 6                                  # hours of history for rolling stats
ROLL_VARS = ["HR", "Resp", "Temp", "MAP", "SBP", "O2Sat"]
KEY_LABS = ["Lactate", "Creatinine", "WBC", "Platelets", "Bilirubin_total",
            "BUN", "Glucose", "pH"]         # get 'hours since last measured'
HS_CAP = 48                                 # cap / value for "never measured"
MEASURED_WINDOW = 24                        # 'measured_24h' indicator

# ---------------------------------------------------------------- splits
SPLIT_FRACS = (0.70, 0.15, 0.15)

# ---------------------------------------------------------------- training
LR_C_GRID = [0.01, 0.1, 1, 10]
RF_GRID = ([dict(max_depth=8, min_samples_leaf=20), dict(max_depth=12, min_samples_leaf=20)]
           if not FAST else [dict(max_depth=10, min_samples_leaf=20)])
RF_TREES = 300 if not FAST else 20
RF_MAX_SAMPLES = 0.25                       # each tree sees 25 % of rows -> much faster
XGB_TRIALS = 30 if not FAST else 1
XGB_MAX_ROUNDS = 1000 if not FAST else 100
XGB_EARLY_STOP = 50 if not FAST else 15
FAST_TRAIN_PATIENT_FRAC = 0.02              # FAST mode: use 30 % of train patients
N_JOBS = -1

# ---------------------------------------------------------------- operating point
RECALL_TARGET = 0.80                        # row-level recall target on validation
ALERT_CONSECUTIVE = 1                       # hours above threshold before alerting
ALERT_BUDGETS = [0.5, 1, 2, 3, 5, 8, 10]   # false alerts / 10 patient-days

# ---------------------------------------------------------------- ESS / SHAP
SHAP_SAMPLE = 3000 if not FAST else 100
ESS_SAMPLE = 1000 if not FAST else 100
ESS_SEEDS = [1, 2, 3, 4, 5] if not FAST else [1]
ESS_NOISE_REPS = 20 if not FAST else 1
ESS_NOISE_FRAC = 0.02
ESS_RF_TREES = 100                          # RF used only for the cross-model check
ESS_HIGH, ESS_MED = 0.75, 0.50
