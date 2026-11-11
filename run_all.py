"""Run the whole pipeline.   python run_all.py            (SEPSIS_FAST=1 for a quick smoke test)"""
import subprocess, sys

STEPS = [["-m", "src.preprocess"], ["-m", "src.train"], ["-m", "src.evaluate", "--split", "val", "--ablation"],
         ["-m", "src.explain"], ["-m", "src.ess"], ["-m", "src.rtv"],
         ["-m", "pytest", "-q", "tests"],
         ["-m", "src.evaluate", "--split", "test", "--final"]]      # last: test set touched ONCE

for s in STEPS:
    print("\n>>>", " ".join(s), flush=True)
    if subprocess.call([sys.executable, *s]) != 0:
        sys.exit(f"step failed: {' '.join(s)}")
print("\nAll done. Start the dashboard with:  streamlit run app/app.py")
