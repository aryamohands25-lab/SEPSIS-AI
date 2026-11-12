# SEPSIS-AI ? Explainable Early Sepsis Risk Prediction

An academic machine-learning prototype for hourly sepsis-risk prediction using ICU patient data. The project compares Logistic Regression, Random Forest and XGBoost, with probability calibration, SHAP explanations, explanation-stability analysis and risk-trajectory monitoring.


## 🚀 Live Demo

**[Open SEPSIS-AI Live Dashboard](https://sepsis-ai-lmjw7f3k7myvdfuycgoegr.streamlit.app/)**


> **Disclaimer:** This is a research and educational prototype. It is not clinically validated and must not be used to diagnose, treat or make decisions about patients.

## 1. Project workflow

Raw ICU data ? cleaning ? temporal feature engineering ? patient-level splitting ? model training ? validation and calibration ? SHAP explanations ? Explanation Stability Score (ESS) ? Risk Trajectory Visualization (RTV).

## 2. Dataset and splitting

- Uses the PhysioNet/CinC Challenge 2019 training datasets (Set A and Set B).
- Combined cohort: **40,336 patients**.
- Patient-level stratified split: 70% training, 15% validation and 15% test.
- Random seed: 42.
- No patient overlap between the three splits.

| Split | Patients | Processed rows |
|---|---:|---:|
| Train | 28,235 | 1,086,436 |
| Validation | 6,050 | 231,472 |
| Test | 6,051 | 234,302 |

### Data-integrity checks

- Patient IDs in processed train and validation data were verified against their saved split assignments.
- All 6,051 test patients were also verified against the test split assignment.
- Hourly continuity was checked in all three processed splits:
  - Train: 1,058,201 adjacent intervals; all exactly one hour.
  - Validation: 225,422 adjacent intervals; all exactly one hour.
  - Test: 228,251 adjacent intervals; all exactly one hour.
- No gaps, duplicate hours or reversed hours were found in these adjacent intervals.

## 3. Temporal feature engineering

Features are constructed using the current hour and earlier observations from the same patient.

- Forward filling is performed within each patient.
- Rolling means, minima, maxima and standard deviations.
- Temporal slopes and changes over the configured observation window.
- qSOFA-style and SIRS-style component scores.
- Laboratory measurement recency and measured-in-last-24-hours indicators.
- Missing values are imputed using medians learned from training data.

The preprocessing pipeline learns its dropped columns and imputation medians from training data only.

The hourly-continuity audit supports interpreting row-based lag windows as consecutive hourly windows in the processed splits.

## 4. Models and validation results

Three machine-learning models were trained and compared:

- Logistic Regression (LR)
- Random Forest (RF)
- XGBoost (XGB)

Probability calibration uses Platt scaling fitted on validation data. Model selection and threshold selection also use validation data.

### Validation performance

| Model | AUROC | AUPRC | Recall | Precision |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.7874 | 0.0792 | 0.8001 | 0.0366 |
| Random Forest | 0.8127 | 0.0868 | 0.8001 | 0.0413 |
| XGBoost | 0.8309 | 0.1008 | 0.8001 | 0.0496 |

The validation positive-label prevalence is approximately 1.8%. XGBoost achieved the highest validation AUROC and AUPRC among these three models.

The selected validation recall target was 0.80. At this operating point, precision is low, indicating a substantial false-alert burden. These results do not establish clinical utility.

## 5. Clinical baseline comparison

The project compares model predictions with:

- qSOFA-style score ? 2
- SIRS-style score ? 2

The qSOFA-style score uses respiratory rate and systolic blood pressure because GCS is unavailable in the dataset. It is therefore not a complete qSOFA implementation.

Validation results:

| Method | AUROC | AUPRC |
|---|---:|---:|
| XGBoost | 0.8309 | 0.1008 |
| qSOFA-style ? 2 | 0.5736 | 0.0221 |
| SIRS-style ? 2 | 0.6279 | 0.0289 |

## 6. Explainability ? SHAP

SHAP is used to explain the raw XGBoost model output.

The project provides:
- Global feature-importance rankings.
- SHAP summary and beeswarm plots.
- Patient-level waterfall explanations.
- A six-hour feature-change explanation.

**Important:** SHAP values explain the raw model margin (log-odds), not the final Platt-calibrated probability directly.

## 7. Explanation Stability Score (ESS)

ESS evaluates how consistently explanations behave under repeated perturbations of engineered features.

Recorded validation analysis:
- Mean ESS: approximately 0.8275.
- Standard deviation: approximately 0.0455.
- Median ESS: approximately 0.8287.

ESS measures explanation stability, not prediction correctness or clinical reliability. The perturbation approach can also create slightly inconsistent combinations of engineered features.

## 8. Risk Trajectory Visualization (RTV)

The RTV component is designed to provide:

- Hour-by-hour predicted risk for each patient.
- First alert hour.
- Lead time relative to the first positive challenge label.
- Comparison with qSOFA-style and SIRS-style rules.
- Alert-budget threshold analysis.
- Patient-level risk trajectory plots.
- A six-hour SHAP difference explanation of changing features.

### Important label-timing clarification

In the PhysioNet/CinC Challenge 2019 dataset, `SepsisLabel` becomes positive approximately six hours before the challenge-defined clinical sepsis time.

Therefore, this project defines its reference time as:

**First positive label hour = first ICU hour with `SepsisLabel = 1`.**

Lead-time results measured against this reference are not measurements of time before the actual clinical onset. The label is an early-warning target and must not be described as the actual onset timestamp.

### Alert thresholds

The standard model threshold is selected using validation data to target recall of at least 0.80. The alert-budget analysis explores alternative thresholds to study the trade-off between detection and false alerts.

False-alert rates and lead-time summaries should be interpreted with the dataset's low positive-label prevalence in mind.

## 9. Test-set interpretation and limitations

The test split was previously inspected during development. It should therefore **not be represented as a pristine, untouched independent holdout**. Existing test outputs are exploratory results for the project version in which they were generated.

For a stronger final scientific evaluation, use a new independent cohort or a properly reserved external/temporal holdout.

Other limitations:
- Retrospective challenge-dataset evaluation is not prospective clinical validation.
- Current-hour measurements are assumed to be available at prediction time; real-world measurement and documentation delays need separate assessment.
- Forward filling assumes the last recorded measurement remains usable until another measurement is available.
- qSOFA-style scoring is incomplete because GCS is unavailable.
- Low precision can result in many false alerts.
- SHAP and ESS do not establish causality, clinical correctness or patient benefit.
- Dataset label timing is not equivalent to actual clinical onset timing.

## 10. Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Place `training_setA.csv` and `training_setB.csv` in `data/raw/`, or configure `SEPSIS_RAW_DIR` to point to the directory containing them.

## 11. Running the project

Open the project directory and activate the virtual environment if one is being used.

Run the dashboard after the required model and preprocessing artifacts are available:

```bash
streamlit run app/app.py
```

Useful development commands:

```bash
python -m pytest -q tests
python -m src.evaluate --split val --ablation
python -m src.explain
python -m src.ess
```

**Caution:** Do not run `python run_all.py` or `python -m src.rtv` casually on the current project state. The full pipeline can retrain models, and the current RTV main workflow reads and reports test-split results. Preserve the existing artifacts unless intentionally starting a new experiment with a clearly documented evaluation protocol.

## 12. Project structure

```text
sepsis-ai/
??? data/
?   ??? raw/
?   ??? processed/
?   ??? splits/
??? models/
??? results/
?   ??? figures/
?   ??? tables/
??? src/
?   ??? config.py
?   ??? features.py
?   ??? preprocess.py
?   ??? models.py
?   ??? train.py
?   ??? evaluate.py
?   ??? explain.py
?   ??? ess.py
?   ??? rtv.py
??? app/
??? tests/
??? requirements.txt
??? README.md
```

## 13. Intended contribution

This project brings together patient-level data splitting, past-only temporal feature engineering, machine-learning model comparison, probability calibration, SHAP-based explainability, explanation stability analysis and patient-level risk trajectories in one reproducible academic prototype.
