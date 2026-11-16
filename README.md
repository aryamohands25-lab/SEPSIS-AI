# SEPSIS-AI Explainable Early Sepsis Risk Prediction

An academic machine-learning prototype for hourly sepsis-risk prediction using ICU patient data. The project compares Logistic Regression, Random Forest and XGBoost, with probability calibration, SHAP explanations, explanation-stability analysis and risk-trajectory monitoring.

LIVE LINK : https://sepsis-ai-hntk2rf8dectj5cbmxttxi.streamlit.app/

> **Disclaimer:** This is a research and educational prototype. It is not clinically validated and must not be used to diagnose, treat or make decisions about patients.

# SEPSIS-AI: An Explainable and Risk-Aware Early Sepsis Prediction System

## 1. Project Overview

SEPSIS-AI is a machine learning-based clinical decision support system designed to support the early identification of sepsis in Intensive Care Unit (ICU) patients.

Sepsis is a life-threatening condition that occurs when the body's response to an infection causes damage to its own tissues and organs. Early identification is essential for timely clinical intervention and improved patient outcomes.

This project uses clinical measurements and patient data collected over time to develop machine learning models that predict sepsis risk. The system aims to combine predictive modelling with explainability and risk monitoring to provide more interpretable and informative predictions.

The project integrates machine learning, explainable artificial intelligence (XAI), risk assessment, and an interactive dashboard.

**Project Domain:** Healthcare AI, Machine Learning and Explainable AI

## 2. Problem Statement

Early identification of sepsis in ICU patients is challenging because clinical indicators can change over time, and patient observations may be incomplete or inconsistent.

Traditional approaches may not always capture complex relationships between clinical variables or changes in patient conditions.

The objective of SEPSIS-AI is to develop a machine learning-based system that analyses routinely collected ICU data to identify potential sepsis risk at an early stage.

The system also aims to make predictions more understandable by incorporating explainability techniques and monitoring changes in patient risk over time.

## 3. Project Objectives

- Develop machine learning models for early sepsis prediction.
- Collect, clean, and preprocess ICU patient data.
- Perform exploratory data analysis and feature engineering.
- Implement and compare multiple machine learning algorithms.
- Evaluate model performance using appropriate classification metrics.
- Incorporate explainable AI techniques to understand model predictions.
- Assess the stability of model explanations.
- Monitor changes in patient risk over time.
- Develop an interactive dashboard for presenting predictions and analysis.

## 4. Dataset Description

The project uses the **PhysioNet/Computing in Cardiology Challenge 2019 dataset**.

Dataset source: [https://physionet.org/content/challenge-2019/1.0.0/](https://physionet.org/content/challenge-2019/1.0.0/)

The dataset contains hourly clinical observations collected from ICU patients and includes physiological measurements, laboratory results, demographic information, and sepsis labels.

Some of the important variables include:

- Heart rate
- Blood pressure
- Respiratory rate
- Body temperature
- Oxygen saturation
- Laboratory measurements
- Patient demographics
- ICU length of stay
- Sepsis labels

The time-series structure of the dataset allows the analysis of changes in clinical measurements over time.

## 5. Overall Project Methodology

The project follows a structured machine learning workflow:

1. Dataset acquisition and loading.
2. Data cleaning and preprocessing.
3. Missing-value handling.
4. Exploratory Data Analysis (EDA).
5. Feature engineering.
6. Patient-level data splitting.
7. Machine learning model development.
8. Model evaluation and comparison.
9. Explainability using SHAP.
10. Explainability Stability Score (ESS).
11. Risk Trajectory Visualization (RTV).
12. Dashboard integration and testing.

# 6 – Data Preparation and Machine Learning Models

Responsible for dataset preparation, preprocessing, exploratory analysis, feature engineering, and the development of two initial machine learning models.

#### 6.1 Data Acquisition and Preprocessing

The PhysioNet 2019 dataset was acquired and processed for machine learning implementation.

The following activities were performed:

- Loaded and integrated the available training datasets.
- Cleaned and processed the clinical observations.
- Handled missing values using appropriate imputation techniques.
- Performed Exploratory Data Analysis (EDA).
- Analysed the distribution of sepsis and non-sepsis observations.
- Performed feature engineering.
- Prepared the dataset for model training.
- Created patient-level training, validation, and test splits.
- Prepared feature columns and supporting preprocessing objects.
- Prepared the data dictionary and data-loading components for application integration.

#### 6.2 Training Data Preparation

A training sample containing 300,000 observations and 77 features was prepared for model development.

The class distribution was:

| Sepsis Label   | Number of Observations |
| -------------- | ---------------------: |
| Non-Sepsis (0) |                294,599 |
| Sepsis (1)     |                  5,401 |
| **Total**      |            **300,000** |

The dataset exhibited significant class imbalance, with sepsis observations representing a relatively small proportion of the training sample.

This imbalance was considered during model development and evaluation.

#### 6.3 Logistic Regression Model

Logistic Regression was implemented as a baseline classification model for predicting sepsis risk.

The implementation involved:

- Model training using the prepared clinical features.
- Model configuration and tuning.
- Validation using the validation dataset.
- Evaluation using precision, recall, F1-score, ROC-AUC, and PR-AUC.
- Generation of a confusion matrix.
- Saving the trained model for future application integration.

**Logistic Regression Evaluation Results**

| Metric    | Non-Sepsis (0) | Sepsis (1) |
| --------- | -------------: | ---------: |
| Precision |           0.99 |       0.05 |
| Recall    |           0.76 |       0.67 |
| F1-score  |           0.86 |       0.09 |
| Support   |        227,310 |      4,162 |

Additional metrics:

- Accuracy: 0.76
- ROC-AUC: 0.7813
- PR-AUC: 0.0726

**Confusion Matrix**

|                   | Predicted Non-Sepsis | Predicted Sepsis |
| ----------------- | -------------------: | ---------------: |
| Actual Non-Sepsis |              171,979 |           55,331 |
| Actual Sepsis     |                1,368 |            2,794 |

The Logistic Regression model achieved a sepsis recall of 0.67, indicating that it identified a considerable proportion of actual sepsis observations. However, its precision was low, indicating a high number of false-positive predictions.

#### 6.4 Random Forest Model

Random Forest was implemented as an ensemble-based classification model to capture more complex relationships between clinical features.

The implementation involved:

- Model development and hyperparameter tuning.
- Training using the prepared dataset.
- Validation and performance evaluation.
- Generation of feature importance results.
- Saving the trained model.
- Preparing model outputs for explainability and application integration.

**Random Forest Evaluation Results**

| Metric    | Non-Sepsis (0) | Sepsis (1) |
| --------- | -------------: | ---------: |
| Precision |           0.99 |       0.10 |
| Recall    |           0.96 |       0.24 |
| F1-score  |           0.97 |       0.15 |
| Support   |        227,310 |      4,162 |

Additional metrics:

- Accuracy: 0.95
- ROC-AUC: 0.7954
- PR-AUC: 0.0711

**Confusion Matrix**

|                   | Predicted Non-Sepsis | Predicted Sepsis |
| ----------------- | -------------------: | ---------------: |
| Actual Non-Sepsis |              218,617 |            8,693 |
| Actual Sepsis     |                3,151 |            1,011 |

The Random Forest model achieved a higher ROC-AUC and overall accuracy than Logistic Regression. However, its recall for sepsis cases was lower, indicating that it missed a larger proportion of actual sepsis observations.

#### 6.5 Initial Model Comparison

The initial evaluation results of both models are summarised below.

| Evaluation Metric  | Logistic Regression | Random Forest |
| ------------------ | ------------------: | ------------: |
| Accuracy           |                0.76 |          0.95 |
| Precision (Sepsis) |                0.05 |          0.10 |
| Recall (Sepsis)    |                0.67 |          0.24 |
| F1-score (Sepsis)  |                0.09 |          0.15 |
| ROC-AUC            |              0.7813 |        0.7954 |
| PR-AUC             |              0.0726 |        0.0711 |

**Initial Findings**

The results indicate that Logistic Regression achieved higher recall for sepsis observations, while Random Forest achieved higher accuracy, ROC-AUC, precision, and F1-score for the positive class.

The findings highlight the difficulty of predicting sepsis in an imbalanced dataset. In particular, recall and PR-AUC are important metrics because identifying actual sepsis cases is a critical objective of the project.

These results represent the initial validation performance of the two models and will be considered in the subsequent shared evaluation and model comparison stages.


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
