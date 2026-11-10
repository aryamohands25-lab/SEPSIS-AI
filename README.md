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
Develop machine learning models for early sepsis prediction.

Collect, clean, and preprocess ICU patient data.

Perform exploratory data analysis and feature engineering.

Implement and compare multiple machine learning algorithms.

Evaluate model performance using appropriate classification metrics.

Incorporate explainable AI techniques to understand model predictions.

Assess the stability of model explanations.

Monitor changes in patient risk over time.

Develop an interactive dashboard for presenting predictions and analysis.

## 4. Dataset Description
The project uses the PhysioNet/Computing in Cardiology Challenge 2019 dataset.

Dataset source: https://physionet.org/content/challenge-2019/1.0.0/

The dataset contains hourly clinical observations collected from ICU patients and includes physiological measurements, laboratory results, demographic information, and sepsis labels.

Some of the important variables include:

Heart rate

Blood pressure

Respiratory rate

Body temperature

Oxygen saturation

Laboratory measurements

Patient demographics

ICU length of stay

Sepsis labels

The time-series structure of the dataset allows the analysis of changes in clinical measurements over time.

## 5. Overall Project Methodology
The project follows a structured machine learning workflow:

Dataset acquisition and loading.

Data cleaning and preprocessing.

Missing-value handling.

Exploratory Data Analysis (EDA).

Feature engineering.

Patient-level data splitting.

Machine learning model development.

Model evaluation and comparison.

Explainability using SHAP.

Explainability Stability Score (ESS).

Risk Trajectory Visualization (RTV).

Dashboard integration and testing.
