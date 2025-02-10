# COVID-19 Mortality Prediction Using Bayesian Networks

## Overview

This project implements a **Bayesian Network Model** to predict the probability of COVID-19 patient mortality using clinical and demographic data. The model is trained using real-world COVID-19 patient data, applying **structure learning** and **parameter learning** techniques.

The script:
- **Cleans and preprocesses** patient data.
- **Builds a Bayesian Network** using `bnlearn`.
- **Performs K-Fold Cross-Validation** (K=10) to assess model accuracy.
- **Generates predictions** for patient mortality (`DATE_DIED`).
- **Computes accuracy metrics** including confusion matrix, precision, and recall.

## Dataset

The dataset used in this project (`Covid_Data.csv`) contains information on COVID-19 patients, including:
- **Boolean Features:** Presence of comorbidities (e.g., Diabetes, Hypertension, COPD).
- **Categorical Features:** Gender, hospitalization status, medical unit classification.
- **Target Variable:** `DATE_DIED` (converted to 0 = Alive, 1 = Deceased).

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
