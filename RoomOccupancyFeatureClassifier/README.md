# Room Occupancy Feature Classifier

## Overview

This project implements a **rule-based classifier** for room occupancy detection using data from the **UCI Machine Learning Repository**.

The project also evaluates the classifier using **K-fold cross-validation**, calculating key performance metrics including **error rate, recall, precision, specificity, and false alarm rate**. Additionally, **ROC curves** and **AUC values** are computed to assess classifier performance.

## Dataset

The dataset used is the [Occupancy Detection Data Set](https://archive.ics.uci.edu/dataset/357/occupancy+detection).  
It consists of multiple sensor readings collected from an office environment, with features including:

- **Temperature**
- **Humidity**
- **Light**
- **CO2**
- **Humidity Ratio**

The goal is to predict whether the office is **occupied (1)** or **not occupied (0)** based on these features.

## Implementation Details

- The dataset is split into **training** and **validation** subsets.
- A **simple rule-based classifier** is developed using **one feature at a time**.
- The classifier determines an **optimal threshold** for each feature using **K-fold cross-validation (K=10)**.
- **Performance Metrics**:
  - **Error rate**
  - **Recall (Sensitivity)**
  - **Precision**
  - **Specificity**
  - **False Alarm Rate**
- **ROC Curves** and **AUC values** are plotted for each feature.
- The **best-performing feature** (with the highest AUC) is selected for final testing.
