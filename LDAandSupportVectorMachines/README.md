# Synthetic Dataset Classifier using LDA and SVM

## Overview

This project explores the classification of two synthetic datasets—**Moons** and **Circles**—using multiple machine learning algorithms:

- **Linear Discriminant Analysis (LDA)**
- **Support Vector Machines (SVM)** with:
  - Linear kernel
  - 2nd-degree polynomial kernel
  - Radial Basis Function (RBF/Gaussian) kernel

Additionally, a **grid search** is performed to analyze how the SVM's **C** and **gamma** parameters affect model performance, revealing cases of underfitting and overfitting.

---

## Datasets

The synthetic datasets are generated using `scikit-learn`:

- `make_moons(n_samples=300, noise=0.15)`
- `make_circles(n_samples=300, noise=0.15, factor=0.5)`

These datasets are designed to challenge linear classifiers and are ideal for visual demonstrations of decision boundaries.

---

## Implementation Details

### 1. Data Visualization

- Initial scatter plots visualize class distribution for each dataset.
- Data points are color-coded by class label.

### 2. LDA Classification

- Datasets are split into training and test sets using `train_test_split`.
- LDA (`LinearDiscriminantAnalysis`) is trained on the training data.
- Predictions are made on the test set.
- Misclassified points are identified.
- A custom `plot_decision_boundary()` function is used to display:
  - Decision regions
  - Correctly and incorrectly classified points

### 3. SVM Classification

Three SVM variants are applied:

- **Linear Kernel**
- **Polynomial Kernel (degree=2)**
- **RBF Kernel**

For each:
- The classifier is trained and evaluated.
- Accuracy and classification metrics (precision, recall, F1-score) are printed.
- Decision boundaries and misclassified points are visualized.

### 4. Grid Search on Gaussian Kernel

- A grid search is conducted over a range of `C` and `gamma` values:
  - `C`: [0.01, 0.1, 1, 10, 100, 1000]
  - `gamma`: [0.01, 0.1, 1, 10, 100]
- Accuracy is printed for each `(C, gamma)` pair.
- The best and worst-performing models are identified.
- Their decision boundaries are visualized.

---

## Evaluation Metrics

Each classifier's performance is measured using:

- **Precision**
- **Recall**
- **F1-Score**
- **Accuracy**

Reported using `classification_report` from `sklearn.metrics`.

---

## Results Summary

- **LDA** performs well on the **Moons** dataset but poorly on **Circles**, as expected for a linear method.
- **SVM with RBF Kernel** delivers the best performance across both datasets.
- The **grid search** confirms that selecting appropriate `C` and `gamma` values is crucial to avoid underfitting or overfitting.
