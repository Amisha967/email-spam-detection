# Email Spam Detection Using Machine Learning

## Project Overview

This project develops a machine learning pipeline for classifying emails as **spam** or **non-spam** using the UCI Spambase dataset.

The workflow includes data preprocessing, exploratory data analysis, feature standardization, model training, evaluation, feature analysis, and model persistence.

## Dataset

The project uses the UCI Spambase dataset.

- Original records: 4,601
- Input features: 57
- Target variable: `spam`
- Spam: `1`
- Non-spam: `0`
- Duplicate records removed: 391
- Final records used: 4,210

The dataset contains numerical features based on word frequencies, character frequencies, and capital-letter run statistics.

## Machine Learning Workflow

1. Load the Spambase dataset
2. Perform data quality checks
3. Analyze class distribution
4. Remove duplicate records
5. Split data into training and testing sets
6. Standardize numerical features
7. Train multiple classification models
8. Evaluate model performance
9. Analyze Logistic Regression coefficients
10. Save trained models using Joblib
11. Build a reusable prediction pipeline

## Models Used

### Gaussian Naive Bayes

A probabilistic classification algorithm based on Bayes' theorem.

### Logistic Regression

A supervised classification model used to classify emails as spam or non-spam.

### Support Vector Machine

An SVM with an RBF kernel was used to learn a nonlinear decision boundary.

## Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### Logistic Regression

- Accuracy: **93.47%**
- Precision: **92.97%**
- Recall: **90.48%**
- F1-score: **91.70%**

### Support Vector Machine

- Accuracy: **93.23%**
- Precision: **93.19%**
- Recall: **89.58%**
- F1-score: **91.35%**

The complete comparison of all three models is available in the Jupyter notebook.

## Feature Analysis

Logistic Regression coefficients were analyzed to identify features with larger absolute contributions to the classification decision.

## Model Persistence

The trained models and scaler were saved using Joblib:

```text
models/
├── logistic_regression_model.pkl
├── naive_bayes_model.pkl
├── svm_model.pkl
└── scaler.pkl
