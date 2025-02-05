# Parkinson's Disease Prediction using Machine Learning

This project explores the application of various machine learning algorithms to predict Parkinson's disease using a dataset. The goal is to evaluate and compare the performance of different models to identify the most effective approach for prediction.

## Dataset

The dataset used in this project contains features extracted from voice recordings of individuals, some of whom have Parkinson's disease. The features include various biomedical voice measurements, and the target variable indicates the presence of Parkinson's disease.

## Machine Learning Models

The following machine learning models are implemented and tested:

1. **Linear Methods**:
   - Lasso Regression (L1 penalty)
   - Ridge Regression (L2 penalty)

2. **Non-Parametric Methods**:
   - K-Nearest Neighbors (KNN)
   - Decision Tree

3. **Support Vector Machines (SVM)**:
   - Linear SVM
   - Radial Basis Function (RBF) SVM

4. **Ensemble Methods**:
   - Simple Averaging Ensemble
   - Majority Voting Ensemble
   - Stacking Ensemble (with a meta-learner)
   - Random Forest
   - AdaBoost
   - Gradient Boosting / XGBoost

## Acknowledgments

Dataset source: https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring

## Getting Started

**Clone the Repository**:
   ```bash
   git clone https://github.com/DataPyPeil/ML_Parkinson_Telemonitoring.git
   cd ML_Parkinson_Telemonitoring
