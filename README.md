# Parkinson's Disease Prediction using Machine Learning

This project explores and discuss the application of various machine learning algorithms to predict Parkinson's disease using a dataset. The goal is to evaluate and compare the performance of different models to identify the most effective approach for prediction.

## Dataset

The dataset used in this project contains features extracted from voice recordings of individuals, some of whom have Parkinson's disease. The features include various biomedical voice measurements, and the target variable indicates the presence of Parkinson's disease.

## Machine Learning Models

The following machine learning models are implemented and tested:

1. **Linear Methods**:
   - Linear Regression (no penalty)
   - Lasso Regression (L1 penalty)
   - Ridge Regression (L2 penalty)

2. **Non-Parametric Methods**:
   - K-Nearest Neighbors (KNN)
   - Decision Tree

3. **Support Vector Machines (SVM)**:
   - Linear SVM
   - Radial Basis Function (RBF) SVM

4. **Ensemble Methods**:
   - Random Forest
   - Simple Averaging Ensemble
   - Stacking Ensemble (with a meta-learner)
   - AdaBoost
   - Gradient Boosting / XGBoost

## Acknowledgments

Dataset source: https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring

ATTRIBUTE INFORMATION:

subject# - Integer that uniquely identifies each subject

age - Subject age

sex - Subject gender '0' - male, '1' - female

test_time - Time since recruitment into the trial. The integer part is the 
number of days since recruitment.

motor_UPDRS - Clinician's motor UPDRS score, linearly interpolated

total_UPDRS - Clinician's total UPDRS score, linearly interpolated

Jitter(%),Jitter(Abs),Jitter:RAP,Jitter:PPQ5,Jitter:DDP - Several measures of 
variation in fundamental frequency

Shimmer,Shimmer(dB),Shimmer:APQ3,Shimmer:APQ5,Shimmer:APQ11,Shimmer:DDA - 
Several measures of variation in amplitude

NHR,HNR - Two measures of ratio of noise to tonal components in the voice

RPDE - A nonlinear dynamical complexity measure

DFA - Signal fractal scaling exponent

PPE - A nonlinear measure of fundamental frequency variation

A Tsanas, MA Little, PE McSharry, LO Ramig (2009)
'Accurate telemonitoring of Parkinson.s disease progression by non-invasive 
speech tests',
IEEE Transactions on Biomedical Engineering (to appear). 

## Getting Started

**Clone the Repository**:
   ```bash
   git clone https://github.com/DataPyPeil/ML_Parkinson_Telemonitoring.git
   cd ML_Parkinson_Telemonitoring
