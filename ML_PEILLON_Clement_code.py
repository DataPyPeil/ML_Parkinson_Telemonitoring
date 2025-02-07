# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 19:57:50 2025

@author: peill
"""
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

root_path = Path(__file__)
plot_path = root_path.parents[0] / 'plots'
#%% IMPORT DATASET

from ucimlrepo import fetch_ucirepo 

# fetch dataset 
parkinsons_telemonitoring = fetch_ucirepo(id=189) 
  
# data (as pandas dataframes) 
X_df = parkinsons_telemonitoring.data.features 
y_df = parkinsons_telemonitoring.data.targets 

#%% EDA

print(f'\nNombre de données :\t\t\t{X_df.shape[0]}')
print(f'Nombre de variables explicatives :\t{X_df.shape[1]}')
print(f'Variables à expliquer : {y_df.columns.values}')
nan_counts = X_df.isna().sum().sum()
print(f'Nan values: {nan_counts}')
# print(f'\nNan values par colonnes\n{X_df.isna().sum()}')

#%% Statistics

m, n = X_df.shape
colors=['navy','darkorange', 'lime']

X_df.describe()
y_df.describe()

# Drop youngest -> not relevant = outlier
ids = X_df[X_df['age'] == 36].index
X_df = X_df.drop(index=ids)
y_df = y_df.drop(index=ids)

X  = X_df.to_numpy()
y = y_df.to_numpy()

# scaler = MinMaxScaler()
# X = scaler.fit_transform(X)
# y = y_df.to_numpy()

# Analyse univariée
plt.figure(figsize=(22,12))
plt.suptitle('Histogramme de répartition de chaque paramètre', fontsize=25)
for k in range (n+2):
    plt.subplot(3,7,k+1)
    if k<n:
        plt.title(X_df.columns[k])
        plt.hist(X[:,k], bins=25, color=colors[k%2])
    else:
        plt.title(y_df.columns[k-n])
        plt.hist(y[:,k-n], color=colors[2])
# plt.savefig(plot_path / 'Feature_Distrib.png')
plt.show()    

# Répartition exponentielle --> Modifier distribution 
columns_modif = X_df.columns.values.copy()
for k in range(2,14):
    X[:,k] = np.log(X[:,k]) 
    columns_modif[k] = 'log(' + columns_modif[k] + ')'

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
y = y_df.to_numpy()
    
 
plt.figure(figsize=(22,12))
plt.suptitle('Histogramme de répartition de chaque paramètre\nLois exponentielles modifiées', fontsize=25)
for k in range (n+2):
    plt.subplot(3,7,k+1)
    if k<n:
        plt.title(columns_modif[k])
        plt.hist(X[:,k], bins=25, color=colors[k%2])
    else:
        plt.title(y_df.columns[k-n])
        plt.hist(y[:,k-n], bins=10, color=colors[2])
# plt.savefig(plot_path / 'Feature_DistribModified.png')
plt.show()     

plt.figure(figsize=(10,10))
sns.boxplot(X)
# plt.savefig(plot_path / 'Feature_BoxPlot.png')
plt.show()


# Analyse multivariée
columns_xy = np.concatenate((columns_modif, y_df.columns.values), axis=0)
df_all = pd.DataFrame(np.concatenate((X, y), axis=1), columns=columns_xy)

plt.figure(figsize=(15,15))
plt.title('Matrice de corrélation', fontsize=15)
sns.heatmap(df_all.corr(), annot=True, fmt='.2f')
# plt.savefig(plot_path / 'Feature_heatmap.png')
plt.show()


# Drop motor_UPDRS inutile
y = y[:,1]

# Split dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

#%% Functions & Imports

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def RMSE(y_gt, y_pred):
    mse = np.sum((y_gt-y_pred)**2)
    return np.sqrt(mse)

def R_squared(y_gt, y_pred):
    u = ((y_gt - y_pred)** 2).sum()
    v = ((y_gt - y_gt.mean()) ** 2).sum()
    return np.array(1-u/v)

def plot_actualVSpredicted(y_gt, y_pred, model_name, rmse, r2):
    y_name = 'total_UPDRS'
    plt.figure(figsize=(7,7))
    plt.title(f'Actual vs Predicted\n{model_name}\nRMSE={rmse:.2f}   |   R²={r2:.2f}', fontsize=15)
    
    x = np.linspace(np.min(y_gt)*0.7, np.max(y_gt)*1.3, 50)
    y = x
    error = 0.05 * y
    y_min = y - error
    y_max = y + error

    plt.plot(x, y, lw=1, ls='-', color='lime')
    plt.scatter(y_gt, y_pred, color='navy', alpha=0.4, s=10, edgecolors='none')
    plt.fill_between(x, y_min, y_max, color='darkorange', alpha=0.2, label="ErrorBand (±5%)")
    
    
    plt.xlim(0, 1.4)
    plt.ylim(np.min(y_pred)*0.8, np.max(y_pred)*1.2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.grid(color='grey', linestyle=':', linewidth=0.5)
    plt.axis('equal')
    plt.legend()
    plt.savefig(plot_path / f'{model_name}.png')
    plt.show()
    
#%% Polynomial models
print('\n--- Polynomial models ---')

reg_models = {'Linear':LinearRegression(), 'Ridge':Ridge(), 'Lasso':Lasso(1e-3)}
# reg_models={'Lasso':Lasso(1e-2)}

for name, reg_model in zip(reg_models.keys(), reg_models.values()):
    for degree in [1, 3, 5]:
        model = make_pipeline(PolynomialFeatures(degree), reg_model)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_train)
        rmse = RMSE(y_train, y_pred)
        r2 = R_squared(y_train, y_pred)
        print(f'{name}\tDegree {degree}\tRMSE={rmse:.2f} R²={r2:.2f}')
        plot_actualVSpredicted(y_train, y_pred, f'{name}_{degree}', rmse, r2)
   
         
#%% Non parametric KNN 
print('\n--- KNN ---')           
from sklearn.neighbors import KNeighborsRegressor
n_neighbors=[3, 5, 7, 9]
for n in n_neighbors[::-1]:
    model = KNeighborsRegressor(n, weights='distance')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = RMSE(y_test, y_pred)
    r2 = R_squared(y_test, y_pred)
    print(f'KNN\tn_neighbors={n}\tRMSE={rmse:.2f} R²={r2:.2f}')
    
    plot_actualVSpredicted(y_test, y_pred, f'KNN_{n}', rmse, r2)

#%% SVM Regression
from sklearn.svm import SVR
print('\n--- Linear SVM ---')

model = SVR(kernel='linear')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = RMSE(y_test, y_pred)
r2 = R_squared(y_test, y_pred)
print(f'SVM\tRMSE={rmse:.2f} R²={r2:.2f}')

plot_actualVSpredicted(y_test, y_pred, 'Linear SVM', rmse, r2)


print('\n--- rbf SVM ---')
model = SVR(kernel='rbf')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = RMSE(y_test, y_pred)
r2 = R_squared(y_test, y_pred)
print(f'SVM\tRMSE={rmse:.2f} R²={r2:.2f}')

plot_actualVSpredicted(y_test, y_pred, 'SVM', rmse, r2)

#%% RandomForest
print('\n--- RandomForest ---')
from sklearn.ensemble import RandomForestRegressor

n_estimator=[5, 10, 50, 100]
for n in n_estimator:
    model = RandomForestRegressor(n_estimators=n, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = RMSE(y_test, y_pred)
    r2 = R_squared(y_test, y_pred)
    print(f'RandomForest\tn_estimator={n}\tRMSE={rmse:.2f} R²={r2:.2f}')
    
    plot_actualVSpredicted(y_test, y_pred, f'RandomForest_{n}', rmse, r2)
    
    
#%% Ensemble method - VotingRegressor - Average
print('\n--- VotingRegressor ---')
from sklearn.ensemble import VotingRegressor

model_A = LinearRegression()
model_B = KNeighborsRegressor(n_neighbors=5)
model_C = SVR(kernel='rbf')
model_D = RandomForestRegressor(n_estimators=10)

estimators = [('linear', model_A),
              ('KNN', model_B),
              ('SVM', model_C),
              ('RF', model_D)]

model = VotingRegressor(estimators=estimators)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = RMSE(y_test, y_pred)
r2 = R_squared(y_test, y_pred)
print(f'VotingRegressor\tRMSE={rmse:.2f} R²={r2:.2f}')

plot_actualVSpredicted(y_test, y_pred, 'VotingRegressor', rmse, r2)

#%% Ensemble Stacking 
print('\n--- StackingRegressor ---')
from sklearn.ensemble import StackingRegressor

model_A = LinearRegression()
model_B = KNeighborsRegressor(n_neighbors=5)
model_C = SVR(kernel='rbf')
model_D = RandomForestRegressor(n_estimators=10)

estimators = [('linear', model_A),
              ('KNN', model_B),
              ('SVM', model_C)]

model = StackingRegressor(estimators=estimators, final_estimator=model_D)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = RMSE(y_test, y_pred)
r2 = R_squared(y_test, y_pred)
print(f'StackingRegressor\tRMSE={rmse:.2f} R²={r2:.2f}')

plot_actualVSpredicted(y_test, y_pred, 'StackingRegressor', rmse, r2)


#%% AdaBoost 
print('\n--- AdaBoost ---')
from sklearn.ensemble import AdaBoostRegressor

n_estimator=[5, 10, 50]
for n in n_estimator:
    estimator = RandomForestRegressor(10)
    model = AdaBoostRegressor(estimator=estimator, n_estimators=n)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = RMSE(y_test, y_pred)
    r2 = R_squared(y_test, y_pred)
    print(f'AdaBoost\tn_estimator={n}\tRMSE={rmse:.2f} R²={r2:.2f}')
    
    plot_actualVSpredicted(y_test, y_pred, f'AdaBoost{n}', rmse, r2)
    

#%% XGBoost
print('\n--- GradientBoosting ---')
from sklearn.ensemble import GradientBoostingRegressor

n_estimator=[5, 10, 50, 100, 500]
for n in n_estimator:
    model = GradientBoostingRegressor(n_estimators=n)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test) 
    rmse = RMSE(y_test, y_pred)
    r2 = R_squared(y_test, y_pred)
    print(f'GradientBoosting\tn_estimator={n}\tRMSE={rmse:.2f} R²={r2:.2f}')
    
    plot_actualVSpredicted(y_test, y_pred, f'GradientBoosting{n}', rmse, r2)

#%% best model optimisation

import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 20, 50)
    max_depth = trial.suggest_int('max_depth', 5, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    
    model = RandomForestRegressor(n_estimators=n_estimators,
                                  max_depth=max_depth,
                                  min_samples_split=min_samples_split)
    
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    
    return float(np.min(score))

%time study = optuna.create_study(direction='maximize')
%time study.optimize(objective, n_trials=30)

study.best_params

best_model = RandomForestRegressor(n_estimators=study.best_params['n_estimators'],
                                   max_depth=study.best_params['max_depth'],
                                   min_samples_split=study.best_params['min_samples_split'])
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
rmse = RMSE(y_test, y_pred)
r2 = R_squared(y_test, y_pred)
print(f'Best RandomForest\tRMSE={rmse:.2f} R²={r2:.2f}')

plot_actualVSpredicted(y_test, y_pred, f'Best RandomForest', rmse, r2)

importance_df = pd.DataFrame({
    'Feature': X_df.columns,
    'Importance': best_model.feature_importances_
})

# Sort by importance (descending)
importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

#%% Patient wise predictions

""" Make predictions for a patient
    dont split patient in X_train X_test
    Plot the whole patient evolution i.e. total_UPDRS with respect to test_time
    """

