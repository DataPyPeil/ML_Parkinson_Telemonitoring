# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 19:57:50 2025

@author: peill
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler


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
        plt.title(X_df.columns[k])
        plt.hist(X[:,k], bins=25, color=colors[k%2])
    else:
        plt.title(y_df.columns[k-n])
        plt.hist(y[:,k-n], bins=10, color=colors[2])
plt.show()     

plt.figure(figsize=(10,10))
sns.boxplot(X)
plt.show()


# Analyse multivariée
columns_xy = np.concatenate((columns_modif, y_df.columns.values), axis=0)
df_all = pd.DataFrame(np.concatenate((X, y), axis=1), columns=columns_xy)

plt.figure(figsize=(15,15))
plt.title('Matrice de corrélation', fontsize=15)
sns.heatmap(df_all.corr(), annot=True, fmt='.2f')
plt.show()


# Analyse outliers if any


    
#%% Linear models

# --> peu de correlation lineaire
# --> do polynomial features + Linear avec Lasso et Ridge regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

def RMSE(y_gt, y_pred):
    mse = np.sum((y_gt-y_pred)**2)
    return np.sqrt(mse)

def R_squared(y_gt, y_pred):
    u = ((y_gt - y_pred)** 2).sum()
    v = ((y_gt - y_gt.mean()) ** 2).sum()
    return 1-u/v

def plot_actualVSpredicted(y_gt, y_pred, model_name):
    
    plt.figure(figsize=(7,7))
    plt.title(f'Actual vs Predicted\n{model_name}')
    x = np.linspace(np.min(y_gt)*0.7, np.max(y_gt)*1.3, 50)
    y = x
    error = 0.05 * y
    y_min = y - error
    y_max = y + error

    plt.scatter(y_gt, y_pred, alpha=0.4, s=10)
    plt.plot(x, y, lw=1, ls='-', color='lime')
    plt.fill_between(x, y_min, y_max, color="lime", alpha=0.2, label="ErrorBand (±5%)")
    
    plt.xlim(0, 1.4)
    plt.ylim(np.min(y_pred)*0.8, np.max(y_pred)*1.2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.grid(color='grey', linestyle=':', linewidth=0.5)
    plt.axis('equal')
    plt.legend()
    plt.show()

# Polynomial models
print('\n--- Polynomial models ---')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

reg_models = {'Linear':LinearRegression(), 'Ridge':Ridge(), 'Lasso':Lasso()}

for name, reg_model in zip(reg_models.keys(), reg_models.values()):
    for degree in [1, 3, 5]:
        model = make_pipeline(PolynomialFeatures(degree), reg_model)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f'{name}\tDegree {degree}\tRMSE={RMSE(y_test, y_pred):.2f}\t\tR²={R_squared(y_test, y_pred):.2f}')
        if name=='Ridge' and degree==5:
            plot_actualVSpredicted(y_test, y_pred, 'Ridge_5')