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

# Analyse outliers if any

# Analyse multivariée
columns_xy = np.concatenate((columns_modif, y_df.columns.values), axis=0)
df_all = pd.DataFrame(np.concatenate((X, y), axis=1), columns=columns_xy)

plt.figure(figsize=(15,15))
plt.title('Matrice de corrélation', fontsize=15)
sns.heatmap(df_all.corr(), annot=True, fmt='.2f')
plt.show()
# --> peu de correlation lineaire
# --> do polynomial features + Linear avec Lasso et Ridge regression




    
    