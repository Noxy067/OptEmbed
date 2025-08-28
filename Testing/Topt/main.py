from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import pandas as pd
import numpy as np
import random
import csv
import torch
from sklearn.metrics import roc_curve
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import pickle


torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


def find_metrics(y_test, y_pred):
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    return r2, mse, rmse, mae


feature_groups_single = ['PAAC', 'DPC', 'CKSAAGP', 'SOCNumber']

feature_arrays = []

for feature_group in feature_groups_single:
    print(f"Processing {feature_group}...")
    # Load each feature group as a NumPy array
    X1 = pd.read_csv(f"../../Features/Topt/{feature_group}/train.csv", header=None).values
    X = pd.read_csv(f"../../Features/Topt/{feature_group}/test.csv", header=None).values
    length = X.shape[0]
    concat = np.concatenate((X1, X), axis=0)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(concat)
    X = X[-length:, :]
    feature_arrays.append(X)

X = pd.read_csv(f"../../Features/Topt/esmc_6b/embeddings_test.csv", header=None).values

feature_arrays.append(X)
# Concatenate all features on the column axis (axis=1)
X_test = np.concatenate(feature_arrays, axis=1)

print(X.shape)
y_test = np.load(f"../../Features/topt_test_labels.npy")

with open(f"base_model_1.pkl", 'rb') as f:
    base_model_1 = pickle.load(f)

with open(f"meta_model.pkl", 'rb') as f:
    meta_model = pickle.load(f)

y_test_proba_1 = base_model_1.predict(X_test).reshape(-1, 1)
X_test = np.concatenate((X_test, y_test_proba_1), axis=1)

print(f"X_test shape after concatenation: {X_test.shape}")

y_pred = meta_model.predict(X_test)

r2, mse, rmse, mae = find_metrics(y_test, y_pred)
print(f"Meta Model Performance: R2={r2}, MSE={mse}, RMSE={rmse}, MAE={mae}")

# save to csv with 3 decimal points
output_file = "results.csv"
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['R2', 'MSE', 'RMSE', 'MAE'])
    writer.writerow([f"{r2:.3f}", f"{mse:.3f}", f"{rmse:.3f}", f"{mae:.3f}"])