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



feature_arrays = []

X = pd.read_csv(f"../../Features/Tm/esmc_6b/embeddings_train.csv", header=None).values

feature_arrays.append(X)
# Concatenate all features on the column axis (axis=1)
X = np.concatenate(feature_arrays, axis=1)

print(X.shape)
y = np.load(f"../../Features/tm_train_labels.npy")

X_base, X_meta, y_base, y_meta = train_test_split(X, y, test_size=0.4, random_state=42)

base_model_1 = SVR(C=10, epsilon=0.1, gamma='scale', kernel='rbf')

base_model_1.fit(X_base, y_base)
with open('base_model_1.pkl', 'wb') as f:
    pickle.dump(base_model_1, f)

y_meta_pred_1 = base_model_1.predict(X_meta).reshape(-1, 1)

X_meta = np.concatenate((X_meta, y_meta_pred_1), axis=1)
print(X_meta.shape)

meta_model = XGBRegressor(
            objective='reg:squarederror',  # Required for regression
            n_estimators=200,
            learning_rate=0.1,
            max_depth=3,
            gamma=5,
            random_state=42
        )
meta_model.fit(X_meta, y_meta)

with open('meta_model.pkl', 'wb') as f:
    pickle.dump(meta_model, f)
