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

y_test = np.load(f"../../Features/tm_test_labels.npy")
y_pred = np.load(f"tm_pred_seq2Tm.npy")

r2, mse, rmse, mae = find_metrics(y_test, y_pred)

print(f"R2: {r2}, MSE: {mse}, RMSE: {rmse}, MAE: {mae}")