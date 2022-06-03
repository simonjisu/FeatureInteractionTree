from matplotlib.pyplot import xticks
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import PolynomialFeatures

def add_selective_interactions(X_original, interaction_dict, cut_off_level=1):
    X = X_original.copy()
    for curr_level in range(1, cut_off_level + 1):
        for interaction in interaction_dict[curr_level]:
            feature_names = interaction.split("+")
            X[interaction] = 1
            for feature in feature_names:
                X[interaction] *= X[feature]
    return X

def get_selective_polynomial_features(X_train, X_test, interaction_dict, cut_off_level=1):
    X_train_selective_polynomial = add_selective_interactions(X_train, interaction_dict, cut_off_level)
    X_test_selective_polynomial = add_selective_interactions(X_test, interaction_dict, cut_off_level)
    return X_train_selective_polynomial, X_test_selective_polynomial

def get_full_polynomial_features(X_train_original, X_test_original):
    X_train, X_test = X_train_original.copy(), X_test_original.copy()
    poly = PolynomialFeatures(X_train.shape[1], interaction_only=True, include_bias=False)
    X_train_polynomial, X_test_polynomial = poly.fit_transform(X_train), poly.fit_transform(X_test)
    return X_train_polynomial, X_test_polynomial

def get_polynomial_model_performance(X_train, y_train, X_test, y_test):
    poly_regression_model = LinearRegression()
    poly_regression_model.fit(X_train, y_train)

    y_pred = poly_regression_model.predict(X_test)
    return mse(y_test, y_pred)

def get_performance_decrease(X_train, y_train, X_test, y_test, interaction_dict, cut_off_level, g_func):
    X_train_full, X_test_full = get_full_polynomial_features(X_train, X_test)
    X_train_selected, X_test_selected = get_selective_polynomial_features(X_train, X_test, interaction_dict, cut_off_level)
    best_performance = get_polynomial_model_performance(X_train_full, y_train, X_test_full, y_test)
    reduced_performance = get_polynomial_model_performance(X_train_selected, y_train, X_test_selected, y_test)
    reduction = best_performance - reduced_performance
    print(f"For {g_func}, performance reduction was {reduction}")
    return reduction
    