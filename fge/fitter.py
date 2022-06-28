import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, accuracy_score

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

from .utils import flatten, c_statistic_harrell
from typing import Dict, Any

class PolyFitter():
    def __init__(self, task_type: str, data: Dict[str, Any], original_score: None | float, max_iter=200):
        self.task_dict = {
            'reg': (LinearRegression, r2_score, dict()),
            'binary': (LogisticRegression, accuracy_score, dict(max_iter=max_iter)),
            'survival': (LinearRegression, c_statistic_harrell, dict())
        }
        self.task_type = task_type
        self.task_model, self.task_metric, self.args = self.task_dict[task_type]
        self.data = data
        
        if original_score is None:
            self.original_score = self.fit_all(full=True)
        else:
            self.original_score = original_score

        self.min_score = self.fit_all(full=False)
        if self.original_score <= self.min_score:
            print("[Warning] simple linear model performance is better than the original model")
            print("This might cause the `gain` unstable")
        
    def get_new_X(self, X_original, nodes):
        X = X_original.to_numpy().copy()
        for feature in nodes:
            if isinstance(feature, int):
                continue
            X = np.concatenate((X, X[:, tuple(flatten(feature))].prod(1)[:, None]), axis=1)
        if isinstance(self.task_model, LinearRegression):
            bias_term = np.ones((X.shape[0], 1))
            X = np.concatenate([bias_term, X], axis=1)
        return X

    def get_selected_X(self, nodes):
        X_train, X_test = self.data['X_train'].copy(), self.data['X_test'].copy()
        new_X_train = self.get_new_X(X_train, nodes)
        new_X_test = self.get_new_X(X_test, nodes)
        return new_X_train, new_X_test

    def fit(self, X_train, X_test, y_train, y_test):
        poly_model = make_pipeline(StandardScaler(), self.task_model(**self.args))
        poly_model.fit(X_train, y_train)
        y_pred = poly_model.predict(X_test)
        return self.task_metric(y_test, y_pred)

    def fit_all(self, full=False):
        X_train, X_test = self.data['X_train'].copy(), self.data['X_test'].copy()
        if full:
            poly = PolynomialFeatures(2, interaction_only=True, include_bias=isinstance(self.task_model, LinearRegression))
            X_train, X_test = poly.fit_transform(X_train), poly.fit_transform(X_test)
        
        performance = self.fit(
            X_train, X_test, self.data['y_train'], self.data['y_test']
        )
        return performance
        
    def fit_selected(self, nodes):
        X_train_selected, X_test_selected = self.get_selected_X(nodes)
        performance_selected = self.fit(
            X_train_selected, X_test_selected, self.data['y_train'], self.data['y_test']
        )
        return performance_selected

    def get_gain(self, performance):
        """
        gain    
        """
        # gain = self.original_score - performance
        # diff = self.original_score - self.min_score  # > 0
        # s = gain / diff
        return self.original_score - performance

    def get_interaction_gain(self, nodes):
        performance_selected = self.fit_selected(nodes)
        gain = self.get_gain(performance_selected)
        return gain
