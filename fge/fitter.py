import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
import statsmodels.api as sm

from sklearn.metrics import r2_score, accuracy_score

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer

from .utils import flatten, c_statistic_harrell
from typing import Dict, Any

class PolyFitter():
    def __init__(self, task_type: str, data: Dict[str, Any], original_score: None | float):
        self.task_dict = {
            'reg': (LinearRegression, r2_score, dict(fit_intercept=True)),
            'binary': (LogisticRegression, accuracy_score, dict(fit_intercept=True, max_iter=1000)),
            'survival': (LinearRegression, c_statistic_harrell, dict(fit_intercept=True))
        }

        self.task_type = task_type
        self.task_model, self.task_metric, self.args = self.task_dict[task_type]
        self.data = data
        self.feature_names = data['X_train'].columns

        if original_score is None:
            self.original_score = self.fit_all(full=True)
        else:
            self.original_score = original_score

        self.linear_model_score = self.get_linear_model_score()

    def get_preprocessor(self, X):
        numerical_columns_selector = selector(dtype_include=np.float64, dtype_exclude=object)
        categorical_columns_selector = selector(dtype_include=np.int64) #pd.CategoricalDtype)
        numerical_columns = numerical_columns_selector(X)
        categorical_columns = categorical_columns_selector(X)
        categorical_preprocessor = OneHotEncoder(
            handle_unknown='infrequent_if_exist', sparse=True
        ) 
        # categorical_preprocessor = FunctionTransformer(lambda x: x)
        numerical_preprocessor = StandardScaler()
        preprocessor = ColumnTransformer([
            ('onehot', categorical_preprocessor, categorical_columns),
            ('standard', numerical_preprocessor, numerical_columns)
        ])
        feature_names = categorical_columns + numerical_columns
        return preprocessor, feature_names 

    def get_new_X(self, X_original, trials): 
        X = X_original.copy()
        for cmbs in trials:
            c_names = [self.feature_names[i] for i in flatten(cmbs)]
            X['+'.join(c_names)] = X.loc[:, c_names].prod(1)
        return X

    def fit(self, model, X_train, X_test, y_train, y_test):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return self.task_metric(y_test, y_pred)

    def fit_selected(self, trials):
        X_train = self.get_new_X(self.data['X_train'], trials)
        X_test = self.get_new_X(self.data['X_test'], trials)
        y_train = self.data['y_train']
        y_test = self.data['y_test']
        preprocessor, feature_names = self.get_preprocessor(X=X_train)
        model = make_pipeline(preprocessor, self.task_model(**self.args))
        performance_selected = self.fit(
            model, X_train, X_test, y_train, y_test
        )
        return performance_selected

    def get_gap(self, performance):
        """
        gap    
        """
        # gain = self.original_score - performance
        # diff = self.original_score - self.min_score  # > 0
        # s = gain / diff
        return self.original_score - performance

    def get_interaction_gap(self, trials):
        performance_selected = self.fit_selected(trials)
        gap = self.get_gap(performance_selected)
        return gap

    def get_linear_model_score(self):
        X_train = self.data['X_train']
        y_train = self.data['y_train']
        X_test = self.data['X_test']
        y_test = self.data['y_test']
        preprocessor, _ = self.get_preprocessor(X_train)
        model = make_pipeline(preprocessor, self.task_model(**self.args))
        performance = self.fit(
            model, X_train, X_test, y_train, y_test
        )
        return performance