import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from sklearn.metrics import r2_score, accuracy_score

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, StandardScaler
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer

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
        self.feature_names = self.data['X_train'].columns
        # Preprocessor pipeline
        numerical_columns_selector = selector(dtype_include=[np.float64, np.int64], dtype_exclude=object)
        categorical_columns_selector = selector(dtype_include=np.int32)#pd.CategoricalDtype)
        self.numerical_columns = numerical_columns_selector(self.data['X_train'])
        self.categorical_columns = categorical_columns_selector(self.data['X_train'])
        categorical_preprocessor = OneHotEncoder()
        numerical_preprocessor = StandardScaler()
        self.preprocessor = ColumnTransformer([
            ('onehot', categorical_preprocessor, self.categorical_columns),
            ('standard', numerical_preprocessor, self.numerical_columns)
        ])
        self.X_train_dense, self.nonzeros_train, self.f_names = self.preprocess_pipeline(self.data['X_train'], train=True)
        self.X_test_dense, self.nonzeros_test, _ = self.preprocess_pipeline(self.data['X_test'], train=False, f_names=self.f_names)

        if original_score is None:
            self.original_score = self.fit_all(full=True)
        else:
            self.original_score = original_score

        # self.min_score = self.fit_all(full=False)
        # if self.original_score <= self.min_score:
        #     print("[Warning] simple linear model performance is better than the original model")
        #     print("This might cause the `gap` unstable")

    def preprocess_pipeline(self, X, train=True, f_names=None):
        if train:
            X_new = self.preprocessor.fit_transform(X)
            f_names = self.preprocessor.get_feature_names_out(X.columns)
            f_names = pd.Index([f.split('__')[1] for f in f_names])
            X_new_dense = pd.DataFrame(X_new.toarray(), index=X.index, columns=f_names)
        else:
            X_new = self.preprocessor.transform(X)
            # X_new_dense = self.preprocessor.transform(X)
            # X_new = csr_matrix(X_new_dense)
            X_new_dense = pd.DataFrame(X_new.toarray(), index=X.index, columns=f_names)
        nonzeros = list(map(lambda x: f_names[x], np.split(X_new.indices, X_new.indptr[1:-1])))
        return X_new_dense, nonzeros, f_names 

    def fit(self, X_train, X_test, y_train, y_test):
        # scaler = StandardScaler()
        # scaler = scaler.fit(X_train)
        poly_model = self.task_model(**self.args)
        poly_model.fit(X_train, y_train)
        y_pred = poly_model.predict(X_test)
        return self.task_metric(y_test, y_pred)

    # def fit_all(self, full=False):
    #     X_train, X_test = self.data['X_train'].copy(), self.data['X_test'].copy()
    #     if full:
    #         poly = PolynomialFeatures(2, interaction_only=True, include_bias=isinstance(self.task_model, LinearRegression))
    #         X_train, X_test = poly.fit_transform(X_train), poly.fit_transform(X_test)
        
    #     performance = self.fit(
    #         X_train, X_test, self.data['y_train'], self.data['y_test']
    #     )
    #     return performance

    def get_new_X(self, trials, train=True):
        if train:
            X_dense = self.X_train_dense.copy()
            nonzeros = self.nonzeros_train
        else:
            X_dense = self.X_test_dense.copy()
            nonzeros = self.nonzeros_test
        for feature in trials:
            c_names = [self.feature_names[i] for i in flatten(feature)]
            combined_c_names = '*'.join(c_names)
            combined = []
            for i, c in enumerate(nonzeros):
                col_select = c[np.array([c.str.contains(j) for j in c_names]).sum(0).astype(bool)]
                combined.append(X_dense.iloc[i][col_select].prod())
            # combined = np.array([X_dense.iloc[i][c].prod() for i, c in enumerate(nonzeros)])
            X_dense[combined_c_names] = combined
        return X_dense

    def fit_selected(self, trials):
        X_train_selected = self.get_new_X(trials=trials, train=True)
        X_test_selected = self.get_new_X(trials=trials, train=False)
        performance_selected = self.fit(
            X_train_selected, X_test_selected, self.data['y_train'], self.data['y_test']
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
