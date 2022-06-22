from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from .utils import flatten, c_statistic_harrell

class PolyFitter():
    def __init__(self, dataset, task_type, original_score, max_iter=200):
        
        self.task_dict = {
            'reg': (LinearRegression, r2_score, dict()),
            'binary': (LogisticRegression, accuracy_score, dict(max_iter=max_iter)),
            'survival': (LinearRegression, c_statistic_harrell, dict())
        }
        
        self.task_model, self.task_metric, self.args = self.task_dict[task_type]
        self.dataset = dataset
        
        if original_score is None:
            self.original_score = self.fit_full()
        else:
            self.original_score = original_score

    def get_new_X(self, X_original, nodes, feature_names):
        X = X_original.copy()
        for feature in nodes.keys():
            if isinstance(feature, int):
                continue
            feature_name = '+'.join([str(feature_names[i]) for i in flatten(feature)]) 
            X[feature_name] = X.iloc[:, flatten(feature)].prod(1)
        return X

    def get_selected_X(self, nodes, feature_names):
        X_train, X_test = self.dataset['X_train'].copy(), self.dataset['X_test'].copy()
        new_X_train = self.get_new_X(X_train, nodes, feature_names)
        new_X_test = self.get_new_X(X_test, nodes, feature_names)
        return new_X_train, new_X_test

    def fit_poly(self, X_train, X_test, y_train, y_test):
        poly_model = make_pipeline(StandardScaler(), self.task_model(**self.args))
        poly_model.fit(X_train, y_train)
        y_pred = poly_model.predict(X_test)
        return self.task_metric(y_test, y_pred)

    def fit_full(self):
        X_train, X_test = self.dataset['X_train'].copy(), self.dataset['X_test'].copy()
        poly = PolynomialFeatures(X_train.shape[1], interaction_only=True, include_bias=False)
        X_train_full, X_test_full = poly.fit_transform(X_train), poly.fit_transform(X_test)
        performance_full = self.fit_poly(
            X_train_full, X_test_full, self.dataset['y_train'], self.dataset['y_test']
        )
        return performance_full
        
    def fit_selected(self, nodes, feature_names):
        X_train_selected, X_test_selected = self.get_selected_X(nodes, feature_names)
        performance_selected = self.fit_poly(
            X_train_selected, X_test_selected, self.dataset['y_train'], self.dataset['y_test']
        )
        return performance_selected