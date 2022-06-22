import numpy as np
import xgboost as xgb
from pathlib import Path
from shap.datasets import adult, boston, nhanesi, communitiesandcrime
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from .utils import c_statistic_harrell

class ModelBuilder():
    def __init__(
            self, 
            dataset_name: str, 
            data_folder: str, 
            eta: float=0.3, 
            max_depth: int=8, 
            subsample: float=1.0, 
            seed: int=8
        ):
        self.dataset_name = dataset_name
        self.loaders = {
            'adult': (adult, 'binary'), 
            'boston': (boston, 'reg'), 
            'nhanesi': (nhanesi, 'survival'), # TODO: need to dealwith NaN values to fit poly
            'crime': (communitiesandcrime, 'reg'),  # TODO: need preprocessing y -> proportion of population
            'california': (fetch_california_housing, 'reg'),
            'lending_club': ()
        }
        self.loader, self.task_type = self.loaders[dataset_name]

        self.data_path = Path(data_folder).resolve()
        if dataset_name == 'california':
            ds = self.loader(data_home=self.data_path / 'california', as_frame=True)
            X, y = ds['data'], ds['target']
            self.X_encoded, self.y_encoded = X.copy(), y.copy()
            self.X_display, self.y_display = X.copy(), y.copy()
        else:
            X, y = self.loader()
            self.X_encoded, self.y_encoded = X, y
            self.X_display, self.y_display = self.loader(display=True)

        # create a train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=seed)
        if self.task_type == 'binary':
            y_train = y_train.astype(np.uint8)
            y_test = y_test.astype(np.uint8)
        else:
            y_train = y_train.astype(np.float32)
            y_test = y_test.astype(np.float32)
        self.dataset = {
            'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test
        }
        # model init
        self.objective_dict = {
            'reg': ('reg:squarederror', r2_score), 
            'binary': ('binary:logistic', accuracy_score),
            'survival': ('survival:cox', c_statistic_harrell)
        }
        objective, self.metric = self.objective_dict[self.task_type]
        self.params = {
            'eta': eta,
            'max_depth': max_depth,
            'objective': objective,
            'subsample': subsample,
            'seed': seed
        }

    def train(self, num_rounds=1000):
        xgb_train = xgb.DMatrix(self.dataset['X_train'], label=self.dataset['y_train'])
        xgb_test = xgb.DMatrix(self.dataset['X_test'], label=self.dataset['y_test'])

        model = xgb.train(
            self.params, xgb_train, num_rounds, 
            evals=[(xgb_test, "test")], 
            verbose_eval=int(num_rounds * 0.2)
        )
        y_pred = model.predict(xgb_test)
        if self.task_type == 'binary':
            y_pred = (y_pred >= 0.5).astype(np.uint8)

        return {
            'dataset_name': self.dataset_name,
            'task_type': self.task_type,
            'model': model,
            'dataset': self.dataset, 
            'score': self.metric(y_true=self.dataset['y_test'], y_pred=y_pred)
        }

