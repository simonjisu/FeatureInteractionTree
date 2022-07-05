import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score, accuracy_score
from .utils import c_statistic_harrell
from .dataset import Dataset

class ModelBuilder():
    def __init__(self):
        # model init
        self.objective_dict = {
            'reg': ('reg:squarederror', r2_score, 'rmse'), 
            'binary': ('binary:logistic', accuracy_score, 'error'),
            'survival': ('survival:cox', c_statistic_harrell, 'rmse')
        }

    def cv(self, xgb_train, params, num_rounds, eval_metric, seed, nfold=3, early_stopping_rounds=10):
        xgb_cv = xgb.cv(
            dtrain=xgb_train, 
            params=params, 
            nfold=nfold,
            num_boost_round=num_rounds, 
            early_stopping_rounds=early_stopping_rounds, 
            metrics=eval_metric, 
            as_pandas=True, 
            seed=seed, 
        )
        best_num_rounds = xgb_cv.iloc[-1].name
        return best_num_rounds

    def train(
            self, 
            dataset: Dataset, 
            eta: float=0.3, 
            max_depth: int=8, 
            subsample: float=1.0, 
            seed: int=8,
            num_rounds: int=300
        ):
        objective, metric_fn, eval_metric = self.objective_dict[dataset.task_type]
        params = {
            'eta': eta,
            'max_depth': max_depth,
            'objective': objective,
            'subsample': subsample,
            'seed': seed
        }
        xgb_train = xgb.DMatrix(
            dataset.data['X_train'], 
            label=dataset.data['y_train'],
        )
        xgb_test = xgb.DMatrix(
            dataset.data['X_test'], 
            label=dataset.data['y_test'],
        )
        self.best_num_rounds = self.cv(xgb_train, params, num_rounds, eval_metric, seed)
        model = xgb.train(
            params, xgb_train, self.best_num_rounds, 
            evals=[(xgb_test, "test")], 
            verbose_eval=int(self.best_num_rounds * 0.2)
        )
        y_pred = model.predict(xgb_test)
        if dataset.task_type == 'binary':
            y_pred = (y_pred >= 0.5).astype(np.uint8)
        score = metric_fn(y_true=dataset.data['y_test'], y_pred=y_pred)

        return {
            'model': model,
            'score': score
        }

