from sklearn.datasets import fetch_california_housing

import sys
import xgboost as xgb
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

main_path = Path().absolute().parent
sys.path.append(str(main_path))

def build_model(seed=8, eta=0.3, max_depth=8, subsample=1.0, num_rounds=200):
    """
    - MedInc median income in block group
    - HouseAge median house age in block group
    - AveRooms average number of rooms per household
    - AveBedrms average number of bedrooms per household
    - Population block group population
    - AveOccup average number of household members
    - Latitude block group latitude
    - Longitude block group longitude
    """
    ca_housing = fetch_california_housing(data_home=main_path / 'data' / 'housing', as_frame=True)
    X, y = ca_housing['data'], ca_housing['target']
    # create a train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=seed)
    xgb_train = xgb.DMatrix(X_train, label=y_train)
    xgb_test = xgb.DMatrix(X_test, label=y_test)

    params = {
        'eta': eta,
        'max_depth': max_depth,
        'objective': 'reg:squarederror',
        'subsample': subsample,
        'seed': seed
    }
    model = xgb.train(params, xgb_train, num_rounds, evals=[(xgb_test, "test")], verbose_eval=int(num_rounds / 10))

    y_pred = model.predict(xgb_test)
    print(f'# of train data: {xgb_train.num_row()}')
    print(f'# of test data: {xgb_test.num_row()}')
    print('R2 square:', r2_score(y_true=y_test, y_pred=y_pred))

    return {
        'model': model,
        'train': (X_train, y_train), 
        'test': (X_test, y_test),
        'origin': ca_housing
    }