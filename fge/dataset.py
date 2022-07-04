import numpy as np
import pandas as pd
from pathlib import Path
from shap.datasets import adult, boston, nhanesi, communitiesandcrime
from sklearn.datasets import fetch_california_housing, fetch_openml
from sklearn.model_selection import train_test_split
from collections import defaultdict
from sklearn.datasets import fetch_openml

class Dataset():
    def __init__(self, dataset_name, data_folder, seed):
        self.loaders = {
            'adult': (adult, 'binary'), 
            'boston': (boston, 'reg'), 
            'titanic': (titanic, 'binary'),
            'nhanesi': (nhanesi, 'survival'), # TODO: need to dealwith NaN values to fit poly
            'crime': (communitiesandcrime, 'reg'),  # TODO: need preprocessing y -> proportion of population
            'california': (california, 'reg'),
            'ames': (ames_house_prices, 'reg'),
            'lending_club': ()
        }
        self.loader, self.task_type = self.loaders[dataset_name]
        self.seed = seed
        self.data_path = Path(data_folder).resolve()
        if dataset_name in ['california', 'titanic', 'ames']:
            ds = self.loader(data_home=self.data_path / dataset_name, as_frame=True, display=False)
            X, y = ds['data'], ds['target'].to_numpy().reshape(-1)
        else:
            X, y = self.loader()
        X = self._preprocess_X(dataset_name, X, y)
        y = self._preprocess_y(y)
        self._generate_cates(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=seed)

        self.feature_names = X.columns
        # original data
        self.data = {
            'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test,    
        }
        self.groups = {
            'train': self._generate_groups(X_train, y_train),
            'test': self._generate_groups(X_test, y_test)
        }

    def _preprocess_X(self, dataset_name, X, y):
        if dataset_name == 'titanic':
            cate_cols = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Title']
            for c in cate_cols:
                X[c] = X[c].astype(np.int32)  # pd.Categorical(X[c]) for instance since SHAP not support Categorical yet
        elif dataset_name == 'nhanesi':
            X = X.drop(columns=X.columns[X.isnull().sum() > 0])
        return X

    def _preprocess_y(self, y):
        if self.task_type == 'binary':
            y = y.astype(np.uint8)
        else:
            y = y.astype(np.float64)
        return y

    def _generate_cates(self, y):
        if self.task_type in ('reg', 'survival'):
            y_cates = pd.qcut(y, q=4, duplicates='drop')
        else:
            y_cates = pd.Categorical(y)
        self.cates = dict(enumerate(y_cates.categories))

    def _generate_groups_idx(self, y):
        groups_idx = {}
        for c_idx, c in self.cates.items():
            cond = y == c if self.task_type == 'binary' else (c.left < y) & (y <= c.right)
            groups_idx[c_idx] = np.arange(y.shape[0])[cond]
        return groups_idx
    
    def _generate_groups(self, X, y):
        groups_idx = self._generate_groups_idx(y)
        groups = defaultdict(dict)
        is_pd = lambda z: isinstance(z, pd.DataFrame) or isinstance(z, pd.Series)
        for c_idx, idx in groups_idx.items():
            groups[c_idx]['X'] = X.iloc[idx, :] if is_pd(X) else X[idx, :]
            groups[c_idx]['y'] = y.iloc[idx] if is_pd(y) else y[idx]
        return groups

    def __getitem__(self, i):
        train_data = self.groups['train']
        test_data = self.groups['test']
        return {
            'X_train': train_data[i]['X'], 
            'X_test': test_data[i]['X'], 
            'y_train': train_data[i]['y'], 
            'y_test': test_data[i]['y']
        }

def california(data_home, as_frame=False, display=False):
    dataset = fetch_california_housing(data_home=data_home, as_frame=as_frame)
    X, y = california_preprocess(dataset, display=display)
    if not as_frame:
        X, y = X.to_numpy(), y.to_numpy().reshape(-1)
    return {'data': X, 'target': y}

def california_preprocess(dataset, display=False):
    X = dataset['data'].copy()
    y = dataset['target']
    X['Latitude'] = pd.qcut(X['Latitude'], 5)
    X['Longitude'] = pd.qcut(X['Longitude'], 5)
    if not display:
        X['Latitude'] = X['Latitude'].cat.codes
        X['Longitude'] = X['Longitude'].cat.codes
    return X, y

def ames_house_prices(data_home, as_frame=False, display=False):
    dataset = fetch_openml(name='house_prices', data_home=data_home, as_frame=as_frame)
    X, y = ames_preprocess(dataset, display=display)
    if not as_frame:
        X, y = X.to_numpy(), y.to_numpy().reshape(-1)
    return {'data': X, 'target': y}

def ames_preprocess(dataset, display=False):
    X = dataset['data'].copy()
    y = dataset['target'] / 1000
    basement_cols = ['BsmtFinType2', 'BsmtExposure', 'BsmtQual', 'BsmtFinType1', 'BsmtCond']  # missing basement should be same number
    cond1 = X.loc[:, ['BsmtFinType2', 'BsmtExposure']].isnull().sum(1) > 0
    cond2 = X.loc[:, ['BsmtQual', 'BsmtFinType1', 'BsmtCond']].isnull().sum(1) == 0
    error_idx = X.loc[cond1 & cond2, basement_cols].index
    X.drop(index=error_idx, inplace=True)
    y.drop(index=error_idx, inplace=True)
    error_idx2 = X['Electrical'].loc[X['Electrical'].isnull()].index
    X.drop(index=error_idx2, inplace=True)
    y.drop(index=error_idx2, inplace=True)
    # Missing = NA
    na_cols = [
        'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageQual', 'GarageCond', 'GarageFinish', 
        'BsmtFinType2', 'BsmtExposure', 'BsmtQual', 'BsmtFinType1', 'BsmtCond'
    ]
    X.loc[:, na_cols] = X.loc[:, na_cols].fillna('NA').values
    X.loc[:, ['MasVnrArea', 'MasVnrType']] = X.loc[:, ['MasVnrArea', 'MasVnrType']].fillna(0).values  # if none should fill 0
    X.loc[X['GarageYrBlt'].isnull(), 'GarageYrBlt'] = X.loc[X['GarageYrBlt'].isnull(), 'YearBuilt'].values # fill it as YearBuilt
    X.loc[:, 'LotFrontage'] = X['LotFrontage'].fillna(X['LotFrontage'].mean())
    X.drop(columns=['Id'], inplace=True)
    X.reset_index(drop=True, inplace=True)
    # Data type correction
    c_int = [
        'MSSubClass', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
        'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'MoSold', 'YrSold'
    ]
    c_float = [
        'LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
        '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
        'ScreenPorch', 'PoolArea', 'MiscVal'
    ]
    c_cates = [
        'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
        'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 
        'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 
        'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 
        'Electrical', 'KitchenQual', 'Functional',  'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 
        'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition'
    ]
    X.loc[:, c_int] = X.loc[:, c_int].astype(np.int64)
    X.loc[:, c_float] = X.loc[:, c_float].astype(np.float64)
    for c in c_cates:
        cates = ['NA'] + list(X[c].unique()[X[c].unique() != 'NA']) if 'NA' in X[c].unique() else list(X[c].unique())
        X.loc[:, c] = pd.Categorical(X[c], categories=cates)
        if not display:
            X.loc[:, c] = X[c].cat.codes
    return X, y

def titanic(data_home, as_frame=False, display=False):
    df_train = pd.read_csv(Path(data_home).resolve() / 'train.csv')
    df_test = pd.read_csv(Path(data_home).resolve() / 'test.csv')
    dataset = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
    X, y = titanic_preprocess(dataset, display=display)
    if not as_frame:
        X, y = X.to_numpy(), y.to_numpy().reshape(-1)
    return {'data': X, 'target': y}

def titanic_preprocess(dataset, display=False):
    # some of preprocess code borrowed from https://www.kaggle.com/code/startupsci/titanic-data-science-solutions

    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    dataset['Title'] = dataset['Title'].replace(
        ['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    dataset.drop(columns=['Name', 'PassengerId', 'Ticket', 'Cabin'], inplace=True)
    nonnull_idx = dataset.isnull().sum(1) == 0 
    dataset = dataset.loc[nonnull_idx].reset_index(drop=True)
    cate_cols = ['Pclass', 'Sex', 'Embarked', 'Title']
    for c in cate_cols:
        dataset[c] = pd.Categorical(dataset[c])
        if not display:
            dataset[c] = dataset[c].cat.codes
    X, y = dataset.iloc[:, 1:], dataset.iloc[:, 0]
    return X, y