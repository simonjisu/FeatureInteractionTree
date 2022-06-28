import numpy as np
import pandas as pd
from pathlib import Path
from shap.datasets import adult, boston, nhanesi, communitiesandcrime
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from collections import defaultdict

class Dataset():
    def __init__(self, dataset_name, data_folder, seed):
        self.loaders = {
            'adult': (adult, 'binary'), 
            'boston': (boston, 'reg'), 
            'titanic': (titanic, 'binary'),
            'nhanesi': (nhanesi, 'survival'), # TODO: need to dealwith NaN values to fit poly
            'crime': (communitiesandcrime, 'reg'),  # TODO: need preprocessing y -> proportion of population
            'california': (fetch_california_housing, 'reg'),
            'lending_club': ()
        }
        self.loader, self.task_type = self.loaders[dataset_name]
        self.seed = seed
        self.data_path = Path(data_folder).resolve()
        if dataset_name in ['california', 'titanic']:
            ds = self.loader(data_home=self.data_path / dataset_name, as_frame=True)
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
        if dataset_name == 'nhanesi':
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


def titanic(data_home, as_frame=True):
    # some of preprocess code borrowed from https://www.kaggle.com/code/startupsci/titanic-data-science-solutions
    def preprocess(dataset, display=False):
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
        
    df_train = pd.read_csv(Path(data_home).resolve() / 'train.csv')
    df_test = pd.read_csv(Path(data_home).resolve() / 'test.csv')
    dataset = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
    X, y = preprocess(dataset, display=False)
    return {'data': X, 'target': y}