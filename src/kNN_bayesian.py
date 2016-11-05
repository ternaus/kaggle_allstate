from __future__ import division
"""
Baysian hyperparameter optimization [https://github.com/fmfn/BayesianOptimization]
for Mean Absoulte Error objective
on default features for https://www.kaggle.com/c/allstate-claims-severity
"""

__author__ = "Vladimir Iglovikov"

import pandas as pd
from sklearn.metrics import mean_absolute_error
from bayes_opt import BayesianOptimization
from tqdm import tqdm
import numpy as np
from scipy.stats import skew, boxcox
import os
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from pylab import *
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsRegressor


RANDOM_STATE = 2016
n_folds = 5


def evalerror(preds, dtrain):
    return mean_absolute_error(np.exp(preds), np.exp(dtrain))


def knn_evaluate(n_neighbors, p):
    knn = KNeighborsRegressor(n_jobs=-1,
                              n_neighbors=round(n_neighbors),
                              p=round(p))

    kf = KFold(n_folds, shuffle=True, random_state=RANDOM_STATE).get_n_splits(X_train)

    return -cross_val_score(knn, X_train, y_train, cv=kf, scoring=make_scorer(evalerror)).mean()


def prepare_data():
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')
    test['loss'] = np.nan
    joined = pd.concat([train, test])

    num_columns = [x for x in train.columns if 'cont' in x]
    print len(num_columns)

    for column in num_columns:
        scaler = StandardScaler()
        joined[column] = scaler.fit_transform(joined[column])

    train = joined[joined['loss'].notnull()].reset_index(drop=True)
    test = joined[joined['loss'].isnull()]

    shift = 200

    ids = test['id']
    X_train = train[num_columns].values
    X_test = test[num_columns].values
    y_train = np.log(train['loss'] + shift).values

    return X_train, y_train

if __name__ == '__main__':
    num_rounds = 10000
    random_state = 2016
    num_iter = 50
    init_points = 10
    shift = 0

    X_train, y_train = prepare_data()

    # previous_points = pd.read_csv('params/parameters.csv')

    knnBO = BayesianOptimization(knn_evaluate, {'n_neighbors': (2, 500),
                                                'p': (1, 10)
                                                })

    # xgbBO.initialize_df(previous_points)

    knnBO.maximize(init_points=init_points, n_iter=num_iter)

    # Save results
    param_cache_path = 'params'
    try:
        os.mkdir(param_cache_path)
    except:
        pass

    file_name = 'params/knn_parameters.csv'
    knnBO.points_to_csv(file_name)