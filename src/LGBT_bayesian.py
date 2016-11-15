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
from scipy.sparse import csr_matrix, hstack
from sklearn.neighbors import KNeighborsRegressor
import clean_data
from pylightgbm.models import GBMRegressor

RANDOM_STATE = 2016
n_folds = 5


def evalerror(preds, dtrain):
    return mean_absolute_error(np.exp(preds), np.exp(dtrain))


def lgbt_evaluate(num_leaves, min_data_in_leaf, feature_fraction, bagging_fraction, bagging_freq):
    lgbt = GBMRegressor(
        exec_path=os.path.expanduser('~/packages/LightGBM/lightgbm'),  # Change this to your LighGBM path
        config='',
        application='regression',
        num_iterations=10000,
        learning_rate=0.01,
        num_leaves=int(round(num_leaves)),
        num_threads=8,
        min_data_in_leaf=int(round(min_data_in_leaf)),
        metric='l1',
        feature_fraction=feature_fraction,
        feature_fraction_seed=2016,
        bagging_fraction=bagging_fraction,
        bagging_freq=int(round(bagging_freq)),
        bagging_seed=2016,
        early_stopping_round=50,
        verbose=False
    )

    kf = KFold(n_folds, shuffle=True, random_state=RANDOM_STATE).get_n_splits(X_train)

    return -cross_val_score(lgbt, X_train, y_train, cv=kf, scoring=make_scorer(evalerror)).mean()


if __name__ == '__main__':
    num_rounds = 1000
    random_state = 2016
    num_iter = 500
    init_points = 10
    shift = 200

    # X_train, y_train, _, _ = clean_data.fancy(shift=200)
    X_train, y_train, _, _, _, _ = clean_data.one_hot_categorical(shift)
    print X_train.shape, y_train.shape

    # previous_points = pd.read_csv('params/parameters.csv')

    lgbtBO = BayesianOptimization(lgbt_evaluate, {'num_leaves': (10, 3000),
                                                  'min_data_in_leaf': (1, 300),
                                                  'feature_fraction': (0, 1),
                                                  'bagging_fraction': (0, 1),
                                                  'bagging_freq': (10, 500)
                                                })

    # xgbBO.initialize_df(previous_points)

    lgbtBO.maximize(init_points=init_points, n_iter=num_iter)

    # Save results
    param_cache_path = 'params'
    try:
        os.mkdir(param_cache_path)
    except:
        pass

    # file_name = 'params/rf_parameters.csv'
    # rfBO.points_to_csv(file_name)
