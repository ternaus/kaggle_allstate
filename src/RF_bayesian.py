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
from sklearn.ensemble import RandomForestRegressor

RANDOM_STATE = 2016
n_folds = 5


def evalerror(preds, dtrain):
    return mean_absolute_error(np.exp(preds), np.exp(dtrain))


def rf_evaluate(max_depth, max_features):#, min_samples_leaf):
    rf = RandomForestRegressor(n_jobs=-1,
                              max_depth=round(max_depth),
                              max_features=max_features,
                               # min_samples_leaf=min_samples_leaf
                               # min_samples_split=round(min_samples_split)
                               )

    kf = KFold(n_folds, shuffle=True, random_state=RANDOM_STATE).get_n_splits(X_train)

    return -cross_val_score(rf, X_train, y_train, cv=kf, scoring=make_scorer(evalerror)).mean()


if __name__ == '__main__':
    num_rounds = 1000
    random_state = 2016
    num_iter = 500
    init_points = 10
    shift = 0

    X_train, y_train, _, _ = clean_data.label_encode(shift=200)
    print X_train.shape, y_train.shape

    # previous_points = pd.read_csv('params/parameters.csv')

    rfBO = BayesianOptimization(rf_evaluate, {'max_depth': (3, 30),
                                                'max_features': (0.1, 1.0),
                                              # 'min_samples_leaf': (0.01, 0.5)
                                                # 'min_samples_split': (2, 10)
                                                })

    # xgbBO.initialize_df(previous_points)

    rfBO.maximize(init_points=init_points, n_iter=num_iter)

    # Save results
    param_cache_path = 'params'
    try:
        os.mkdir(param_cache_path)
    except:
        pass

    file_name = 'params/rf_parameters.csv'
    rfBO.points_to_csv(file_name)
