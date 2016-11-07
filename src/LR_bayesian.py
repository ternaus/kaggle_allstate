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


if __name__ == '__main__':
    num_rounds = 10000
    random_state = 2016
    num_iter = 50
    init_points = 10
    shift = 0

    X_train, y_train, _, _ = clean_data.oof_categorical(scale=True)
    print X_train.shape, y_train.shape

    # previous_points = pd.read_csv('params/parameters.csv')

    knnBO = BayesianOptimization(knn_evaluate, {'n_neighbors': (2, 10),
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
