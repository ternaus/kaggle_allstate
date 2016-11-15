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

exec_path = os.path.expanduser("~/packages/LightGBM/lightgbm")


train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
test['loss'] = np.nan
joined = pd.concat([train, test])

#
# def logregobj(preds, dtrain):
#     labels = dtrain.get_label()
#     con = 2
#     x = preds - labels
#     grad = con * x / (np.abs(x) + con)
#     hess = con**2 / (np.abs(x) + con)**2
#     return grad, hess
#
#
# def evalerror(preds, dtrain):
#     labels = dtrain.get_label()
#     return 'mae', mean_absolute_error(np.exp(preds), np.exp(labels))


for column in list(train.select_dtypes(include=['object']).columns):
    # g = train.groupby(column)['loss'].mean()
    # g = train.groupby(column)['loss'].median()

    if train[column].nunique() != test[column].nunique():
        # Let's find extra categories...
        set_train = set(train[column].unique())
        set_test = set(test[column].unique())
        remove_train = set_train - set_test
        remove_test = set_test - set_train

        remove = remove_train.union(remove_test)
        print column, remove

        def filter_cat(x):
            if x in remove:
                return np.nan
            return x

        joined[column] = joined[column].apply(lambda x: filter_cat(x), 1)
        print 'unique =', joined[column].nunique()

    joined[column] = pd.factorize(joined[column].values, sort=True)[0]

train = joined[joined['loss'].notnull()]
test = joined[joined['loss'].isnull()]

shift = 200
y_train = np.log(train['loss'] + shift)
ids = test['id']
X_train = train.drop(['loss', 'id'], 1)
X_test = test.drop(['loss', 'id'], 1)



def evalerror(preds, dtrain):
    return mean_absolute_error(np.exp(preds), np.exp(dtrain))


def lgm_evaluate(num_leaves, min_data_in_leaf, feature_fraction, bagging_fraction, bagging_freq):
    lgm = GBMRegressor(exec_path=exec_path,
                        application='regression',
                        num_iterations=10000,
                        tree_learner='serial',
                        early_stopping_round=50,
                        learning_rate=0.01,
                        num_leaves=round(num_leaves),
                        min_data_in_leaf=round(min_data_in_leaf),
                        feature_fraction=max(feature_fraction, 0),
                        bagging_fraction=max(bagging_fraction, 0),
                        bagging_freq=round(bagging_freq),
                        metric='l2',
                        bagging_seed=RANDOM_STATE,
                        metric_freq=1,
                        verbose=False
                    )


    kf = KFold(n_folds, shuffle=True, random_state=RANDOM_STATE).get_n_splits(X_train)

    return -cross_val_score(lgm, X_train, y_train, cv=kf, scoring=make_scorer(evalerror)).mean()


if __name__ == '__main__':
    num_rounds = 1000
    random_state = 2016
    num_iter = 500
    init_points = 10
    shift = 0
    #
    # X_train, y_train, _, _ = clean_data.label_encode(shift=200)
    # print X_train.shape, y_train.shape

    # previous_points = pd.read_csv('params/parameters.csv')

    lgmBO = BayesianOptimization(lgm_evaluate, {'num_leaves': (15, 500),
                                                'min_data_in_leaf': (15, 200),
                                                'feature_fraction': (0.3, 1),
                                                'bagging_fraction': (0.3, 1),
                                                'bagging_freq': (1, 20),
                                                })

    # xgbBO.initialize_df(previous_points)

    lgmBO.maximize(init_points=init_points, n_iter=num_iter)

    # Save results
    param_cache_path = 'params'
    try:
        os.mkdir(param_cache_path)
    except:
        pass

    file_name = 'params/lgm_parameters.csv'
    lgmBO.points_to_csv(file_name)
