"""
Baysian hyperparameter optimization [https://github.com/fmfn/BayesianOptimization]
for Mean Absoulte Error objective
on default features for https://www.kaggle.com/c/allstate-claims-severity
"""

__author__ = "Vladimir Iglovikov"

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from bayes_opt import BayesianOptimization
from tqdm import tqdm
import numpy as np
from scipy.stats import skew, boxcox
import os
import xgboost as xgb


def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(preds), np.exp(labels))


def xgb_evaluate(min_child_weight,
                 colsample_bytree,
                 max_depth,
                 subsample,
                 gamma,
                 alpha):

    params['min_child_weight'] = int(min_child_weight)
    params['cosample_bytree'] = max(min(colsample_bytree, 1), 0)
    params['max_depth'] = int(max_depth)
    params['subsample'] = max(min(subsample, 1), 0)
    params['gamma'] = max(gamma, 0)
    params['alpha'] = max(alpha, 0)

    cv_result = xgb.cv(params, xgtrain, num_boost_round=num_rounds, nfold=5,
                         seed=random_state,
                         feval=evalerror,
                         callbacks=[xgb.callback.early_stop(50)])

    return -cv_result['test-mae-mean'].values[-1]


def prepare_data():
    train = pd.read_csv('../data/train.csv')

    y = np.log(train['loss'] + shift)

    print train.info()
    numerical_feats = train.dtypes[train.dtypes != "object"].index
    # compute skew and do Box-Cox transformation
    skewed_feats = train[numerical_feats].apply(lambda x: skew(x.dropna()))
    print 'Skew in numeric features:'
    print skewed_feats
    # transform features with skew > 0.25 (this can be varied to find optimal value)
    skewed_feats = skewed_feats[skewed_feats > 0.25]
    skewed_feats = skewed_feats.index
    for feats in tqdm(skewed_feats):
        train[feats] += 1
        train[feats], lam = boxcox(train[feats])

    categorical_columns = train.select_dtypes(include=['object']).columns

    for column in tqdm(categorical_columns):
        le = LabelEncoder()
        train[column] = le.fit_transform(train[column])

    X = train.drop(['loss', 'id'], 1)
    xgtrain = xgb.DMatrix(X, label=y)

    return xgtrain


if __name__ == '__main__':
    num_rounds = 10000
    random_state = 2016
    num_iter = 100
    init_points = 100
    shift = 0

    xgtrain = prepare_data()

    params = {
        'eta': 0.1,
        'silent': 1,
        'eval_metric': 'mae',
        'verbose_eval': True,
        'seed': random_state
    }

    previous_points = pd.read_csv('params/parameters.csv')

    xgbBO = BayesianOptimization(xgb_evaluate, {'min_child_weight': (1, 5),
                                                'colsample_bytree': (0.3, 0.6),
                                                'max_depth': (10, 15),
                                                'subsample': (0.7, 1),
                                                'gamma': (0, 3),
                                                'alpha': (0, 3),
                                                })

    # xgbBO.initialize_df(previous_points)

    xgbBO.maximize(init_points=init_points, n_iter=num_iter)

    # Save results
    param_cache_path = 'params'
    try:
        os.mkdir(param_cache_path)
    except:
        pass

    file_name = str(num_rounds) + '_' + str(params['eta']) + '.csv'
    xgbBO.points_to_csv(file_name)
