import pandas as pd

import sys

sys.path += ['/home/vladimir/packages/xgboost/python-package']
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from pylab import *

from sklearn.utils import shuffle
import clean_data
import lightgbm as lgb


def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    con = 2
    x = preds - labels
    den = np.abs(x) + con
    grad = con * x / den
    hess = (con / den)**2
    return grad, hess


def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'mae', mean_absolute_error(preds**4, labels**4), False


X_train, y_train, X_test, y_mean, X_test_id, X_train_id = clean_data.fancy_sqrt(quadratic=True,
                                                                                add_aggregates=False)
print list(X_train.columns)
X_train = X_train.values
X_test = X_test.values

print X_train.shape, X_test.shape

RANDOM_STATE = 2016

lgbm_params = arams = {
    'boosting_type': 'gbdt',
    'num_leaves': 200,
    'learning_rate': 0.01,
    'feature_fraction': 0.4,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'verbose': 0
}

n_folds = 10
num_train = len(y_train)

kf = StratifiedKFold(n_folds, shuffle=True, random_state=RANDOM_STATE)

classes = clean_data.classes(y_train, bins=200)


class lgbmWrapper(object):
    def __init__(self, seed=2016, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 250)

    def train(self, X_train, y_train, seed, X_val, y_val):
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

        self.param['seed'] = seed

        self.model = lgb.train(lgbm_params,
                               lgb_train,
                               fobj=logregobj,
                               feval=evalerror,
                               num_boost_round=30000,
                               valid_sets=lgb_val,  # eval training data
                               early_stopping_rounds=100
                               )

    def predict(self, x):
        return self.model.predict(x, num_iteration=self.model.best_iteration)


nbags = 2


def get_oof(clf):
    pred_oob = np.zeros(X_train.shape[0])
    pred_test = np.zeros(X_test.shape[0])

    for i, (train_index, test_index) in enumerate(kf.split(classes, classes)):
        print "Fold = ", i
        x_tr = X_train[train_index]
        y_tr = y_train[train_index]

        x_te = X_train[test_index]
        y_te = y_train[test_index]

        pred = np.zeros(x_te.shape[0])

        for j in range(nbags):
            x_tr, y_tr = shuffle(x_tr, y_tr, random_state=RANDOM_STATE + i + j)
            clf.train(x_tr, y_tr, RANDOM_STATE + i, x_te, y_te)

            pred += clf.predict(x_te)**4
            pred_test += clf.predict(X_test)**4

        pred /= nbags
        pred_oob[test_index] = pred
        score = mean_absolute_error(y_te**4, pred)
        print('Fold ', i, '- MAE:', score)

    return pred_oob, pred_test


xg = lgbmWrapper(seed=RANDOM_STATE, params=lgbm_params)
xg_oof_train, xg_oof_test = get_oof(xg)

print("lgbm-CV: {}".format(mean_absolute_error(y_train**4, xg_oof_train)))

oof_train = pd.DataFrame({'id': X_train_id, 'loss': xg_oof_train})
oof_train.to_csv('oof/lgbm_train_s1.csv', index=False)

xg_oof_test /= (n_folds * nbags)

oof_test = pd.DataFrame({'id': X_test_id, 'loss': xg_oof_test})
oof_test.to_csv('oof/lgbm_test_s1.csv', index=False)

