import pandas as pd

import sys

sys.path += ['/home/vladimir/packages/xgboost/python-package']
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from pylab import *
from tqdm import tqdm
from sklearn.utils import shuffle
import clean_data

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
test['loss'] = np.nan
joined = pd.concat([train, test])


def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    con = 0.7
    x = preds - labels
    den = np.abs(x) + con
    grad = con * x / den
    hess = (con / den)**2
    return grad, hess


def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'mae', mean_absolute_error(preds**4, labels**4)


X_train, y_train, X_test, y_mean, X_test_id, X_train_id = clean_data.fancy_sqrt(quadratic=True)

X_train = X_train.values
X_test = X_test.values

print X_train.shape, X_test.shape

RANDOM_STATE = 2016
xgb_params = {
    'min_child_weight': 100,
    'eta': 0.01,
    'colsample_bytree': 0.7,
    'max_depth': 12,
    'subsample': 0.7,
    # 'alpha': 5,
    # 'lambda': 5,
    # 'gamma': 1,
    'silent': 1,
    # 'base_score': y_mean,
    'verbose_eval': 1,
    'seed': RANDOM_STATE,
    'nrounds': 30000
}

n_folds = 10
num_train = len(y_train)
num_test = test.shape[0]

kf = StratifiedKFold(n_folds, shuffle=True, random_state=RANDOM_STATE)

classes = clean_data.classes(y_train, bins=100)


class XgbWrapper(object):
    def __init__(self, seed=2016, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 250)

    def train(self, x_train, y_train, seed, x_val, y_val):
        dtrain = xgb.DMatrix(x_train, label=y_train)
        self.param['seed'] = seed
        dval = xgb.DMatrix(x_val, label=y_val)
        watchlist = [(dtrain, 'train'), (dval, 'val')]
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds, watchlist, obj=logregobj, feval=evalerror, early_stopping_rounds=50)

    def predict(self, x):
        return self.gbdt.predict(xgb.DMatrix(x))


nbags = 1


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


xg = XgbWrapper(seed=RANDOM_STATE, params=xgb_params)
xg_oof_train, xg_oof_test = get_oof(xg)

print("XG-CV: {}".format(mean_absolute_error(y_train**4, xg_oof_train)))

oof_train = pd.DataFrame({'id': X_train_id, 'loss': xg_oof_train})
oof_train.to_csv('oof/xgb_train_s6.csv', index=False)

xg_oof_test /= (n_folds * nbags)

oof_test = pd.DataFrame({'id': X_test_id, 'loss': xg_oof_test})
oof_test.to_csv('oof/xgb_test_s6.csv', index=False)

