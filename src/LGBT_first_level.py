from __future__ import division
import pandas as pd

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold
import clean_data
from pylab import *
from sklearn.utils import shuffle
from pylightgbm.models import GBMRegressor
import os

shift = 200
X_train, y_train, X_test, y_mean, test_ids, train_ids = clean_data.fancy(shift=shift)

X_train = X_train.values
X_test = X_test.values
y_train = y_train.values

num_rounds = 300000
RANDOM_STATE = 2016

n_folds = 10
num_train = X_train.shape
num_test = X_test.shape[0]

kf = StratifiedKFold(n_folds, shuffle=True, random_state=RANDOM_STATE)

classes = clean_data.classes(y_train, bins=100)


class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None):
        # params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)


nbags = 10


def get_oof():
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
            lgbt_params = {
                'exec_path': os.path.expanduser('~/packages/LightGBM/lightgbm'),  # Change this to your LighGBM path
                'config': '',
                'application': 'regression',
                'num_iterations': 3000,
                'learning_rate': 0.01,
                'num_leaves': 202,
                'num_threads': 8,
                'min_data_in_leaf': 9,
                'metric': 'l1',
                'feature_fraction': 0.3149,
                'feature_fraction_seed': 2016 + i + j,
                'bagging_fraction': 1,
                'bagging_freq': 100,
                'bagging_seed': 2016 + i + j,
                'early_stopping_round': 25,
                # metric_freq=1,
                'verbose': False
            }
            clf = GBMRegressor(**lgbt_params)
            clf.fit(x_tr, y_tr)

            pred += np.exp(clf.predict(x_te))
            pred_test += np.exp(clf.predict(X_test))

        pred /= nbags
        pred_oob[test_index] = pred
        score = mean_absolute_error(np.exp(y_te), pred)
        print('Fold ', i, '- MAE:', score)

    return pred_oob, pred_test

xg_oof_train, xg_oof_test = get_oof()

print("lgbt-CV: {}".format(mean_absolute_error(np.exp(y_train), xg_oof_train)))

oof_train = pd.DataFrame({'id': train_ids, 'loss': (xg_oof_train - shift)})
oof_train.to_csv('oof/lgbt_train_1.csv', index=False)

xg_oof_test /= (n_folds * nbags)

oof_test = pd.DataFrame({'id': test_ids, 'loss': (xg_oof_test - shift)})
oof_test.to_csv('oof/lgbt_test_1.csv', index=False)
