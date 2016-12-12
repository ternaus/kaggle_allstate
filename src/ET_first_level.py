from __future__ import division
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
import clean_data
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
from pylab import *

shift = 200
X_train, y_train, X_test, y_mean, test_ids, train_ids = clean_data.fancy(shift=shift)

X_train = X_train.values
X_test = X_test.values
y_train = y_train.values

num_rounds = 300000
RANDOM_STATE = 2016

n_folds = 10
num_train = X_train.shape[0]
num_test = X_test.shape[0]

classes = clean_data.classes(y_train, bins=100)
kf = StratifiedKFold(n_folds, shuffle=True, random_state=RANDOM_STATE)

et_params = {
    'n_jobs': -1,
    'n_estimators': 1000,
    'max_features': 0.5,
    'max_depth': 28,
    'min_samples_leaf': 2,
}


class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)


def get_oof(clf):
    oof_train = np.zeros((num_train,))
    oof_test = np.zeros((num_test,))
    oof_test_skf = np.empty((n_folds, num_test))

    for i, (train_index, test_index) in enumerate(kf.split(classes, classes)):
        print "Fold = ", i
        x_tr = X_train[train_index]
        y_tr = y_train[train_index]
        x_te = X_train[test_index]
        y_te = y_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = np.exp(clf.predict(x_te))
        oof_test_skf[i, :] = np.exp(clf.predict(X_test))
        print mean_absolute_error(np.exp(y_te), oof_train[test_index])
        print
    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


et = SklearnWrapper(clf=ExtraTreesRegressor, seed=RANDOM_STATE, params=et_params)
xg_oof_train, xg_oof_test = get_oof(et)


print("et-CV: {}".format(mean_absolute_error(np.exp(y_train), xg_oof_train)))

oof_train = pd.DataFrame({'id': train_ids, 'loss': (xg_oof_train - shift)[:, 0]})
oof_train.to_csv('oof/et_train_2.csv', index=False)

oof_test = pd.DataFrame({'id': test_ids, 'loss': (xg_oof_test - shift)[:, 0]})
oof_test.to_csv('oof/et_test_2.csv', index=False)

