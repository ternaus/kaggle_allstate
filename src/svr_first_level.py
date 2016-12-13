from __future__ import division
import pandas as pd
import clean_data

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from pylab import *
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVR, SVR

from sklearn.metrics import make_scorer


def eval_f(x, y):
    return mean_absolute_error(x**4, y**4)


X_train, y_train, X_test, y_mean, test_ids, train_ids = clean_data.one_hot_categorical_sqrt(quadratic=True)
num_rounds = 300000
RANDOM_STATE = 2016

n_folds = 10
num_train = len(y_train)
num_test = X_test.shape[0]

kf = StratifiedKFold(n_folds, shuffle=True, random_state=RANDOM_STATE)
classes = clean_data.classes(y_train, bins=100)

# svr_params = {'C': np.logspace(-3, -1, 10)}
#
# lr = LinearSVR()
# clf = GridSearchCV(lr, param_grid=svr_params, n_jobs=-1, cv=kf.get_n_splits(classes, classes), scoring=make_scorer(eval_f), iid=False, verbose=2)
#
# clf.fit(X_train, y_train)
#
# for i in clf.grid_scores_:
#     print i


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

        oof_train[test_index] = clf.predict(x_te)**4
        oof_test_skf[i, :] = clf.predict(X_test)**4
        print mean_absolute_error(y_te**4, oof_train[test_index])
        print
    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


svr_params = {'C': 0.0027825594022071257}

svr = SklearnWrapper(clf=LinearSVR, seed=RANDOM_STATE, params=svr_params)

xg_oof_train, xg_oof_test = get_oof(svr)

print("svr-CV: {}".format(mean_absolute_error(y_train**4, xg_oof_train)))

oof_train = pd.DataFrame({'id': train_ids, 'loss': xg_oof_train[:, 0]})
oof_train.to_csv('oof/svr_train_sqrt_1.csv', index=False)

oof_test = pd.DataFrame({'id': test_ids, 'loss': xg_oof_test[:, 0]})
oof_test.to_csv('oof/svr_test_sqrt_1.csv', index=False)

