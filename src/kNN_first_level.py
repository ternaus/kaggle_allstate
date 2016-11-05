"""
kNN on numerical features
"""
from __future__ import division
import pandas as pd

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from pylab import *

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
test['loss'] = np.nan
joined = pd.concat([train, test])

num_columns = [x for x in train.columns if 'cont' in x]
print len(num_columns)

for column in num_columns:
    scaler = StandardScaler()
    joined[column] = scaler.fit_transform(joined[column])


train = joined[joined['loss'].notnull()].reset_index(drop=True)
test = joined[joined['loss'].isnull()]

shift = 200

ids = test['id']
X_train = train[num_columns].values
X_test = test[num_columns].values
y_train = np.log(train['loss'] + shift).values

num_rounds = 300000
RANDOM_STATE = 2016

n_folds = 5
num_train = len(y_train)
num_test = test.shape[0]

kf = KFold(n_folds, shuffle=True, random_state=RANDOM_STATE)

knn_params = {
    'n_jobs': -1,
    'p': 1,
    'n_neighbors': 190
}


class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None):
        # params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)


def get_oof(clf):
    oof_train = np.zeros((num_train,))
    oof_test = np.zeros((num_test,))
    oof_test_skf = np.empty((n_folds, num_test))

    for i, (train_index, test_index) in enumerate(kf.split(X_train)):
        print "Fold = ", i
        x_tr = X_train[train_index]
        y_tr = y_train[train_index]
        x_te = X_train[test_index]
        y_te = y_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(X_test)
        print mean_absolute_error(np.exp(y_te), np.exp(oof_train[test_index]))
        print
    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


et = SklearnWrapper(clf=KNeighborsRegressor, seed=RANDOM_STATE, params=knn_params)
xg_oof_train, xg_oof_test = get_oof(et)

print
print xg_oof_train[:, 0].shape, train.shape
print xg_oof_train[:, 0]

print("kNN_numeric-CV: {}".format(mean_absolute_error(np.exp(y_train), np.exp(xg_oof_train))))

oof_train = pd.DataFrame({'id': train['id'], 'loss': (np.exp(xg_oof_train) - shift)[:, 0]})
oof_train.to_csv('oof/knn_numeric_train.csv', index=False)

oof_test = pd.DataFrame({'id': test['id'], 'loss': (np.exp(xg_oof_test) - shift)[:, 0]})
oof_test.to_csv('oof/knn_numeric_test.csv', index=False)

