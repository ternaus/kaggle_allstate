"""
kNN on numerical features
"""
from __future__ import division
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from pylab import *
from sklearn.preprocessing import StandardScaler


def eval_f(x, y):
    return mean_absolute_error(np.exp(x), np.exp(y))


from sklearn.neighbors import KNeighborsRegressor


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

num_train = len(y_train)
num_test = test.shape[0]

n_folds = 5
kf = KFold(n_folds, shuffle=True, random_state=RANDOM_STATE)


parameters = {'n_neighbors': range(2, 20), 'p': range(2, 20)}

kNN = KNeighborsRegressor()
clf = RandomizedSearchCV(kNN, parameters, n_jobs=-1, cv=kf.get_n_splits(X_train), scoring=make_scorer(eval_f), verbose=10)
# clf = GridSearchCV(lr, parameters, n_jobs=-1, cv=5, scoring=make_scorer(eval_f), iid=False)
# clf = GridSearchCV(lr, parameters, n_jobs=-1, cv=kf, scoring='neg_mean_absolute_error')
#
clf.fit(X_train, y_train)

print
print clf.cv_results_
print clf.scorer_
#
#
#
# kf = KFold(n_folds, shuffle=True, random_state=RANDOM_STATE)
#
# et_params = {
#     'n_jobs': -1,
#     'n_estimators': 1000,
#     'max_features': 0.5,
#     'max_depth': 28,
#     'min_samples_leaf': 2,
# }
#
# class SklearnWrapper(object):
#     def __init__(self, clf, seed=0, params=None):
#         params['random_state'] = seed
#         self.clf = clf(**params)
#
#     def train(self, x_train, y_train):
#         self.clf.fit(x_train, y_train)
#
#     def predict(self, x):
#         return self.clf.predict(x)
#
#
# def get_oof(clf):
#     oof_train = np.zeros((num_train,))
#     oof_test = np.zeros((num_test,))
#     oof_test_skf = np.empty((n_folds, num_test))
#
#     for i, (train_index, test_index) in enumerate(kf.split(X_train)):
#         print "Fold = ", i
#         x_tr = X_train[train_index]
#         y_tr = y_train[train_index]
#         x_te = X_train[test_index]
#         y_te = y_train[test_index]
#
#         clf.train(x_tr, y_tr)
#
#         oof_train[test_index] = clf.predict(x_te)
#         oof_test_skf[i, :] = clf.predict(X_test)
#         print mean_absolute_error(np.exp(y_te), np.exp(oof_train[test_index]))
#         print
#     oof_test[:] = oof_test_skf.mean(axis=0)
#     return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
#
#
# et = SklearnWrapper(clf=ExtraTreesRegressor, seed=RANDOM_STATE, params=et_params)
# xg_oof_train, xg_oof_test = get_oof(et)
#
# print
# print xg_oof_train[:, 0].shape, train.shape
# print xg_oof_train[:, 0]
#
# print("et-CV: {}".format(mean_absolute_error(np.exp(y_train), np.exp(xg_oof_train))))
#
# oof_train = pd.DataFrame({'id': train['id'], 'loss': (np.exp(xg_oof_train) - shift)[:, 0]})
# oof_train.to_csv('oof/et_train.csv', index=False)
#
# oof_test = pd.DataFrame({'id': test['id'], 'loss': (np.exp(xg_oof_test) - shift)[:, 0]})
# oof_test.to_csv('oof/et_test.csv', index=False)
#
