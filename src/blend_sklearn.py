"""
Blending models
"""

from __future__ import division

import pandas as pd

import sys

sys.path += ['/home/vladimir/packages/xgboost/python-package']

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from pylab import *


def eval_f(x, y):
    return mean_absolute_error(np.exp(x), np.exp(y))

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

xgb_train_0 = pd.read_csv('oof/xgb_train.csv').rename(columns={'loss': 'xgb_loss_0'})
xgb_test_0 = pd.read_csv('oof/xgb_test.csv').rename(columns={'loss': 'xgb_loss_0'})

xgb_train = pd.read_csv('oof/xgb_train_t.csv').rename(columns={'loss': 'xgb_loss'})
xgb_test = pd.read_csv('oof/xgb_test_t.csv').rename(columns={'loss': 'xgb_loss'})

nn_train = pd.read_csv('oof/NN_train.csv').rename(columns={'loss': 'nn_loss'})
nn_test = pd.read_csv('oof/NN_test.csv').rename(columns={'loss': 'nn_loss'})

nn_train_1 = pd.read_csv('oof/NN_train_1.csv').rename(columns={'loss': 'nn_loss_1'})
nn_test_1 = pd.read_csv('oof/NN_test_1.csv').rename(columns={'loss': 'nn_loss_1'})


et_train = pd.read_csv('oof/et_train.csv').rename(columns={'loss': 'et_loss'})
et_test = pd.read_csv('oof/et_test.csv').rename(columns={'loss': 'et_loss'})


lr_train = pd.read_csv('oof/lr_train.csv').rename(columns={'loss': 'lr_loss'})
lr_test = pd.read_csv('oof/lr_test.csv').rename(columns={'loss': 'lr_loss'})

knn_numeric_train = pd.read_csv('oof/knn_numeric_train.csv').rename(columns={'loss': 'knn_numeric_loss'})
knn_numeric_test = pd.read_csv('oof/knn_numeric_test.csv').rename(columns={'loss': 'knn_numeric_loss'})


X_train = (train[['id', 'loss']]
           .merge(xgb_train_0, on='id')
           .merge(xgb_train, on='id')
           .merge(nn_train, on='id')
           .merge(nn_train_1, on='id')
           .merge(et_train, on='id')
           # .merge(lr_train, on='id')
           # .merge(knn_numeric_train, on='id')
           )

X_test = (test[['id', 'cat1']]
          .merge(xgb_test_0, on='id')
          .merge(xgb_test, on='id')
          .merge(nn_test, on='id')
          .merge(nn_test_1, on='id')
          .merge(et_test, on='id')
          # .merge(lr_train, on='id')
          # .merge(knn_numeric_test, on='id')
          .drop('cat1', 1))

shift = 200

y_train = np.log(X_train['loss'] + shift)

X_train = X_train.drop(['id', 'loss'], 1).applymap(lambda x: np.log(x + shift)).values
X_test = X_test.drop('id', 1).applymap(lambda x: np.log(x + shift)).values

# y_train = X_train['loss']

# X_train = X_train.drop(['id', 'loss'], 1).values
# X_test = X_test.drop('id', 1).values


test_ids = test['id']

print X_train.shape, xgb_train.shape, nn_train.shape

num_rounds = 300000
RANDOM_STATE = 2016


parameters = {'alpha': [0, 0.01, 0.1, 1, 10, 50, 100, 150, 180, 190, 200, 205, 210, 220, 250, 300, 500]}

n_folds = 5
kf = KFold(n_folds, shuffle=True, random_state=RANDOM_STATE).get_n_splits(X_train)

lr = Ridge(fit_intercept=False)
clf = GridSearchCV(lr, parameters, n_jobs=-1, cv=kf, scoring=make_scorer(eval_f), iid=False)

clf.fit(X_train, y_train)

for i in clf.grid_scores_:
    print i

lr = Ridge(fit_intercept=False, alpha=0)
lr.fit(X_train, y_train)

test_pred = np.exp(lr.predict(X_test)) - shift
submission = pd.DataFrame()
submission['loss'] = test_pred
submission['id'] = test_ids
submission.to_csv('blen1.csv', index=False)

# print clf.scorer_
#
#
# print '[{datetime}] train set size = {size}'.format(datetime=str(datetime.datetime.now()), size=X_train.shape)
#
# print '[{datetime}] splitting'.format(datetime=str(datetime.datetime.now()))
# xgtrain = xgb.DMatrix(X_train, label=y)
#
#
# res = xgb.cv(params, xgtrain, num_boost_round=num_rounds, nfold=5, stratified=False,
#              early_stopping_rounds=50, verbose_eval=1, show_stdv=True, maximize=False, feval=evalerror)
#
# best_nrounds = res.shape[0] - 1
# cv_mean = res.iloc[-1, 0]
# cv_std = res.iloc[-1, 1]
# print('CV-Mean: {0}+{1}'.format(cv_mean, cv_std))
#
# model = xgb.train(params, xgtrain, int(best_nrounds / (1 - 1.0/5)), feval=evalerror)
#
# xgtest = xgb.DMatrix(X_test)
#
# print '[{datetime}] Creating submission'.format(datetime=str(datetime.datetime.now()))
#
#
# prediction = np.exp(model.predict(xgtest)) - shift
#
# submission = pd.DataFrame()
# submission['loss'] = prediction
# submission['id'] = test_ids
# submission.to_csv('xgb+NN.csv', index=False)
