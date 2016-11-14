"""
Blending models
"""

from __future__ import division

import pandas as pd

import sys

sys.path += ['/home/vladimir/packages/xgboost/python-package']

import xgboost as xgb

from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import KFold

from pylab import *


def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(preds), np.exp(labels))


train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

xgb_train = pd.read_csv('oof/xgb_train.csv').rename(columns={'loss': 'xgb_loss'})
xgb_test = pd.read_csv('oof/xgb_test.csv').rename(columns={'loss': 'xgb_loss'})

nn_train = pd.read_csv('oof/NN_train.csv').rename(columns={'loss': 'nn_loss'})
nn_test = pd.read_csv('oof/NN_test.csv').rename(columns={'loss': 'nn_loss'})
et_train = pd.read_csv('oof/et_train.csv').rename(columns={'loss': 'et_loss'})
et_test = pd.read_csv('oof/et_test.csv').rename(columns={'loss': 'et_loss'})

X_train = (train[['id', 'loss']]
           .merge(xgb_train, on='id')
           .merge(nn_train, on='id')
           .merge(et_train, on='id')
           )

X_test = (test[['id', 'cat1']]
          .merge(xgb_test, on='id')
          .merge(nn_test, on='id')
          .merge(et_test, on='id')
          .drop('cat1', 1))


shift = 200

y = np.log(X_train['loss'] + shift)
X_train = X_train.drop(['id', 'loss'], 1).applymap(lambda x: np.log(x + shift))
X_test = X_test.drop('id', 1).applymap(lambda x: np.log(x + shift))

test_ids = test['id']

print X_train.shape, xgb_train.shape, nn_train.shape

num_rounds = 300000
RANDOM_STATE = 2016
params = {
    'booster': 'gblinear',
    'lambda': 10,
    # #     "objective": "binary:logistic",
    # # 'booster': 'dart',
    # # 'rate_drop': 0.1,
    # # 'scale_pos_weight':  1,
    # 'min_child_weight': 1,
    # 'eta': 0.01,
    # # 'colsample_bytree': 0.6,
    # 'max_depth': 3,
    # 'subsample': 0.8,
    # # 'alpha': 5,
    # 'gamma': 1,
    # 'silent': 1,
    # # 'base_score': 3,
    # # 'eval_metric': ['rmse', 'mae'],
    # 'verbose_eval': True,
    # 'seed': RANDOM_STATE
}


print '[{datetime}] train set size = {size}'.format(datetime=str(datetime.datetime.now()), size=X_train.shape)

print '[{datetime}] splitting'.format(datetime=str(datetime.datetime.now()))
xgtrain = xgb.DMatrix(X_train, label=y)


res = xgb.cv(params, xgtrain, num_boost_round=num_rounds, nfold=5, stratified=False,
             early_stopping_rounds=50, verbose_eval=1, show_stdv=True, maximize=False, feval=evalerror)

best_nrounds = res.shape[0] - 1
cv_mean = res.iloc[-1, 0]
cv_std = res.iloc[-1, 1]
print('CV-Mean: {0}+{1}'.format(cv_mean, cv_std))

model = xgb.train(params, xgtrain, int(best_nrounds / (1 - 1.0/5)), feval=evalerror)

xgtest = xgb.DMatrix(X_test)

print '[{datetime}] Creating submission'.format(datetime=str(datetime.datetime.now()))


prediction = np.exp(model.predict(xgtest)) - shift

submission = pd.DataFrame()
submission['loss'] = prediction
submission['id'] = test_ids
submission.to_csv('xgb+NN.csv', index=False)
