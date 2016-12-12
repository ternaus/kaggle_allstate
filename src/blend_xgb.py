"""
Blending models
"""

from __future__ import division

import pandas as pd

import sys
import datetime

sys.path += ['/home/vladimir/packages/xgboost/python-package']

import xgboost as xgb
import numpy as np

from sklearn.metrics import mean_absolute_error
from sklearn.utils import shuffle


def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(preds), np.exp(labels))


def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    con = 2
    x = preds - labels
    grad = con * x / (np.abs(x) + con)
    hess = con**2 / (np.abs(x) + con)**2
    return grad, hess

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

xgb_train_1 = pd.read_csv('oof/xgb_train_t1.csv').rename(columns={'loss': 'xgb_loss_1'})
xgb_test_1 = pd.read_csv('oof/xgb_test_t1.csv').rename(columns={'loss': 'xgb_loss_1'})

xgb_train_s1 = pd.read_csv('oof/xgb_train_s1.csv').rename(columns={'loss': 'xgb_loss_s1'})
xgb_test_s1 = pd.read_csv('oof/xgb_test_s1.csv').rename(columns={'loss': 'xgb_loss_s1'})

xgb_train_s2 = pd.read_csv('oof/xgb_train_s2.csv').rename(columns={'loss': 'xgb_loss_s2'})
xgb_test_s2 = pd.read_csv('oof/xgb_test_s2.csv').rename(columns={'loss': 'xgb_loss_s2'})

xgb_train_s3 = pd.read_csv('oof/xgb_train_s3.csv').rename(columns={'loss': 'xgb_loss_s3'})
xgb_test_s3 = pd.read_csv('oof/xgb_test_s3.csv').rename(columns={'loss': 'xgb_loss_s3'})

xgb_train_s4 = pd.read_csv('oof/xgb_train_s4.csv').rename(columns={'loss': 'xgb_loss_s4'})
xgb_test_s4 = pd.read_csv('oof/xgb_test_s4.csv').rename(columns={'loss': 'xgb_loss_s4'})

nn_train_1 = pd.read_csv('oof/NN_train_p1.csv').rename(columns={'loss': 'nn_loss_1'})
nn_test_1 = pd.read_csv('oof/NN_test_p1.csv').rename(columns={'loss': 'nn_loss_1'})

nn_train_2 = pd.read_csv('oof/NN_train_p2.csv').rename(columns={'loss': 'nn_loss_2'})
nn_test_2 = pd.read_csv('oof/NN_test_p2.csv').rename(columns={'loss': 'nn_loss_2'})

nn_train_3 = pd.read_csv('oof/NN_train_p3.csv').rename(columns={'loss': 'nn_loss_3'})
nn_test_3 = pd.read_csv('oof/NN_test_p3.csv').rename(columns={'loss': 'nn_loss_3'})

lgbt_train_1 = pd.read_csv('oof/lgbt_train_1.csv').rename(columns={'loss': 'lgbt_loss_1'})
lgbt_test_1 = pd.read_csv('oof/lgbt_test_1.csv').rename(columns={'loss': 'lgbt_loss_1'})

et_train_1 = pd.read_csv('oof/et_train.csv').rename(columns={'loss': 'et_loss'})
et_test_1 = pd.read_csv('oof/et_test.csv').rename(columns={'loss': 'et_loss'})

et_train_s1 = pd.read_csv('oof/et_train_s1.csv').rename(columns={'loss': 'et_loss_s1'})
et_test_s1 = pd.read_csv('oof/et_test_s1.csv').rename(columns={'loss': 'et_loss_s1'})

rf_train_s1 = pd.read_csv('oof/rf_train_s1.csv').rename(columns={'loss': 'rf_loss_s1'})
rf_test_s1 = pd.read_csv('oof/rf_test_s1.csv').rename(columns={'loss': 'rf_loss_s1'})


X_train = (train[['id', 'loss']]
           .merge(xgb_train_1, on='id')
           .merge(xgb_train_s1, on='id')
            .merge(xgb_train_s2, on='id')
           .merge(xgb_train_s3, on='id')
           .merge(xgb_train_s4, on='id')
           .merge(nn_train_1, on='id')
           .merge(nn_train_2, on='id')
            .merge(nn_train_3, on='id')
            .merge(lgbt_train_1, on='id')
           .merge(et_train_1, on='id')
           .merge(et_train_s1, on='id')
           .merge(rf_train_s1, on='id')
           )

X_test = (test[['id', 'cat1']]
          .merge(xgb_test_1, on='id')
          .merge(xgb_test_s1, on='id')
          .merge(xgb_test_s2, on='id')
          .merge(xgb_test_s3, on='id')
          .merge(xgb_test_s4, on='id')
          .merge(nn_test_1, on='id')
          .merge(nn_test_2, on='id')
          .merge(nn_test_3, on='id')
          .merge(lgbt_test_1, on='id')
          .merge(et_test_1, on='id')
          .merge(et_test_s1, on='id')
          .merge(rf_test_s1, on='id')
          .drop('cat1', 1))


y_train = np.sqrt(np.sqrt(X_train['loss'].values))

X_train_id = X_train['id'].values
X_test_id = X_test['id'].values


X_train = X_train.drop(['id', 'loss'], 1).applymap(lambda x: np.sqrt(np.sqrt(x))).values
X_test = X_test.drop('id', 1).applymap(lambda x: np.sqrt(np.sqrt(x))).values


num_rounds = 300000
RANDOM_STATE = 2016
params = {
    #     "objective": "binary:logistic",
    # 'booster': 'dart',
    # 'rate_drop': 0.1,
    # 'scale_pos_weight':  1,
    'min_child_weight': 10,
    'eta': 0.01,
    'colsample_bytree': 0.75,
    'max_depth': 5,
    'subsample': 0.9,
    # 'alpha': 5,
    'lambda': 5,
    'gamma': 0,
    'silent': 1,
    # 'base_score': 3,
    # 'eval_metric': 'mae'],
    'verbose_eval': True,
    'seed': RANDOM_STATE
}


print '[{datetime}] train set size = {size}'.format(datetime=str(datetime.datetime.now()), size=X_train.shape)

print '[{datetime}] splitting'.format(datetime=str(datetime.datetime.now()))
xgtrain = xgb.DMatrix(X_train, label=y_train)


res = xgb.cv(params, xgtrain, num_boost_round=num_rounds, nfold=5, stratified=False,
             early_stopping_rounds=50, verbose_eval=1, show_stdv=True, maximize=False,
             obj=logregobj, feval=evalerror)

best_nrounds = res.shape[0] - 1
cv_mean = res.iloc[-1, 0]
cv_std = res.iloc[-1, 1]
print('CV-Mean: {0}+{1}'.format(cv_mean, cv_std))


model = xgb.train(params, xgtrain, int(best_nrounds / (1 - 1.0/5)), feval=evalerror)

xgtest = xgb.DMatrix(X_test)

print '[{datetime}] Creating submission'.format(datetime=str(datetime.datetime.now()))


prediction = model.predict(xgtest)**4

submission = pd.DataFrame()
submission['loss'] = prediction
submission['id'] = X_test_id
submission.to_csv('xgb+NN.csv', index=False)
