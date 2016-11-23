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


def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    con = 2
    x = preds - labels
    grad = con * x / (np.abs(x) + con)
    hess = con**2 / (np.abs(x) + con)**2
    return grad, hess



train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

xgb_train_0 = pd.read_csv('oof/xgb_train.csv').rename(columns={'loss': 'xgb_loss_0'})
xgb_test_0 = pd.read_csv('oof/xgb_test.csv').rename(columns={'loss': 'xgb_loss_0'})

xgb_train = pd.read_csv('oof/xgb_train_t.csv').rename(columns={'loss': 'xgb_loss'})
xgb_test = pd.read_csv('oof/xgb_test_t.csv').rename(columns={'loss': 'xgb_loss'})

xgb_train_1 = pd.read_csv('oof/xgb_train_t.csv').rename(columns={'loss': 'xgb_loss_1'})
xgb_test_1 = pd.read_csv('oof/xgb_test_t.csv').rename(columns={'loss': 'xgb_loss_1'})

xgb_train_2 = pd.read_csv('oof/xgb_train_t2.csv').rename(columns={'loss': 'xgb_loss_2'})
xgb_test_2 = pd.read_csv('oof/xgb_test_t2.csv').rename(columns={'loss': 'xgb_loss_2'})

xgb_train_3 = pd.read_csv('oof/xgb_train_t3.csv').rename(columns={'loss': 'xgb_loss_3'})
xgb_test_3 = pd.read_csv('oof/xgb_test_t3.csv').rename(columns={'loss': 'xgb_loss_3'})

xgb_train_4 = pd.read_csv('oof/xgb_train_t4.csv').rename(columns={'loss': 'xgb_loss_4'})
xgb_test_4 = pd.read_csv('oof/xgb_test_t4.csv').rename(columns={'loss': 'xgb_loss_4'})

xgb_train_5 = pd.read_csv('oof/xgb_train_t5.csv').rename(columns={'loss': 'xgb_loss_5'})
xgb_test_5 = pd.read_csv('oof/xgb_test_t5.csv').rename(columns={'loss': 'xgb_loss_5'})

xgb_train_6 = pd.read_csv('oof/xgb_train_t6.csv').rename(columns={'loss': 'xgb_loss_6'})
xgb_test_6 = pd.read_csv('oof/xgb_test_t6.csv').rename(columns={'loss': 'xgb_loss_6'})

nn_train = pd.read_csv('oof/NN_train.csv').rename(columns={'loss': 'nn_loss'})
nn_test = pd.read_csv('oof/NN_test.csv').rename(columns={'loss': 'nn_loss'})

nn_train_1 = pd.read_csv('oof/NN_train_1.csv').rename(columns={'loss': 'nn_loss_1'})
nn_test_1 = pd.read_csv('oof/NN_test_1.csv').rename(columns={'loss': 'nn_loss_1'})

nn_train_2 = pd.read_csv('oof/NN_train_2.csv').rename(columns={'loss': 'nn_loss_1'})
nn_test_2 = pd.read_csv('oof/NN_test_2.csv').rename(columns={'loss': 'nn_loss_1'})

nn_train_4 = pd.read_csv('oof/NN_train_4.csv').rename(columns={'loss': 'nn_loss_4'})
nn_test_4 = pd.read_csv('oof/NN_test_4.csv').rename(columns={'loss': 'nn_loss_4'})


nn_train_p1 = pd.read_csv('oof/NN_train_p1.csv').rename(columns={'loss': 'nn_loss_p1'})
nn_test_p1 = pd.read_csv('oof/NN_test_p1.csv').rename(columns={'loss': 'nn_loss_p1'})

nn_train_p2 = pd.read_csv('oof/NN_train_p2.csv').rename(columns={'loss': 'nn_loss_p3'})
nn_test_p2 = pd.read_csv('oof/NN_test_p2.csv').rename(columns={'loss': 'nn_loss_p3'})

nn_train_p3 = pd.read_csv('oof/NN_train_p3.csv').rename(columns={'loss': 'nn_loss_p3'})
nn_test_p3 = pd.read_csv('oof/NN_test_p3.csv').rename(columns={'loss': 'nn_loss_p3'})

nn_train_p4 = pd.read_csv('oof/NN_train_p4.csv').rename(columns={'loss': 'nn_loss_p4'})
nn_test_p4 = pd.read_csv('oof/NN_test_p4.csv').rename(columns={'loss': 'nn_loss_p4'})

et_train = pd.read_csv('oof/et_train.csv').rename(columns={'loss': 'et_loss'})
et_test = pd.read_csv('oof/et_test.csv').rename(columns={'loss': 'et_loss'})

rf_train = pd.read_csv('oof/rf_train.csv').rename(columns={'loss': 'rf_loss'})
rf_test = pd.read_csv('oof/rf_test.csv').rename(columns={'loss': 'rf_loss'})

#
# lr_train = pd.read_csv('oof/lr_train.csv').rename(columns={'loss': 'lr_loss'})
# lr_test = pd.read_csv('oof/lr_test.csv').rename(columns={'loss': 'lr_loss'})

lgbt_train = pd.read_csv('oof/lgbt_train.csv').rename(columns={'loss': 'lgbt_loss'})
lgbt_test = pd.read_csv('oof/lgbt_test.csv').rename(columns={'loss': 'lgbt_loss'})

lgbt_train_1 = pd.read_csv('oof/lgbt_train_1.csv').rename(columns={'loss': 'lgbt_loss_1'})
lgbt_test_1 = pd.read_csv('oof/lgbt_test_1.csv').rename(columns={'loss': 'lgbt_loss_1'})


knn_numeric_train = pd.read_csv('oof/knn_numeric_train.csv').rename(columns={'loss': 'knn_numeric_loss'})
knn_numeric_test = pd.read_csv('oof/knn_numeric_test.csv').rename(columns={'loss': 'knn_numeric_loss'})


X_train = (train[['id', 'loss']]
           .merge(xgb_train_0, on='id')
           .merge(xgb_train, on='id')
           # .merge(xgb_train_1, on='id')
           #  .merge(xgb_train_2, on='id')
            .merge(xgb_train_3, on='id')
            .merge(xgb_train_4, on='id')
            .merge(xgb_train_5, on='id')
            .merge(xgb_train_6, on='id')
           .merge(nn_train, on='id')
           .merge(nn_train_1, on='id')
           .merge(nn_train_2, on='id')
            .merge(nn_train_4, on='id')
            .merge(nn_train_p1, on='id')
            .merge(nn_train_p2, on='id')
            .merge(nn_train_p3, on='id')
            .merge(nn_train_p4, on='id')
           .merge(et_train, on='id')
            .merge(rf_train, on='id')
           # .merge(lr_train, on='id')
           #  .merge(lgbt_train, on='id')
            .merge(lgbt_train_1, on='id')
           # .merge(knn_numeric_train, on='id')
           )

X_test = (test[['id', 'cat1']]
          .merge(xgb_test_0, on='id')
          .merge(xgb_test, on='id')
          # .merge(xgb_test_1, on='id')
          #   .merge(xgb_test_2, on='id')
          .merge(xgb_test_3, on='id')
            .merge(xgb_test_4, on='id')
            .merge(xgb_test_5, on='id')
            .merge(xgb_test_6, on='id')
          .merge(nn_test, on='id')
          .merge(nn_test_1, on='id')
          .merge(nn_test_2, on='id')
          .merge(nn_test_4, on='id')
          .merge(nn_test_p1, on='id')
            .merge(nn_test_p2, on='id')
            .merge(nn_test_p3, on='id')
            .merge(nn_test_p4, on='id')
          .merge(et_test, on='id')
          .merge(rf_test, on='id')
          # .merge(lr_test, on='id')
          # .merge(lgbt_test, on='id')
          .merge(lgbt_test_1, on='id')
          # .merge(knn_numeric_test, on='id')
          .drop('cat1', 1))

shift = 200

y_train = np.log(X_train['loss'] + shift)

X_train_id = X_train['id'].values
X_test_id = X_test['id'].values


X_train = X_train.drop(['id', 'loss'], 1).applymap(lambda x: np.log(x + shift)).values
X_test = X_test.drop('id', 1).applymap(lambda x: np.log(x + shift)).values


test_ids = test['id']

print X_train.shape, xgb_train.shape, nn_train.shape

num_rounds = 300000
RANDOM_STATE = 2016
params = {
    #     "objective": "binary:logistic",
    # 'booster': 'dart',
    # 'rate_drop': 0.1,
    # 'scale_pos_weight':  1,
    'min_child_weight': 100,
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


prediction = np.exp(model.predict(xgtest)) - shift

submission = pd.DataFrame()
submission['loss'] = prediction
submission['id'] = test_ids
submission.to_csv('xgb+NN.csv', index=False)
