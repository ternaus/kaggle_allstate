import pandas as pd

from sklearn.model_selection import train_test_split
import sys

sys.path += ['/home/vladimir/packages/xgboost/python-package']
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

from pylab import *
from tqdm import tqdm
import clean_data

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
test['loss'] = np.nan
joined = pd.concat([train, test])


def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    con = 2
    x = preds - labels
    grad = con * x / (np.abs(x) + con)
    hess = con**2 / (np.abs(x) + con)**2
    return grad, hess


def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(preds), np.exp(labels))


shift = 200

X_train, y_train, X_test, y_mean, X_test_id, X_train_id = clean_data.fancy(shift)

X_train = X_train.values
X_test = X_test.values
y_train = y_train.values


num_rounds = 300000
RANDOM_STATE = 2016
xgb_params = {
    'min_child_weight': 1,
    'eta': 0.02,
    'colsample_bytree': 0.5,
    'max_depth': 13,
    'subsample': 0.8,
    'alpha': 5,
    'gamma': 1,
    'silent': 1,
    # 'base_score': 2,
    'verbose_eval': 1,
    'seed': RANDOM_STATE,
    'nrounds': 8000
}

print '[{datetime}] train set size = {size}'.format(datetime=str(datetime.datetime.now()), size=X_train.shape)

print '[{datetime}] splitting'.format(datetime=str(datetime.datetime.now()))
xgtrain = xgb.DMatrix(X_train, label=y_train)


res = xgb.cv(xgb_params, xgtrain, num_boost_round=num_rounds, nfold=5, stratified=False,
             early_stopping_rounds=100, verbose_eval=1, show_stdv=True, feval=evalerror, maximize=False, obj=logregobj)

# X_train, X_val, y_train, y_val = train_test_split(X, y_train, test_size=0.1, random_state=RANDOM_STATE)

# print X_train.shape, X_val.shape, y_train.shape, y_val.shape


# xgtrain = xgb.DMatrix(X_train, label=y_train)
# xgtrain = xgb.DMatrix(X, label=y_train)
#
# xgval = xgb.DMatrix(X_val, label=y_val)
# xgtest = xgb.DMatrix(X_test)
# #
# eval_list = [(xgtrain, 'train'), (xgval, 'val')]
# cv_result = xgb.cv(params, xgtrain, num_boost_round=num_rounds, nfold=5,
#              seed=2016,
#              feval=evalerror,
#              callbacks=[xgb.callback.early_stop(50)])

# # # model = xgb.train(params, xgtrain, 200, eval_list, early_stopping_rounds=20)
# model = xgb.train(params, xgtrain, num_rounds, eval_list, early_stopping_rounds=20, feval=evalerror, obj=logregobj)
# model = xgb.train(params, xgtrain, int(6985 / 0.8), feval=evalerror, obj=logregobj)
# #
# # val_prediction = np.exp(model.predict(xgval))
# #
# # val_score = mean_absolute_error(np.exp(y_val), val_prediction)
# #
# # print '[{datetime}] Val_score = {val_score}'.format(datetime=str(datetime.datetime.now()), val_score=val_score)
# #
# print '[{datetime}] Creating submission'.format(datetime=str(datetime.datetime.now()))
#
#
# prediction = np.exp(model.predict(xgtest)) - shift
#
# submission = pd.DataFrame()
# submission['loss'] = prediction
# submission['id'] = ids
# submission.to_csv('sub13.csv', index=False)
