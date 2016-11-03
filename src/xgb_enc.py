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

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
test['loss'] = np.nan
joined = pd.concat([train, test])


def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    con = 2
    x = preds - labels
    grad =con * x / (np.abs(x) + con)
    hess =con**2 / (np.abs(x) + con)**2
    return grad, hess


def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(preds), np.exp(labels))


for column in list(train.select_dtypes(include=['object']).columns):
    # g = train.groupby(column)['loss'].mean()
    # g = train.groupby(column)['loss'].median()

    if train[column].nunique() != test[column].nunique():
        # Let's find extra categories...
        set_train = set(train[column].unique())
        set_test = set(test[column].unique())
        remove_train = set_train - set_test
        remove_test = set_test - set_train

        remove = remove_train.union(remove_test)
        print column, remove

        def filter_cat(x):
            if x in remove:
                return np.nan
            return x

        joined[column] = joined[column].apply(lambda x: filter_cat(x), 1)
        print 'unique =', joined[column].nunique()

    joined[column] = pd.factorize(joined[column].values, sort=True)[0]

train = joined[joined['loss'].notnull()]
test = joined[joined['loss'].isnull()]

shift = 200
y = np.log(train['loss'] + shift)
ids = test['id']
X = train.drop(['loss', 'id'], 1)
X_test = test.drop(['loss', 'id'], 1)

num_rounds = 300000
RANDOM_STATE = 2016
params = {
    #     "objective": "binary:logistic",
    # 'booster': 'dart',
    # 'rate_drop': 0.1,
    # 'scale_pos_weight':  1,
    'min_child_weight': 1,
    'eta': 0.01,
    'colsample_bytree': 0.6,
    'max_depth': 13,
    'subsample': 0.8,
    'alpha': 5,
    'gamma': 1,
    'silent': 1,
    # 'base_score': 3,
    # 'eval_metric': ['rmse', 'mae'],
    'verbose_eval': True,
    'seed': RANDOM_STATE
}


print '[{datetime}] train set size = {size}'.format(datetime=str(datetime.datetime.now()), size=X.shape)

print '[{datetime}] splitting'.format(datetime=str(datetime.datetime.now()))
xgtrain = xgb.DMatrix(X, label=y)


res = xgb.cv(params, xgtrain, num_boost_round=num_rounds, nfold=5, stratified=False,
             early_stopping_rounds=50, verbose_eval=1, show_stdv=True, feval=evalerror, maximize=False, obj=logregobj)

# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=RANDOM_STATE)

# print X_train.shape, X_val.shape, y_train.shape, y_val.shape


# xgtrain = xgb.DMatrix(X_train, label=y_train)
# xgtrain = xgb.DMatrix(X, label=y)
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
