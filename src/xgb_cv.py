import pandas as pd

from sklearn.model_selection import train_test_split
import sys

sys.path += ['/home/vladimir/packages/xgboost/python-package']

import xgboost as xgb
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import KFold

from pylab import *
from tqdm import tqdm

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
test['loss'] = np.nan
joined = pd.concat([train, test])

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
# y_train = train['loss']

ids = test['id']
X = train.drop(['loss', 'id'], 1)
X_test = test.drop(['loss', 'id'], 1)


num_rounds = 20000
RANDOM_STATE = 2016
params = {
    #     "objective": "binary:logistic",
    # 'booster': 'dart',
    # 'rate_drop': 0.1,
    # 'scale_pos_weight':  1,
    'min_child_weight': 1,
    'eta': 0.001,
    'colsample_bytree': 0.5,
    'max_depth': 12,
    'subsample': 0.8,
    'gamma': 1,
    'alpha': 1,
    'silent': 1,
    'eval_metric': ['rmse', 'mae'],
    'verbose_eval': True,
    'seed': RANDOM_STATE
}

print '[{datetime}] train set size = {size}'.format(datetime=str(datetime.datetime.now()), size=X.shape)

print '[{datetime}] splitting'.format(datetime=str(datetime.datetime.now()))

## cv-folds
nfolds = 5
folds = KFold(len(y), n_folds = nfolds, shuffle=True, random_state=111)

pred = []

test_preds = []

for (inTr, inTe) in folds:
    print inTr
    X_train = X.values[inTr]
    y_train = y.values[inTr]
    X_val = X.values[inTe]
    y_val = y.values[inTe]
    print X_train.shape, y_train.shape, X_val.shape, y_val.shape

    # X_train, X_val, y_train, y_val = train_test_split(X, y_train, test_size=0.1, random_state=RANDOM_STATE)

    # print X_train.shape, X_val.shape, y_train.shape, y_val.shape
    #
    # print X_train.shape, y_train.shape

    xgtrain = xgb.DMatrix(X_train, label=y_train)
    # xgtrain = xgb.DMatrix(X, label=y_train)

    xgval = xgb.DMatrix(X_val, label=y_val)
    xgtest = xgb.DMatrix(X_test.values)

    eval_list = [(xgtrain, 'train'), (xgval, 'val')]

    # model = xgb.train(params, xgtrain, 200, eval_list, early_stopping_rounds=20)
    model = xgb.train(params, xgtrain, num_rounds, eval_list, early_stopping_rounds=20, feval=evalerror)

    val_prediction = np.exp(model.predict(xgval)) - shift
    # val_prediction = model.predict(xgval)

    val_score = mean_absolute_error(np.exp(y_val) - shift, val_prediction)
    # val_score = mean_absolute_error(y_val, val_prediction)
    print val_score
    prediction = np.exp(model.predict(xgtest)) - shift
    # prediction = model.predict(xgtest)
    test_preds += [prediction]

    pred += [val_score]

print pred
print np.mean(pred), np.std(pred)


print test_preds[0].shape
submission = pd.DataFrame()
submission['loss'] = np.array(test_preds).mean(axis=0)

submission['id'] = ids
submission.to_csv('sub5.csv', index=False)
