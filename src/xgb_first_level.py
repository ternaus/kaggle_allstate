import pandas as pd

import sys

sys.path += ['/home/vladimir/packages/xgboost/python-package']
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from pylab import *

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
test['loss'] = np.nan
joined = pd.concat([train, test])


def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(preds), np.exp(labels))


def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    con = 2
    x =preds-labels
    grad =con * x / (np.abs(x) + con)
    hess =con**2 / (np.abs(x) + con)**2
    return grad, hess


for column in list(train.select_dtypes(include=['object']).columns):
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

train = joined[joined['loss'].notnull()].reset_index(drop=True)
test = joined[joined['loss'].isnull()]

shift = 200

ids = test['id']
X_train = train.drop(['loss', 'id'], 1).values
X_test = test.drop(['loss', 'id'], 1).values
y_train = np.log(train['loss'] + shift).values

num_rounds = 300000
RANDOM_STATE = 2016
xgb_params = {
    'min_child_weight': 1,
    'eta': 0.01,
    'colsample_bytree': 0.5,
    'max_depth': 13,
    'subsample': 0.8,
    'alpha': 5,
    'gamma': 1,
    'silent': 1,
    # 'base_score': 2,
    'verbose_eval': 1,
    'seed': RANDOM_STATE,
    'nrounds': int(5903 / 0.8)
}

n_folds = 5
num_train = len(y_train)
num_test = test.shape[0]

kf = KFold(n_folds, shuffle=True, random_state=RANDOM_STATE)


class XgbWrapper(object):
    def __init__(self, seed=0, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 250)

    def train(self, x_train, y_train):
        dtrain = xgb.DMatrix(x_train, label=y_train)
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds, feval=evalerror, obj=logregobj)

    def predict(self, x):
        return self.gbdt.predict(xgb.DMatrix(x))


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


xg = XgbWrapper(seed=RANDOM_STATE, params=xgb_params)
xg_oof_train, xg_oof_test = get_oof(xg)

print
print xg_oof_train[:, 0].shape, train.shape
print xg_oof_train[:, 0]

print("XG-CV: {}".format(mean_absolute_error(np.exp(y_train), np.exp(xg_oof_train))))

oof_train = pd.DataFrame({'id': train['id'], 'loss': (np.exp(xg_oof_train) - shift)[:, 0]})
oof_train.to_csv('oof/xgb_train.csv', index=False)

oof_test = pd.DataFrame({'id': test['id'], 'loss': (np.exp(xg_oof_test) - shift)[:, 0]})
oof_test.to_csv('oof/xgb_test.csv', index=False)

