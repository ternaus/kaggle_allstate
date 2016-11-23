"""
Blending models
"""

from __future__ import division

import pandas as pd

import sys

sys.path += ['/home/vladimir/packages/xgboost/python-package']

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from pylab import *
import clean_data


def eval_f(x, y):
    return mean_absolute_error(np.exp(x), np.exp(y))

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

xgb_train = pd.read_csv('oof2/xgb_train.csv').rename(columns={'loss': 'xgb_loss'})
xgb_test = pd.read_csv('oof2/xgb_test.csv').rename(columns={'loss': 'xgb_loss'})

nn_train = pd.read_csv('oof2/NN_train.csv').rename(columns={'loss': 'nn_loss'})
nn_test = pd.read_csv('oof2/NN_test.csv').rename(columns={'loss': 'nn_loss'})

nn_train_1 = pd.read_csv('oof2/NN_train_1.csv').rename(columns={'loss': 'nn_loss_1'})
nn_test_1 = pd.read_csv('oof2/NN_test_1.csv').rename(columns={'loss': 'nn_loss_1'})

nn_train_2 = pd.read_csv('oof2/NN_train_2.csv').rename(columns={'loss': 'nn_loss_2'})
nn_test_2 = pd.read_csv('oof2/NN_test_2.csv').rename(columns={'loss': 'nn_loss_2'})

lr_train = pd.read_csv('oof2/lr_train.csv').rename(columns={'loss': 'lr_loss'})
lr_test = pd.read_csv('oof2/lr_test.csv').rename(columns={'loss': 'lr_loss'})


X_train = (train[['id', 'loss']]
           .merge(nn_train, on='id')
           .merge(nn_train_1, on='id')
           .merge(nn_train_2, on='id')
            # .merge(xgb_train, on='id')
           .merge(lr_train, on='id')
           )

X_test = (test[['id', 'cat1']]
          .merge(nn_test, on='id')
          .merge(nn_test_1, on='id')
          .merge(nn_test_2, on='id')
        # .merge(xgb_test, on='id')
          .merge(lr_test, on='id')
          .drop('cat1', 1))

shift = 400

y_train = np.log(X_train['loss'] + shift)

X_train_id = X_train['id'].values
X_test_id = X_test['id'].values

X_train = X_train.drop(['id', 'loss'], 1).applymap(lambda x: np.log(x + shift)).values
X_test = X_test.drop('id', 1).applymap(lambda x: np.log(x + shift)).values

# y_train = X_train['loss']

# X_train = X_train.drop(['id', 'loss'], 1).values
# X_test = X_test.drop('id', 1).values


test_ids = test['id']

num_rounds = 300000
RANDOM_STATE = 2016


parameters = {'alpha': [0, 0.001, 0.005, 0.01, 0.1, 1, 10, 50, 100, 150, 180, 190, 200, 205, 210, 220, 240, 250, 260, 300, 400, 500, 750, 800, 850, 1000,
                        1100, 1200, 1300, 1400, 1500, 2000, 3000, 5000]}

n_folds = 5

kf = StratifiedKFold(n_folds, shuffle=True, random_state=RANDOM_STATE)

lr = Ridge(fit_intercept=False)
clf = GridSearchCV(lr, parameters, n_jobs=-1, cv=kf.get_n_splits(X_train, y_train), scoring=make_scorer(eval_f), iid=False)

clf.fit(X_train, y_train)

for i in clf.grid_scores_:
    print i

num_train = X_train.shape[0]
num_test = X_test.shape[0]

classes = clean_data.classes(y_train, bins=100)


nbags = 1


def get_oof(clf):
    pred_oob = np.zeros(X_train.shape[0])
    pred_test = np.zeros(X_test.shape[0])

    for i, (train_index, test_index) in enumerate(kf.split(classes, classes)):
        print "Fold = ", i
        x_tr = X_train[train_index]
        y_tr = y_train[train_index]

        x_te = X_train[test_index]
        y_te = y_train[test_index]

        pred = np.zeros(x_te.shape[0])

        for j in range(nbags):
            # x_tr, y_tr = shuffle(x_tr, y_tr, random_state=RANDOM_STATE + i + j)
            clf.fit(x_tr, y_tr)
            print clf.coef_

            pred += np.exp(clf.predict(x_te))
            pred_test += np.exp(clf.predict(X_test))

        pred /= nbags
        pred_oob[test_index] = pred
        score = mean_absolute_error(np.exp(y_te), pred)
        print('Fold ', i, '- MAE:', score)

    return pred_oob, pred_test

lr = Ridge(fit_intercept=False, alpha=300)

lr_oof_train, lr_oof_test = get_oof(lr)

print lr_oof_train.shape, X_train_id.shape

print("LR-CV: {}".format(mean_absolute_error(np.exp(y_train), lr_oof_train)))

oof_train = pd.DataFrame({'id': X_train_id, 'loss': (lr_oof_train - shift)})
oof_train.to_csv('oof3/lr_train.csv', index=False)

lr_oof_test /= n_folds

oof_test = pd.DataFrame({'id': X_test_id, 'loss': (lr_oof_test - shift)})
oof_test.to_csv('oof3/lr_test.csv', index=False)

