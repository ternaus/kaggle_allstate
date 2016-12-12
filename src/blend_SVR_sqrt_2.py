"""
Blending models
"""

from __future__ import division

import pandas as pd

import sys

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from pylab import *
import clean_data
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR, SVR


def eval_f(x, y):
    return mean_absolute_error(x**4, y**4)


train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

xgb_train = pd.read_csv('oof2/xgb_train.csv').rename(columns={'loss': 'xgb_loss'})
xgb_test = pd.read_csv('oof2/xgb_test.csv').rename(columns={'loss': 'xgb_loss'})

nn_train_1 = pd.read_csv('oof2/NN_train_s8_mean.csv').rename(columns={'loss': 'nn_loss_8'})
nn_test_1 = pd.read_csv('oof2/NN_test_s8_mean.csv').rename(columns={'loss': 'nn_loss_8'})

lgbm_train_s1 = pd.read_csv('oof2/lgbm_train_sqrt.csv').rename(columns={'loss': 'lgbm_loss_s1'})
lgbm_test_s1 = pd.read_csv('oof2/lgbm_test_sqrt.csv').rename(columns={'loss': 'lgbm_loss_s1'})

lr_train = pd.read_csv('oof2/lr_train_sqrt.csv').rename(columns={'loss': 'lr_loss'})
lr_test = pd.read_csv('oof2/lr_test_sqrt.csv').rename(columns={'loss': 'lr_loss'})

svr_train = pd.read_csv('oof2/svr_train_sqrt.csv').rename(columns={'loss': 'svr_loss'})
svr_test = pd.read_csv('oof2/svr_test_sqrt.csv').rename(columns={'loss': 'svr_loss'})


X_train = (train[['id', 'loss']]
           .merge(xgb_train, on='id')
           .merge(nn_train_1, on='id')
           .merge(lgbm_train_s1, on='id')
            .merge(lr_train, on='id')
            .merge(svr_train, on='id')
           )

X_test = (test[['id', 'cat1']]
          .merge(xgb_test, on='id')
          .merge(nn_test_1, on='id')
          .merge(lgbm_test_s1, on='id')
          .merge(lr_test, on='id')
          .merge(svr_test, on='id')
          .drop('cat1', 1))

y_train = np.sqrt(np.sqrt(X_train['loss'].values))

X_train_id = X_train['id'].values
X_test_id = X_test['id'].values

X_train = X_train.drop(['id', 'loss'], 1).applymap(lambda x: np.sqrt(np.sqrt(x))).values
X_test = X_test.drop('id', 1).applymap(lambda x: np.sqrt(np.sqrt(x))).values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# X_test = np.hstack([X_test, nn_class_test.drop('id', 1).values])
# X_train = np.hstack([X_train, nn_class_train.drop('id', 1).values])

test_ids = test['id']

num_rounds = 300000
RANDOM_STATE = 2016


parameters = {'C': np.logspace(-2, -1, 10)}

n_folds = 6

kf = StratifiedKFold(n_folds, shuffle=True, random_state=RANDOM_STATE)
classes = clean_data.classes(y_train, bins=100)

svr = LinearSVR(random_state=RANDOM_STATE)


clf = GridSearchCV(svr, parameters, n_jobs=-1, cv=kf.get_n_splits(classes, classes), scoring=make_scorer(eval_f),
                   iid=False, verbose=2)

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

            pred += clf.predict(x_te)**4
            pred_test += clf.predict(X_test)**4

        pred /= nbags
        pred_oob[test_index] = pred
        score = mean_absolute_error(y_te**4, pred)
        print('Fold ', i, '- MAE:', score)

    return pred_oob, pred_test

svr = LinearSVR(C=0.059948425031894091)
# svr = Ridge()

svr_oof_train, svr_oof_test = get_oof(svr)

print("SVR-CV: {}".format(mean_absolute_error(y_train**4, svr_oof_train)))

oof_train = pd.DataFrame({'id': X_train_id, 'loss': svr_oof_train})
oof_train.to_csv('oof3/svr_train_sqrt_1.csv', index=False)

svr_oof_test /= n_folds

oof_test = pd.DataFrame({'id': X_test_id, 'loss': svr_oof_test})
oof_test.to_csv('oof3/svr_test_sqrt_1.csv', index=False)

