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

xgb_train_7 = pd.read_csv('oof/xgb_train_t7.csv').rename(columns={'loss': 'xgb_loss_7'})
xgb_test_7 = pd.read_csv('oof/xgb_test_t7.csv').rename(columns={'loss': 'xgb_loss_7'})


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

nn_train_p5 = pd.read_csv('oof/NN_train_p5.csv').rename(columns={'loss': 'nn_loss_p5'})
nn_test_p5 = pd.read_csv('oof/NN_test_p5.csv').rename(columns={'loss': 'nn_loss_p5'})


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
            .merge(xgb_train_7, on='id')
           .merge(nn_train, on='id')
           .merge(nn_train_1, on='id')
           .merge(nn_train_2, on='id')
            .merge(nn_train_4, on='id')
            .merge(nn_train_p1, on='id')
            .merge(nn_train_p2, on='id')
            .merge(nn_train_p3, on='id')
            .merge(nn_train_p4, on='id')
            .merge(nn_train_p5, on='id')
           .merge(et_train, on='id')
            .merge(rf_train, on='id')
           # .merge(lr_train, on='id')
           #  .merge(lgbt_train, on='id')
           #  .merge(lgbt_train_1, on='id')
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
            .merge(xgb_test_7, on='id')
          .merge(nn_test, on='id')
          .merge(nn_test_1, on='id')
          .merge(nn_test_2, on='id')
          .merge(nn_test_4, on='id')
          .merge(nn_test_p1, on='id')
            .merge(nn_test_p2, on='id')
            .merge(nn_test_p3, on='id')
            .merge(nn_test_p4, on='id')
            .merge(nn_test_p5, on='id')
          .merge(et_test, on='id')
          .merge(rf_test, on='id')
          # .merge(lr_test, on='id')
          # .merge(lgbt_test, on='id')
          # .merge(lgbt_test_1, on='id')
          # .merge(knn_numeric_test, on='id')
          .drop('cat1', 1))

# xgb_coef = 1.006
# nn_coef = 1.019
#
# X_train['xgb_loss_0'] *= xgb_coef
# X_train['xgb_loss_1'] *= xgb_coef
# X_train['xgb_loss_2'] *= xgb_coef
# X_train['xgb_loss_3'] *= xgb_coef
# X_train['xgb_loss_4'] *= xgb_coef
# X_train['xgb_loss_5'] *= xgb_coef
# X_train['xgb_loss_6'] *= xgb_coef
# X_train['xgb_loss'] *= xgb_coef
#
#
# X_test['xgb_loss_0'] *= xgb_coef
# X_test['xgb_loss_1'] *= xgb_coef
# X_test['xgb_loss_2'] *= xgb_coef
# X_test['xgb_loss_3'] *= xgb_coef
# X_test['xgb_loss_4'] *= xgb_coef
# X_test['xgb_loss_5'] *= xgb_coef
# X_test['xgb_loss_6'] *= xgb_coef
# X_test['xgb_loss'] *= xgb_coef



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

print X_train.shape, xgb_train.shape, nn_train.shape

num_rounds = 300000
RANDOM_STATE = 2016


parameters = {'alpha': [0, 0.001, 0.005, 0.01, 0.1, 1, 10, 50, 100, 150, 180, 190, 200, 205, 210, 220, 240, 250, 260, 300, 500, 800, 1000]}

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

            pred += np.exp(clf.predict(x_te))
            pred_test += np.exp(clf.predict(X_test))

        pred /= nbags
        pred_oob[test_index] = pred
        score = mean_absolute_error(np.exp(y_te), pred)
        print('Fold ', i, '- MAE:', score)

    return pred_oob, pred_test

lr = Ridge(fit_intercept=False, alpha=0)

lr_oof_train, lr_oof_test = get_oof(lr)

print lr_oof_train.shape, X_train_id.shape

print("LR-CV: {}".format(mean_absolute_error(np.exp(y_train), lr_oof_train)))

oof_train = pd.DataFrame({'id': X_train_id, 'loss': (lr_oof_train - shift)})
oof_train.to_csv('oof2/lr_train.csv', index=False)

lr_oof_test /= n_folds

oof_test = pd.DataFrame({'id': X_test_id, 'loss': (lr_oof_test - shift)})
oof_test.to_csv('oof2/lr_test.csv', index=False)

