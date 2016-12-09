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

from sklearn.metrics import make_scorer
from pylab import *
import clean_data
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsRegressor

def eval_f(x, y):
    return mean_absolute_error(x**4, y**4)


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

xgb_train_s5 = pd.read_csv('oof/xgb_train_s5.csv').rename(columns={'loss': 'xgb_loss_s5'})
xgb_test_s5 = pd.read_csv('oof/xgb_test_s5.csv').rename(columns={'loss': 'xgb_loss_s5'})

xgb_train_s6 = pd.read_csv('oof/xgb_train_s6.csv').rename(columns={'loss': 'xgb_loss_s6'})
xgb_test_s6 = pd.read_csv('oof/xgb_test_s6.csv').rename(columns={'loss': 'xgb_loss_s6'})

xgb_train_s7 = pd.read_csv('oof/xgb_train_s7.csv').rename(columns={'loss': 'xgb_loss_s7'})
xgb_test_s7 = pd.read_csv('oof/xgb_test_s7.csv').rename(columns={'loss': 'xgb_loss_s7'})

# xgb_train_s8 = pd.read_csv('oof/xgb_train_s8.csv').rename(columns={'loss': 'xgb_loss_s8'})
# xgb_test_s8 = pd.read_csv('oof/xgb_test_s8.csv').rename(columns={'loss': 'xgb_loss_s8'})

xgb_train_t1 = pd.read_csv('oof/xgb_train_t1.csv').rename(columns={'loss': 'xgb_loss_t1'})
xgb_test_t1 = pd.read_csv('oof/xgb_test_t1.csv').rename(columns={'loss': 'xgb_loss_t1'})

xgb_train_t2 = pd.read_csv('oof/xgb_train_t2.csv').rename(columns={'loss': 'xgb_loss_t2'})
xgb_test_t2 = pd.read_csv('oof/xgb_test_t2.csv').rename(columns={'loss': 'xgb_loss_t2'})

xgb_train_t3 = pd.read_csv('oof/xgb_train_t3.csv').rename(columns={'loss': 'xgb_loss_t3'})
xgb_test_t3 = pd.read_csv('oof/xgb_test_t3.csv').rename(columns={'loss': 'xgb_loss_t3'})

xgb_train_t4 = pd.read_csv('oof/xgb_train_t4.csv').rename(columns={'loss': 'xgb_loss_t4'})
xgb_test_t4 = pd.read_csv('oof/xgb_test_t4.csv').rename(columns={'loss': 'xgb_loss_t4'})

xgb_train_t5 = pd.read_csv('oof/xgb_train_t5.csv').rename(columns={'loss': 'xgb_loss_t5'})
xgb_test_t5 = pd.read_csv('oof/xgb_test_t5.csv').rename(columns={'loss': 'xgb_loss_t5'})

xgb_train_t6 = pd.read_csv('oof/xgb_train_t6.csv').rename(columns={'loss': 'xgb_loss_t6'})
xgb_test_t6 = pd.read_csv('oof/xgb_test_t6.csv').rename(columns={'loss': 'xgb_loss_t6'})

xgb_train_t7 = pd.read_csv('oof/xgb_train_t7.csv').rename(columns={'loss': 'xgb_loss_t7'})
xgb_test_t7 = pd.read_csv('oof/xgb_test_t7.csv').rename(columns={'loss': 'xgb_loss_t7'})

nn_train_1 = pd.read_csv('oof/NN_train_p1.csv').rename(columns={'loss': 'nn_loss_1'})
nn_test_1 = pd.read_csv('oof/NN_test_p1.csv').rename(columns={'loss': 'nn_loss_1'})

nn_train_2 = pd.read_csv('oof/NN_train_p2.csv').rename(columns={'loss': 'nn_loss_2'})
nn_test_2 = pd.read_csv('oof/NN_test_p2.csv').rename(columns={'loss': 'nn_loss_2'})

nn_train_3 = pd.read_csv('oof/NN_train_p3.csv').rename(columns={'loss': 'nn_loss_3'})
nn_test_3 = pd.read_csv('oof/NN_test_p3.csv').rename(columns={'loss': 'nn_loss_3'})

nn_train_4 = pd.read_csv('oof/NN_train_p4.csv').rename(columns={'loss': 'nn_loss_4'})
nn_test_4 = pd.read_csv('oof/NN_test_p4.csv').rename(columns={'loss': 'nn_loss_4'})

nn_train_5 = pd.read_csv('oof/NN_train_p5.csv').rename(columns={'loss': 'nn_loss_5'})
nn_test_5 = pd.read_csv('oof/NN_test_p5.csv').rename(columns={'loss': 'nn_loss_5'})

nn_train_6 = pd.read_csv('oof/NN_train_p6.csv').rename(columns={'loss': 'nn_loss_6'})
nn_test_6 = pd.read_csv('oof/NN_test_p6.csv').rename(columns={'loss': 'nn_loss_6'})

# nn_class_train = pd.read_csv('oof/NN_class_train.csv')
# nn_class_test = pd.read_csv('oof/NN_class_test.csv')

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
           .merge(xgb_train_s5, on='id')
           .merge(xgb_train_s6, on='id')
           .merge(xgb_train_s7, on='id')
           # .merge(xgb_train_s8, on='id')
           .merge(xgb_train_t1, on='id')
           .merge(xgb_train_t2, on='id')
           .merge(xgb_train_t3, on='id')
           .merge(xgb_train_t4, on='id')
           .merge(xgb_train_t5, on='id')
           .merge(xgb_train_t6, on='id')
           .merge(xgb_train_t7, on='id')
           .merge(nn_train_1, on='id')
           .merge(nn_train_2, on='id')
           .merge(nn_train_3, on='id')
           .merge(nn_train_4, on='id')
           .merge(nn_train_5, on='id')
           .merge(nn_train_6, on='id')
           .merge(lgbt_train_1, on='id')
           .merge(et_train_1, on='id')
           .merge(et_train_s1, on='id')
           .merge(rf_train_s1, on='id')
            # .merge(nn_class_train, on='id')
           )

X_test = (test[['id', 'cat1']]
          .merge(xgb_test_1, on='id')
          .merge(xgb_test_s1, on='id')
          .merge(xgb_test_s2, on='id')
          .merge(xgb_test_s3, on='id')
          .merge(xgb_test_s4, on='id')
          .merge(xgb_test_s5, on='id')
          .merge(xgb_test_s6, on='id')
          .merge(xgb_test_s7, on='id')
          # .merge(xgb_test_s8, on='id')
          .merge(xgb_test_t1, on='id')
          .merge(xgb_test_t2, on='id')
          .merge(xgb_test_t3, on='id')
          .merge(xgb_test_t4, on='id')
          .merge(xgb_test_t5, on='id')
          .merge(xgb_test_t6, on='id')
          .merge(xgb_test_t7, on='id')
          .merge(nn_test_1, on='id')
          .merge(nn_test_2, on='id')
          .merge(nn_test_3, on='id')
          .merge(nn_test_4, on='id')
          .merge(nn_test_5, on='id')
          .merge(nn_test_6, on='id')
          .merge(lgbt_test_1, on='id')
          .merge(et_test_1, on='id')
          .merge(et_test_s1, on='id')
          .merge(rf_test_s1, on='id')
          # .merge(nn_class_test, on='id')
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


parameters = {'n_neighbors': [20], 'p': [2]}

n_folds = 10

kf = StratifiedKFold(n_folds, shuffle=True, random_state=RANDOM_STATE)
classes = clean_data.classes(y_train, bins=100)

kNN = KNeighborsRegressor(n_jobs=-1)

clf = GridSearchCV(kNN, parameters, cv=kf.get_n_splits(classes, classes), scoring=make_scorer(eval_f),
                   iid=False, verbose=True)

clf.fit(X_train, y_train)

for i in clf.grid_scores_:
    print i
#
# num_train = X_train.shape[0]
# num_test = X_test.shape[0]
#
# classes = clean_data.classes(y_train, bins=100)
#
#
# nbags = 1
#
#
# def get_oof(clf):
#     pred_oob = np.zeros(X_train.shape[0])
#     pred_test = np.zeros(X_test.shape[0])
#
#     for i, (train_index, test_index) in enumerate(kf.split(classes, classes)):
#         print "Fold = ", i
#         x_tr = X_train[train_index]
#         y_tr = y_train[train_index]
#
#         x_te = X_train[test_index]
#         y_te = y_train[test_index]
#
#         pred = np.zeros(x_te.shape[0])
#
#         for j in range(nbags):
#             # x_tr, y_tr = shuffle(x_tr, y_tr, random_state=RANDOM_STATE + i + j)
#             clf.fit(x_tr, y_tr)
#             print clf.coef_
#
#             pred += clf.predict(x_te)**4
#             pred_test += clf.predict(X_test)**4
#
#         pred /= nbags
#         pred_oob[test_index] = pred
#         score = mean_absolute_error(y_te**4, pred)
#         print('Fold ', i, '- MAE:', score)
#
#     return pred_oob, pred_test
#
# svr = LinearSVR(C=0.016681005372000592)
# # svr = Ridge()
#
# svr_oof_train, svr_oof_test = get_oof(svr)
#
# print("SVR-CV: {}".format(mean_absolute_error(y_train**4, svr_oof_train)))
#
# oof_train = pd.DataFrame({'id': X_train_id, 'loss': svr_oof_train})
# oof_train.to_csv('oof2/svr_train.csv', index=False)
#
# svr_oof_test /= n_folds
#
# oof_test = pd.DataFrame({'id': X_test_id, 'loss': svr_oof_test})
# oof_test.to_csv('oof2/svr_test.csv', index=False)
#
