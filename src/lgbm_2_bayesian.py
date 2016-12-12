"""
Baysian hyperparameter optimization [https://github.com/fmfn/BayesianOptimization]
for Mean Absoulte Error objective
on default features for https://www.kaggle.com/c/allstate-claims-severity
"""
from __future__ import division

__author__ = "Vladimir Iglovikov"

from bayes_opt import BayesianOptimization
import pandas as pd

from lightgbm.sklearn import LGBMRegressor
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import mean_absolute_error
from pylab import *
import clean_data


def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    con = 2
    x = preds - labels
    den = np.abs(x) + con
    grad = con * x / den
    hess = con**2 / den**2
    return grad, hess


def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'mae', mean_absolute_error(preds**4, labels**4), False


if __name__ == '__main__':
    num_rounds = 10000
    random_state = 2016
    num_iter = 100
    init_points = 10000

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
    RANDOM_STATE = 2016

    n_folds = 10


    classes = clean_data.classes(y_train, bins=100)

    def lgbm_evaluate(min_child_weight,
                     colsample_bytree,
                     subsample,
                      num_leaves
                     ):


        scores = []


        nbags = 1

        for i, (inTr, inTe) in enumerate(StratifiedKFold(n_folds, shuffle=True, random_state=RANDOM_STATE).split(classes, classes)):

            xtr = X_train[inTr]
            ytr = y_train[inTr]
            xte = X_train[inTe]
            yte = y_train[inTe]

            for j in range(nbags):

                lgbm = LGBMRegressor(num_leaves=int(num_leaves),
                                     n_estimators=10000,
                                     subsample=subsample,
                                     colsample_bytree=colsample_bytree,
                                     min_child_weight=int(min_child_weight))

                lgbm.fit(xtr, ytr, early_stopping_rounds=50, eval_set=[(xte, yte)], eval_metric=evalerror, verbose=False)

                scores += [mean_absolute_error(yte**4, lgbm.predict(xte, lgbm.best_iteration)**4)]

        return -np.mean(scores)


    lgbmBO = BayesianOptimization(lgbm_evaluate, {'min_child_weight': (1, 200),
                                                'colsample_bytree': (0.1, 1),
                                                'subsample': (0.1, 1),
                                                'num_leaves': (100, 300)
                                                })

    lgbmBO.maximize(init_points=init_points, n_iter=num_iter)

