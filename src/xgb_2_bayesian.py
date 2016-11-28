"""
Baysian hyperparameter optimization [https://github.com/fmfn/BayesianOptimization]
for Mean Absoulte Error objective
on default features for https://www.kaggle.com/c/allstate-claims-severity
"""

__author__ = "Vladimir Iglovikov"

import pandas as pd
from sklearn.metrics import mean_absolute_error
from bayes_opt import BayesianOptimization

import numpy as np

import os
import xgboost as xgb
from sklearn.utils import shuffle


def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(preds), np.exp(labels))




if __name__ == '__main__':
    num_rounds = 10000
    random_state = 2016
    num_iter = 100
    init_points = 100
    shift = 0

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
              # .merge(lgbt_test_1, on='id')
              # .merge(knn_numeric_test, on='id')
              .drop('cat1', 1))

    shift = 200

    y_train = np.log(X_train['loss'] + shift)
    print X_train.info()

    X_train = X_train.drop(['id', 'loss'], 1).applymap(lambda x: np.log(x + shift))
    X_test = X_test.drop('id', 1).applymap(lambda x: np.log(x + shift))

    xgtrain = xgb.DMatrix(X_train, label=y_train)


    params = {
        'eta': 0.1,
        'silent': 1,
        'eval_metric': 'mae',
        'verbose_eval': 1,
        'seed': random_state
    }

    # previous_points = pd.read_csv('params/parameters.csv')
    def xgb_evaluate(min_child_weight,
                     colsample_bytree,
                     max_depth,
                     subsample,
                     gamma,
                     ld,
                     alpha):

        params['min_child_weight'] = int(min_child_weight)
        params['cosample_bytree'] = max(min(colsample_bytree, 1), 0)
        params['max_depth'] = int(max_depth)
        params['subsample'] = max(min(subsample, 1), 0)
        params['gamma'] = max(gamma, 0)
        params['lambda'] = max(ld, 0)
        params['alpha'] = max(alpha, 0)

        cv_result = xgb.cv(params, xgtrain, num_boost_round=num_rounds, nfold=5, stratified=False,
                           early_stopping_rounds=50, verbose_eval=1, show_stdv=True, maximize=False, feval=evalerror)

        return -cv_result['test-mae-mean'].values[-1]


    xgbBO = BayesianOptimization(xgb_evaluate, {'min_child_weight': (1, 200),
                                                'colsample_bytree': (0.1, 1),
                                                'max_depth': (3, 20),
                                                'subsample': (0.1, 1),
                                                'gamma': (0, 3),
                                                'ld': (0, 10),
                                                'alpha': (0, 10),
                                                })

    # xgbBO.initialize_df(previous_points)

    xgbBO.maximize(init_points=init_points, n_iter=num_iter)

    # Save results
    param_cache_path = 'params'
    try:
        os.mkdir(param_cache_path)
    except:
        pass

    file_name = 'params/parameters_xgb2.csv'
    xgbBO.points_to_csv(file_name)