from __future__ import division
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

def return_first_level_sqrt():
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

    nn_train_1 = pd.read_csv('oof/NN_train_p1.csv').rename(columns={'loss': 'nn_loss_1'})
    nn_test_1 = pd.read_csv('oof/NN_test_p1.csv').rename(columns={'loss': 'nn_loss_1'})

    nn_train_2 = pd.read_csv('oof/NN_train_p2.csv').rename(columns={'loss': 'nn_loss_2'})
    nn_test_2 = pd.read_csv('oof/NN_test_p2.csv').rename(columns={'loss': 'nn_loss_2'})

    nn_train_3 = pd.read_csv('oof/NN_train_p3.csv').rename(columns={'loss': 'nn_loss_3'})
    nn_test_3 = pd.read_csv('oof/NN_test_p3.csv').rename(columns={'loss': 'nn_loss_3'})

    nn_class_train = pd.read_csv('oof/NN_class_train.csv')
    nn_class_test = pd.read_csv('oof/NN_class_test.csv')

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
              .merge(xgb_test_s5, on='id')
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

    # X_test = np.hstack([X_test, nn_class_test.drop('id', 1).values])
    # X_train = np.hstack([X_train, nn_class_train.drop('id', 1).values])

    return X_train, y_train, X_test, X_train_id, X_test_id
