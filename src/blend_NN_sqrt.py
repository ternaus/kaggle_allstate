"""
Blending models NN
"""

from __future__ import division

import pandas as pd

import sys
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model, Sequential
from keras.layers import Input, BatchNormalization, Dropout, PReLU, MaxoutDense, Dense
from keras.optimizers import Adam, Nadam, Adadelta
from keras.regularizers import l1l2
from keras.layers.noise import GaussianNoise
from keras.layers.core import Activation

sys.path += ['/home/vladimir/packages/xgboost/python-package']

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from pylab import *
import clean_data

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

# X_train_mean = X_train.mean(axis=0)
#
# print X_train_mean
# print len(X_train_mean)

# X_train -= X_train.mean()

test_ids = test['id']

num_rounds = 3
RANDOM_STATE = 2016


def nn_model():
    model = Sequential()
    model.add(Dense(100, input_dim=X_train.shape[1], init='he_normal', activation='elu'))
    # model.add(Dropout(0.5))
    model.add(Dense(100, init='he_normal', activation='elu'))
    model.add(Dense(1, init='he_normal'))
    return model


def f_eval(y_true, y_pred):
    return K.mean(K.abs(y_pred**4 - y_true**4))

n_folds = 10
scores = []

classes = clean_data.classes(y_train, bins=100)

nbags = 10

pred_oob = np.zeros((X_train.shape[0], nbags))
pred_test = np.zeros((X_test.shape[0], n_folds * nbags))

print pred_test.shape

for i, (inTr, inTe) in enumerate(StratifiedKFold(n_folds, shuffle=True, random_state=RANDOM_STATE).split(classes, classes)):
    xtr = X_train[inTr]
    ytr = y_train[inTr]
    xte = X_train[inTe]
    yte = y_train[inTe]
    # pred = np.zeros(xte.shape[0])

    for j in range(nbags):
        model = nn_model()
        model.compile(loss='mae',
                      optimizer='adadelta',
                      # optimizer=Nadam(lr=1e-3),
                      metrics=[f_eval]
                      )
        callbacks = [
            ModelCheckpoint('keras_cache/keras-regressor-' + str(i + 1) + '.hdf5', monitor='val_loss',
                            save_best_only=True, verbose=0),
            EarlyStopping(patience=25, monitor='val_f_eval')
        ]
        model.fit(xtr, ytr,
                  validation_data=(xte, yte),
                  nb_epoch=2000,
                  callbacks=callbacks)

        model.load_weights('keras_cache/keras-regressor-' + str(i + 1) + '.hdf5')
        model.compile(loss='mae',
                      optimizer='adadelta',
                      # optimizer=Nadam(lr=1e-3),
                      metrics=[f_eval])

        pred_oob[inTe, j] = model.predict(xte)[:, 0]**4
        pred_test[:, j * n_folds + i - 1] = model.predict(X_test)[:, 0]**4


pred_oob_median = np.median(pred_oob, axis=1)
pred_oob_mean = np.mean(pred_oob, axis=1)

pred_test_median = np.median(pred_test, axis=1)
pred_test_mean = np.mean(pred_test, axis=1)


print('Total - MAE:', mean_absolute_error(y_train**4, pred_oob_median))
print('Total - MAE:', mean_absolute_error(y_train**4, pred_oob_mean))

# train predictions
df = pd.DataFrame({'id': X_train_id, 'loss': pred_oob_median})
df.to_csv('oof2/NN_train_s4_median.csv', index=False)

df = pd.DataFrame({'id': X_train_id, 'loss': pred_oob_mean})
df.to_csv('oof2/NN_train_s4_mean.csv', index=False)


df = pd.DataFrame({'id': X_test_id, 'loss': pred_test_median})
df.to_csv('oof2/NN_test_s4_median.csv', index=False)

df = pd.DataFrame({'id': X_test_id, 'loss': pred_test_mean})
df.to_csv('oof2/NN_test_s4_mean.csv', index=False)
