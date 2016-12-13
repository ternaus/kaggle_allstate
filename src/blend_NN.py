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
from sklearn.model_selection import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from pylab import *
import clean_data


def eval_f(x, y):
    return mean_absolute_error(np.exp(x), np.exp(y))

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

xgb_train_1 = pd.read_csv('oof/xgb_train_t1.csv').rename(columns={'loss': 'xgb_loss_1'})
xgb_test_1 = pd.read_csv('oof/xgb_test_t1.csv').rename(columns={'loss': 'xgb_loss_1'})

xgb_train_s1 = pd.read_csv('oof/xgb_train_s1.csv').rename(columns={'loss': 'xgb_loss_s1'})
xgb_test_s1 = pd.read_csv('oof/xgb_test_s1.csv').rename(columns={'loss': 'xgb_loss_s1'})

nn_train_1 = pd.read_csv('oof/NN_train_p1.csv').rename(columns={'loss': 'nn_loss_1'})
nn_test_1 = pd.read_csv('oof/NN_test_p1.csv').rename(columns={'loss': 'nn_loss_1'})

nn_train_2 = pd.read_csv('oof/NN_train_p2.csv').rename(columns={'loss': 'nn_loss_2'})
nn_test_2 = pd.read_csv('oof/NN_test_p2.csv').rename(columns={'loss': 'nn_loss_2'})

nn_train_3 = pd.read_csv('oof/NN_train_p3.csv').rename(columns={'loss': 'nn_loss_3'})
nn_test_3 = pd.read_csv('oof/NN_test_p3.csv').rename(columns={'loss': 'nn_loss_3'})

lgbt_train_1 = pd.read_csv('oof/lgbt_train_1.csv').rename(columns={'loss': 'lgbt_loss_1'})
lgbt_test_1 = pd.read_csv('oof/lgbt_test_1.csv').rename(columns={'loss': 'lgbt_loss_1'})


X_train = (train[['id', 'loss']]
           .merge(xgb_train_1, on='id')
           .merge(xgb_train_s1, on='id')
           .merge(nn_train_1, on='id')
           .merge(nn_train_2, on='id')
            .merge(nn_train_3, on='id')
            .merge(lgbt_train_1, on='id')
           )

X_test = (test[['id', 'cat1']]
          .merge(xgb_test_1, on='id')
          .merge(xgb_test_s1, on='id')
          .merge(nn_test_1, on='id')
          .merge(nn_test_2, on='id')
          .merge(nn_test_3, on='id')
          .merge(lgbt_test_1, on='id')
          .drop('cat1', 1))


shift = 200

y_train = np.log(X_train['loss'] + shift)

X_train_id = X_train['id'].values
X_test_id = X_test['id'].values


X_train = X_train.drop(['id', 'loss'], 1).applymap(lambda x: np.log(x + shift)).values
X_test = X_test.drop('id', 1).applymap(lambda x: np.log(x + shift)).values

# X_train_mean = X_train.mean(axis=0)
#
# print X_train_mean
# print len(X_train_mean)

# X_train -= X_train.mean()

test_ids = test['id']

num_rounds = 300000
RANDOM_STATE = 2016


def nn_model():
    model = Sequential()
    model.add(Dense(10, input_dim=X_train.shape[1], init='he_normal', activation='elu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(100, init='he_normal', activation='elu'))
    model.add(Dense(1, init='he_normal'))
    return model


def f_eval(y_true, y_pred):
    return K.mean(K.abs(K.exp(y_pred) - K.exp(y_true)))

n_folds = 10
scores = []

classes = clean_data.classes(y_train, bins=100)


def eval_f(x, y):
    return mean_absolute_error(np.exp(x), np.exp(y))

pred_oob = np.zeros(X_train.shape[0])
pred_test = np.zeros(X_test.shape[0])

nbags = 5


for i, (inTr, inTe) in enumerate(KFold(n_folds, shuffle=True, random_state=RANDOM_STATE).split(classes, classes)):
    xtr = X_train[inTr]
    ytr = y_train[inTr]
    xte = X_train[inTe]
    yte = y_train[inTe]
    pred = np.zeros(xte.shape[0])

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
            EarlyStopping(patience=15, monitor='val_f_eval')
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
        pred += np.exp(model.predict(xte)[:, 0])

        pred_test += np.exp(model.predict(X_test)[:, 0])

    pred /= nbags
    pred_oob[inTe] = pred
    score = mean_absolute_error(np.exp(yte), pred)
    print('Fold ', i, '- MAE:', score)

print('Total - MAE:', mean_absolute_error(np.exp(y_train), pred_oob))

# train predictions
df = pd.DataFrame({'id': X_train_id, 'loss': pred_oob - shift})
df.to_csv('oof2/NN_train_1.csv', index=False)

# test predictions
pred_test /= (n_folds * nbags)
df = pd.DataFrame({'id': X_test_id, 'loss': pred_test - shift})
df.to_csv('oof2/NN_test_1.csv', index=False)
