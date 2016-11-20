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

nn_train = pd.read_csv('oof/NN_train.csv').rename(columns={'loss': 'nn_loss'})
nn_test = pd.read_csv('oof/NN_test.csv').rename(columns={'loss': 'nn_loss'})

nn_train_1 = pd.read_csv('oof/NN_train_1.csv').rename(columns={'loss': 'nn_loss_1'})
nn_test_1 = pd.read_csv('oof/NN_test_1.csv').rename(columns={'loss': 'nn_loss_1'})

nn_train_2 = pd.read_csv('oof/NN_train_2.csv').rename(columns={'loss': 'nn_loss_1'})
nn_test_2 = pd.read_csv('oof/NN_test_2.csv').rename(columns={'loss': 'nn_loss_1'})

nn_train_4 = pd.read_csv('oof/NN_train_4.csv').rename(columns={'loss': 'nn_loss_4'})
nn_test_4 = pd.read_csv('oof/NN_test_4.csv').rename(columns={'loss': 'nn_loss_4'})


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
           .merge(nn_train, on='id')
           .merge(nn_train_1, on='id')
           .merge(nn_train_2, on='id')
            .merge(nn_train_4, on='id')
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
          .merge(nn_test, on='id')
          .merge(nn_test_1, on='id')
          .merge(nn_test_2, on='id')
          .merge(nn_test_4, on='id')
          .merge(et_test, on='id')
          .merge(rf_test, on='id')
          # .merge(lr_test, on='id')
          # .merge(lgbt_test, on='id')
          # .merge(lgbt_test_1, on='id')
          # .merge(knn_numeric_test, on='id')
          .drop('cat1', 1))

shift = 200

y_train = np.log(X_train['loss'] + shift)

X_train_id = X_train['id'].values
X_test_id = X_test['id'].values


X_train = X_train.drop(['id', 'loss'], 1).applymap(lambda x: np.log(x + shift)).values
X_test = X_test.drop('id', 1).applymap(lambda x: np.log(x + shift)).values

X_train_mean = X_train.mean(axis=0)

print X_train_mean
# print len(X_train_mean)

# X_train -= X_train.mean()

test_ids = test['id']

print X_train.shape, xgb_train.shape, nn_train.shape

num_rounds = 300000
RANDOM_STATE = 2016


def nn_model():
    model = Sequential()
    model.add(Dense(5, input_dim=X_train.shape[1], init='he_normal', activation='elu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(100, init='he_normal', activation='elu'))
    model.add(Dense(1, init='he_normal'))
    return model


def f_eval(y_true, y_pred):
    return K.mean(K.abs(K.exp(y_pred) - K.exp(y_true)))

n_folds = 5
scores = []

classes = clean_data.classes(y_train, bins=100)

def eval_f(x, y):
    return mean_absolute_error(np.exp(x), np.exp(y))

pred_oob = np.zeros(X_train.shape[0])
pred_test = np.zeros(X_test.shape[0])

nbags = 1


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
            EarlyStopping(patience=15)
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
df.to_csv('oof2/NN_train.csv', index=False)

# test predictions
pred_test /= (n_folds * nbags)
df = pd.DataFrame({'id': X_test_id, 'loss': pred_test - shift})
df.to_csv('oof2/NN_test.csv', index=False)
