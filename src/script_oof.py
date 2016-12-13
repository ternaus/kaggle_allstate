
'''
Author: Danijel Kivaranovic
Title: Neural network (Keras) with sparse data
'''

## import libraries
import numpy as np

np.random.seed(123)

import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.optimizers import Nadam
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from tqdm import tqdm
import keras.backend as K
import clean_data


## Batch generators ##################################################################################################################################
def f_eval(y_true, y_pred):
    return K.mean(K.abs(K.exp(y_pred + y_mean) - K.exp(y_true + y_mean)))

def nn_model():
    model = Sequential()
    model.add(Dense(400, init='he_normal', activation='elu', input_dim=xtrain.shape[1]))
    # model.add(PReLU())
    model.add(Dropout(0.4))
    # model.add(BatchNormalization())
    model.add(Dense(200, init='he_normal', activation='elu'))
    # model.add(PReLU())
    model.add(Dropout(0.4))
    # model.add(BatchNormalization())
    model.add(Dense(50, init='he_normal', activation='elu'))
    # model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(1, init='he_normal'))
    return model

shift = 0
# xtrain, y_train, xtest, y_mean, id_test, id_train = clean_data.one_hot_categorical(shift=shift, subtract_mean=True)


# print xtrain.shape
# print y_train
xtrain, y, xtest, y_mean, id_test, id_train = clean_data.oof_categorical(shift=0, scale=True, subtract_min=True)

# print pd.DataFrame(xtrain.todense()).describe()
print y.min(), y.max(), y.mean()

xtrain = csr_matrix(xtrain)
xtest = csr_matrix(xtest)

## cv-folds
nfolds = 5
folds = KFold(len(y), n_folds=nfolds, shuffle=True, random_state=2016)

## train models
i = 0
nbags = 5
nepochs = 25
batch_size = 2**7
pred_oob = np.zeros(xtrain.shape[0])
pred_test = np.zeros(xtest.shape[0])

for i, (inTr, inTe) in enumerate(folds):
    xtr = xtrain[inTr]
    ytr = y[inTr]
    xte = xtrain[inTe]
    yte = y[inTe]
    pred = np.zeros(xte.shape[0])
    for j in range(nbags):
        model = nn_model()
        # model.compile(loss='mse', optimizer=Nadam(1e-5), metrics=[f_eval])
        model.compile(loss='mae', optimizer='adadelta', metrics=[f_eval])
        fit = model.fit(xtr, ytr, validation_data=(xte, yte), batch_size=batch_size, nb_epoch=nepochs)
        # pred += model.predict_generator(generator=batch_generatorp(xte, 800, False), val_samples=xte.shape[0])[:,0]
        # pred_test += model.predict_generator(generator=batch_generatorp(xtest, 800, False), val_samples=xtest.shape[0])[:, 0]
#     pred /= nbags
#     pred_oob[inTe] = pred
#     score = mean_absolute_error(np.exp(yte + y_mean), np.exp(pred + y_mean))
#     print('Fold ', i, '- MAE:', score)
#
# print('Total - MAE:', mean_absolute_error(np.exp(y_train + y_mean), np.exp(pred_oob + y_mean)))
#
# # train predictions
# df = pd.DataFrame({'id': id_train, 'loss': np.exp(pred_oob + y_mean) - shift})
# df.to_csv('preds_oob.csv', index=False)
#
# # test predictions
# pred_test /= (nfolds*nbags)
# df = pd.DataFrame({'id': id_test, 'loss': np.exp(pred_test + y_mean) - shift})
# df.to_csv('submission_keras.csv', index=False)
#
#
#
#
