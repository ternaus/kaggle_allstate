
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
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Convolution1D, MaxPooling1D
from keras.layers.advanced_activations import PReLU, ELU

from tqdm import tqdm
import keras.backend as K
import clean_data


## Batch generators ##################################################################################################################################
def f_eval(y_true, y_pred):
    return K.mean(K.abs(K.exp(y_pred + y_mean) - K.exp(y_true + y_mean)))


def batch_generator(X, y, batch_size, shuffle):
    #chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index, :].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0


def batch_generatorp(X, batch_size, shuffle):
    number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :].toarray()
        counter += 1
        yield X_batch
        if counter == number_of_batches:
            counter = 0


def nn_model():
    model = Sequential()
    model.add(Dense(400, init='he_normal', activation='elu', input_dim=xtrain.shape[1]))
    model.add(Dropout(0.4))
    # model.add(BatchNormalization())
    model.add(Dense(200, init='he_normal', activation='elu'))
    model.add(Dropout(0.4))
    # model.add(BatchNormalization())
    model.add(Dense(50, init='he_normal', activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    # model.compile(loss='mae', optimizer='nadam')
    return(model)

# def nn_model():
#     model = Sequential()
#     model.add(Dense(300, init='he_normal', input_dim=xtrain.shape[1]))
#     # model.add(BatchNormalization())
#     model.add(ELU())
#     model.add(Dropout(0.4))
#     model.add(Dense(100, init='he_normal'))
#     # model.add(BatchNormalization())
#     model.add(ELU())
#     model.add(Dropout(0.4))
#     model.add(Dense(50, init='he_normal', activation='elu'))
#     model.add(Dropout(0.2))
#     model.add(Dense(1, init='he_normal'))
#     return model

shift = 0
xtrain, y, xtest, y_mean, id_test, id_train = clean_data.one_hot_categorical(shift=shift, subtract_mean=True)


# print xtrain.shape
# print y
# xtrain, y, xtest, y_mean, id_test, id_train = clean_data.oof_categorical(shift=0, scale=True, subtract_min=True)

print xtrain.shape, y.shape
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
nepochs = 20
batch_size = 2**6
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
        fit = model.fit_generator(generator=batch_generator(xtr, ytr, batch_size, True),
                                  nb_epoch=nepochs,
                                  samples_per_epoch=10000 * batch_size,
                                  validation_data=batch_generator(xte, yte, batch_size, False),
                                  nb_val_samples=xte.shape[0])
        pred += np.exp(model.predict_generator(generator=batch_generatorp(xte, 800, False), val_samples=xte.shape[0])[:, 0] + y_mean)
        pred_test += np.exp(model.predict_generator(generator=batch_generatorp(xtest, 800, False), val_samples=xtest.shape[0])[:, 0] + y_mean)
    pred /= nbags
    pred_oob[inTe] = pred
    score = mean_absolute_error(np.exp(yte + y_mean), pred + y_mean)
    print('Fold ', i, '- MAE:', score)

print('Total - MAE:', mean_absolute_error(np.exp(y + y_mean), pred_oob + y_mean))

# train predictions
df = pd.DataFrame({'id': id_train, 'loss': pred_oob - shift})
df.to_csv('preds_oob.csv', index=False)

# test predictions
pred_test /= (nfolds*nbags)
df = pd.DataFrame({'id': id_test, 'loss': pred_test - shift})
df.to_csv('submission_keras.csv', index=False)




