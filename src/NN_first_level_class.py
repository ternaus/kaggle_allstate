
'''
Author: Danijel Kivaranovic
Title: Neural network (Keras) with sparse data
'''

## import libraries
import numpy as np

np.random.seed(2016)

import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.optimizers import Nadam
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Convolution1D, MaxPooling1D
from keras.layers.advanced_activations import PReLU, ELU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm
import keras.backend as K
import clean_data
from sklearn.utils import shuffle


# Batch generators ##################################################################################################################################
def f_eval(y_true, y_pred):
    return K.mean(K.abs(K.pow(y_pred + y_mean, 4) - K.pow(y_true + y_mean, 4)))


def batch_generator(X, y, batch_size, shuffle):
    # changelog code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
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
        if counter == number_of_batches:
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
    model.add(Dense(500, activation='elu', init='he_normal', input_dim=xtrain.shape[1]))
    # model.add(PReLU())
    # model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(500, init='he_normal', activation='elu'))
    # model.add(PReLU())
    # model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(500, init='he_normal', activation='elu'))
    # model.add(PReLU())
    # model.add(BatchNormalization())
    model.add(Dense(num_classes, init='he_normal', activation='softmax'))
    return model

xtrain, y_train, xtest, y_mean, id_test, id_train = clean_data.one_hot_categorical_sqrt(subtract_mean=True, quadratic=False)

print xtrain.shape, y_train.shape

print y_train.min(), y_train.max(), y_train.mean()

xtrain = csr_matrix(xtrain)
xtest = csr_matrix(xtest)

# cv-folds
n_folds = 10

RANDOM_STATE = 2016

kf = StratifiedKFold(n_folds, shuffle=True, random_state=RANDOM_STATE)

num_classes = 10

classes = clean_data.classes(y_train, bins=num_classes)

# train models
i = 0
nbags = 2
nepochs = 1000
pred_oob = np.zeros((xtrain.shape[0], num_classes))
pred_test = np.zeros((xtest.shape[0], num_classes))


for i, (inTr, inTe) in enumerate(kf.split(classes, classes)):
    xtr = xtrain[inTr]
    ytr = classes[inTr]
    xte = xtrain[inTe]
    yte = classes[inTe]
    pred = np.zeros((xte.shape[0], num_classes))

    for j in range(nbags):
        xtr, ytr = shuffle(xtr, ytr, random_state=RANDOM_STATE)
        model = nn_model()
        model.compile(loss='sparse_categorical_crossentropy',
                      # optimizer='adadelta',
                      optimizer='nadam',
                      # optimizer=Nadam(lr=1e-2),
                      metrics=['accuracy']
                      )

        callbacks = [
            ModelCheckpoint('keras_cache/keras-regressor-' + str(i + 1) + str(j) + '.hdf5', monitor='val_loss',
                            save_best_only=True, verbose=0),
            EarlyStopping(patience=25, monitor='val_loss')
        ]

        model.fit_generator(generator=batch_generator(xtr, ytr, batch_size, True),
                            nb_epoch=nepochs,
                            samples_per_epoch=xtr.shape[0],
                            validation_data=batch_generator(xte, yte, batch_size, False),
                            nb_val_samples=xte.shape[0],
                                  callbacks=callbacks)

        model.load_weights('keras_cache/keras-regressor-' + str(i + 1) + str(j) + '.hdf5')

        model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adadelta',
                  # optimizer=Nadam(lr=1e-3),
                  metrics=['accuracy']
                  )

        pred += model.predict_generator(generator=batch_generatorp(xte, 800, False), val_samples=xte.shape[0])
        pred_test += model.predict_generator(generator=batch_generatorp(xtest, 800, False), val_samples=xtest.shape[0])

    pred /= nbags
    pred_oob[inTe, :] = pred

print pred_oob.shape
#train predictions

df = pd.DataFrame(pred_oob)
df['id'] = id_train

df.to_csv('oof/NN_class_train.csv', index=False)

# test predictions
pred_test /= (n_folds * nbags)

df = pd.DataFrame(pred_test)
df['id'] = id_test

df.to_csv('oof/NN_class_test.csv', index=False)

