
'''
Author: Danijel Kivaranovic
Title: Neural network (Keras) with sparse data
'''

## import libraries
import numpy as np
np.random.seed(123)

import pandas as pd
import subprocess
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.optimizers import Nadam
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.layers.advanced_activations import PReLU
from tqdm import tqdm
import keras.backend as K

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
        X_batch = X[batch_index,:].toarray()
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
        if (counter == number_of_batches):
            counter = 0

########################################################################################################################################################

## read data
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

## set test loss to NaN
test['loss'] = np.nan

## response and IDs
shift = 200
y = np.log(train['loss'].values + shift)

y_mean = y.mean()

y = y - y_mean
id_train = train['id'].values
id_test = test['id'].values

## stack train test
ntrain = train.shape[0]
joined = pd.concat((train, test), axis=0)
joined = joined.drop(['cat110', 'cat116'], 1)

## Preprocessing and transforming to sparse data

cat_columns = joined.select_dtypes(include=['object']).columns

for column in list(cat_columns):
    if train[column].nunique() != test[column].nunique():
        # Let's find extra categories...
        set_train = set(train[column].unique())
        set_test = set(test[column].unique())
        remove_train = set_train - set_test
        remove_test = set_test - set_train

        remove = remove_train.union(remove_test)
        # print column, remove

        def filter_cat(x):
            if x in remove:
                return np.nan
            return x

        joined[column] = joined[column].apply(lambda x: filter_cat(x), 1)
        # print 'unique =', joined[column].nunique()

sparse_data = []

for f in tqdm(cat_columns):
    dummy = pd.get_dummies(joined[f].astype('category'))
    tmp = csr_matrix(dummy)
    sparse_data.append(tmp)


f_num = [f for f in joined.columns if 'cont' in f]
scaler = StandardScaler()
tmp = csr_matrix(scaler.fit_transform(joined[f_num]))
sparse_data.append(tmp)

del(joined, train, test)

## sparse train and test data
xtr_te = hstack(sparse_data, format='csr')
xtrain = xtr_te[:ntrain, :]
xtest = xtr_te[ntrain:, :]

print('Dim train', xtrain.shape)
print('Dim test', xtest.shape)

del(xtr_te, sparse_data, tmp)

# def nn_model():
#     model = Sequential()
#     model.add(Dense(400, init='he_normal', activation='elu', input_dim=xtrain.shape[1]))
#     model.add(Dropout(0.4))
#     # model.add(BatchNormalization())
#     model.add(Dense(200, init='he_normal', activation='elu'))
#     model.add(Dropout(0.4))
#     # model.add(BatchNormalization())
#     model.add(Dense(50, init='he_normal', activation='elu'))
#     model.add(Dropout(0.2))
#     model.add(Dense(1))
#     # model.compile(loss='mae', optimizer='nadam')
#     return(model)

def nn_model():
    model = Sequential()
    model.add(Dense(400, input_dim = xtrain.shape[1], init='he_normal', activation='elu'))
    # model.add(PReLU())
    model.add(Dropout(0.4))
    model.add(Dense(200, init='he_normal', activation='elu'))
    # model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(1, init='he_normal'))
    # model.compile(loss='mae', optimizer='adadelta')
    return(model)


## cv-folds
nfolds = 5
folds = KFold(len(y), n_folds=nfolds, shuffle=True, random_state=2016)

## train models
i = 0
nbags = 5
nepochs = 10
batch_size = 2**7
pred_oob = np.zeros(xtrain.shape[0])
pred_test = np.zeros(xtest.shape[0])

for (inTr, inTe) in folds:
    xtr = xtrain[inTr]
    ytr = y[inTr]
    xte = xtrain[inTe]
    yte = y[inTe]
    pred = np.zeros(xte.shape[0])
    for j in range(nbags):
        model = nn_model()
        model.compile(loss='mse', optimizer=Nadam(1e-4), metrics=[f_eval])
        fit = model.fit_generator(generator=batch_generator(xtr, ytr, batch_size, True),
                                  nb_epoch=nepochs,
                                  samples_per_epoch=10000 * batch_size,
                                  validation_data=batch_generator(xte, yte, batch_size, False),
                                  nb_val_samples=xte.shape[0])
        pred += model.predict_generator(generator=batch_generatorp(xte, 800, False), val_samples=xte.shape[0])[:,0]
        pred_test += model.predict_generator(generator=batch_generatorp(xtest, 800, False), val_samples=xtest.shape[0])[:, 0]
    pred /= nbags
    pred_oob[inTe] = pred
    score = mean_absolute_error(np.exp(yte + y_mean), np.exp(pred + y_mean))
    i += 1
    print('Fold ', i, '- MAE:', score)

print('Total - MAE:', mean_absolute_error(np.exp(y + y_mean), np.exp(pred_oob + y_mean)))

## train predictions
df = pd.DataFrame({'id': id_train, 'loss': np.exp(pred_oob + y_mean) - shift})
df.to_csv('preds_oob.csv', index = False)

## test predictions
pred_test /= (nfolds*nbags)
df = pd.DataFrame({'id': id_test, 'loss': np.exp(pred_test + y_mean) - shift})
df.to_csv('submission_keras.csv', index=False)




