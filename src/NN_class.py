import numpy as np
np.random.seed(123)

import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from keras.layers import Dense
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import ShuffleSplit
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras.layers import Input, BatchNormalization, Dropout
from keras.optimizers import Adam, Nadam
from keras.utils import np_utils

# def f_eval(y_true, y_pred):
#     return K.mean(K.abs(K.exp(y_pred + y_mean) - K.exp(y_true + y_mean)))


def batch_generator(X, y, batch_size, shuffle):
    # changelog code for fitting from generator
    # (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index, :].toarray()
        y_batch = y[batch_index, :]
        counter += 1
        yield X_batch, y_batch
        if counter == number_of_batches:
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0


def batch_generatorp(X, batch_size):
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

# read data
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

# set test loss to NaN
test['loss'] = np.nan

# response and IDs
shift = 200
y = np.log(train['loss'].values + shift)
# y = train['loss'].values
# y_mean = y.mean()
# y = y - y_mean

num_classes = 20

a, b = np.histogram(y, bins=num_classes - 1)

y = np.searchsorted(b, y)

id_train = train['id'].values
id_test = test['id'].values

# stack train test
num_train = train.shape[0]
joined = pd.concat((train, test), axis=0)

for column in list(train.select_dtypes(include=['object']).columns):
    # g = train.groupby(column)['loss'].mean()
    # g = train.groupby(column)['loss'].median()

    if train[column].nunique() != test[column].nunique():
        # Let's find extra categories...
        set_train = set(train[column].unique())
        set_test = set(test[column].unique())
        remove_train = set_train - set_test
        remove_test = set_test - set_train

        remove = remove_train.union(remove_test)
        print column, remove

        def filter_cat(x):
            if x in remove:
                return np.nan
            return x

        joined[column] = joined[column].apply(lambda x: filter_cat(x), 1)
        print 'unique =', joined[column].nunique()

    joined[column] = pd.factorize(joined[column].values, sort=True)[0]

# Preprocessing and transforming to sparse data
sparse_data = []

cat_columns = [f for f in joined.columns if 'cat' in f]
for f in tqdm(cat_columns):

    dummy = pd.get_dummies(joined[f].astype('category'))
    tmp = csr_matrix(dummy)
    sparse_data.append(tmp)

num_columns = [f for f in joined.columns if 'cont' in f]
scaler = StandardScaler()
tmp = csr_matrix(scaler.fit_transform(joined[num_columns]))
sparse_data.append(tmp)

del(joined, train, test)

# sparse train and test data
xtr_te = hstack(sparse_data, format='csr')
xtrain = xtr_te[:num_train, :]
xtest = xtr_te[num_train:, :]

print('Dim train', xtrain.shape)
print('Dim test', xtest.shape)

del(xtr_te, sparse_data, tmp)

RANDOM_STATE = 2016


# neural net
def nn_model():
    inputs = Input(shape=(xtrain.shape[1], ))
    # x = Dropout(0.5)(inputs)
    x = Dense(400, init='he_normal', activation='elu', name='l1')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(400, init='he_normal', activation='elu', name='l2')(x)
    # x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(400, init='he_normal', activation='elu', name='l3')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax', name='output')(x)
    model = Model(input=inputs, output=x)

    model.compile(loss='categorical_crossentropy',
                  optimizer=Nadam(lr=1e-4),
                  )
    return model

sss = ShuffleSplit(n_splits=10, test_size=0.2, random_state=RANDOM_STATE)

train_index, test_index = sss.split(xtrain, y).next()

X_train = xtrain[train_index]
y_train = y[train_index]
X_val = xtrain[test_index]
y_val = y[test_index]


Y_train = np_utils.to_categorical(y_train, num_classes)
Y_test = np_utils.to_categorical(y_val, num_classes)

num_epoch = 1000

model = nn_model()
fit = model.fit_generator(generator=batch_generator(X_train, y_train, 256, True),
                          verbose=1,
                          validation_data=batch_generator(X_val, y_val, 256, False),
                          nb_val_samples=X_val.shape[0],
                          nb_epoch=num_epoch,
                          samples_per_epoch=X_train.shape[0],
                          callbacks=[EarlyStopping(patience=100)]
                          )

# val_prediction = model.predict_generator(batch_generatorp(X_val, 256), X_val.shape[0])
#
# print 'val_accuracy = ', mean_absolute_error(np.exp(y_val + y_mean), np.exp(val_prediction + y_mean))

#
# # train models
# i = 0
# nbags = 5
# num_epoch = 200
# pred_oob = np.zeros(X_train.shape[0])
# pred_test = np.zeros(X_test.shape[0])
#
#
#     X_train = X_train[train_index]
#     y_train = y[train_index]
#     X_val = X_train[test_index]
#     y_val = y[test_index]
#     pred_val = np.zeros(X_val.shape[0])
#     for j in range(nbags):
#         model = nn_model()
#         fit = model.fit_generator(generator=batch_generator(X_train, y_train, 128, True),
#                                   nb_epoch=num_epoch,
#                                   samples_per_epoch=X_train.shape[0])
#         pred_val += model.predict_generator(generator=batch_generatorp(X_val, 800, False), val_samples=X_val.shape[0])[:,0]
#         pred_test += model.predict_generator(generator=batch_generatorp(X_test, 800, False), val_samples=X_test.shape[0])[:,0]
#     pred_val /= nbags
#     pred_oob[test_index] = pred_val
#     score = mean_absolute_error(np.exp(y_val), np.exp(pred_val))
#     i += 1
#     print('Fold ', i, '- MAE:', score)
#
# print('Total - MAE:', mean_absolute_error(y, pred_oob))
#
# ## train predictions
# df = pd.DataFrame({'id': id_train, 'loss': np.exp(pred_oob + y_mean)})
# df.to_csv('preds_oob.csv', index = False)
#
# ## test predictions
# pred_test /= (nfolds*nbags)
# df = pd.DataFrame({'id': id_test, 'loss': np.exp(pred_test + y_mean)})
# df.to_csv('submission_keras.csv', index=False)
#
#
#
#
