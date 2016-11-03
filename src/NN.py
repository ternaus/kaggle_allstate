import numpy as np
np.random.seed(123)

import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from keras.layers import Dense, merge, ELU
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import ShuffleSplit
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.models import Model, Sequential
from keras.layers import Input, BatchNormalization, Dropout, PReLU, MaxoutDense
from keras.optimizers import Adam, Nadam, Adadelta
from keras.regularizers import l1l2
from keras.layers.noise import GaussianNoise
from keras.layers.core import Activation

def f_eval(y_true, y_pred):
    return K.mean(K.abs(K.exp(y_std * y_pred + y_mean) - K.exp(y_std * y_true + y_mean)))


def batch_generator(X, y, batch_size, shuffle):
    # changelog code for fitting from generator
    # (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    number_of_batches = np.ceil(X.shape[0] / batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter+1)]
        X_batch = X[batch_index, :].toarray()
        y_batch = y[batch_index]
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

y_mean = y.mean()

y = y - y_mean

y_std = np.std(y)

y /= y_std
id_train = train['id'].values
id_test = test['id'].values

# stack train test
num_train = train.shape[0]
joined = pd.concat((train, test), axis=0)
#
# joined['cont2'] = joined['cont2'].astype(str)
# joined['cont3'] = joined['cont3'].astype(str)
# joined['cont4'] = joined['cont4'].astype(str)
# joined['cont5'] = joined['cont5'].astype(str)
# joined['cont9'] = joined['cont9'].astype(str)
# joined['cont10'] = joined['cont10'].astype(str)
# joined['cont11'] = joined['cont11'].astype(str)
# joined['cont12'] = joined['cont12'].astype(str)
# joined['cont13'] = joined['cont13'].astype(str)

joined = joined.drop(['cat110', 'cat116'], 1)

cat_columns = joined.select_dtypes(include=['object']).columns


for column in list(cat_columns):
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

    # joined[column] = pd.factorize(joined[column].values, sort=True)[0]

# Preprocessing and transforming to sparse data
sparse_data = []

for f in tqdm(cat_columns):
    dummy = pd.get_dummies(joined[f].astype('category'))
    tmp = csr_matrix(dummy)
    sparse_data.append(tmp)

num_columns = [x for x in joined.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns if x not in ['id', 'loss']]

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

# def nn_model():
#     model = Sequential()
#     model.add(Dense(400, init='he_normal', activation='elu', input_dim=xtrain.shape[1]))
#     model.add(Dropout(0.5))
#     # model.add(BatchNormalization())
#     model.add(Dense(500, init='he_normal', activation='elu'))
#     model.add(Dropout(0.5))
#     # model.add(BatchNormalization())
#     # model.add(Dense(500, init='he_normal', activation='elu'))
#     # model.add(Dropout(0.5))
#     model.add(Dense(1))
#     # model.compile(loss='mae', optimizer='nadam')
#     return(model)
#
# # neural net
def nn_model():
    inputs = Input(shape=(xtrain.shape[1], ))
    l1 = Dense(700, init='he_uniform', activation='elu', name='l1')(inputs)
    x = Dropout(0.4)(l1)
    l2 = Dense(500, init='he_uniform', activation='elu', name='l2')(x)
    x = Dropout(0.4)(l2)
    x = merge([l1, x], mode='concat')
    x = BatchNormalization()(x)
    l3 = Dense(300, init='he_normal', activation='elu', name='l3')(x)
    x = Dropout(0.4)(l3)
    # x = Dropout(0.5)(x)
    x = merge([l2, x], mode='concat')
    x = BatchNormalization()(x)
    l4 = Dense(50, init='he_normal', activation='elu', name='l4')(x)
    # x = Dropout(0.5)(l4)
    # x = Dense(1, name='output')(x)
    x = Dense(1, name='output')(l4)
    model = Model(input=inputs, output=x)

    return model

# def nn_model():
#     model = Sequential()
#     model.add(MaxoutDense(500, 4, init='he_uniform', input_shape=(X_train.shape[1], )))
#     model.add(Dropout(0.5))
#     model.add(MaxoutDense(500, 4, init='he_uniform'))
#     model.add(Dropout(0.5))
#     model.add(Dense(1))
#     return model

# def nn_model():
#     model = Sequential()
#     model.add(GaussianNoise(1, input_shape=(X_train.shape[1], )))
#     model.add(Dense(500, init='he_uniform', activation='elu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(500, init='he_uniform', activation='elu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(1))
#     return model



sss = ShuffleSplit(n_splits=10, test_size=0.2, random_state=RANDOM_STATE)

train_index, test_index = sss.split(xtrain, y).next()

X_train = xtrain[train_index]
y_train = y[train_index]
X_val = xtrain[test_index]
y_val = y[test_index]

num_epoch = 3000

model = nn_model()

batch_size = 2**7

model.compile(loss='mse',
                  optimizer=Nadam(lr=1e-4),
                  metrics=[f_eval]
                  )

model.fit_generator(generator=batch_generator(X_train, y_train, batch_size, True),
                          verbose=1,
                          validation_data=batch_generator(X_val, y_val, batch_size, False),
                          nb_val_samples=X_val.shape[0],
                          nb_epoch=num_epoch,
                          samples_per_epoch=10000 * batch_size,
                          callbacks=[EarlyStopping(patience=10)]
                          )

# model.compile(loss='mse',
#                   optimizer=Nadam(lr=1e-5),
#                   metrics=[f_eval]
#                   )
#
# model.fit_generator(generator=batch_generator(X_train, y_train, 2**13, True),
#                           verbose=1,
#                           validation_data=batch_generator(X_val, y_val, 2**13, False),
#                           nb_val_samples=X_val.shape[0],
#                           nb_epoch=num_epoch,
#                           samples_per_epoch=X_train.shape[0],
#                           callbacks=[EarlyStopping(patience=100)]
#                           )


val_prediction = model.predict_generator(batch_generatorp(X_val, 16 * 256), X_val.shape[0])

print 'val_accuracy = ', mean_absolute_error(np.exp(y_val * y_std + y_mean), np.exp(val_prediction * y_std + y_mean))

test_prediction = model.predict_generator(batch_generatorp(xtest, 16 * 256), xtest.shape[0])

df = pd.DataFrame({'id': id_test, 'loss': np.exp(test_prediction.T[0] * y_std + y_mean) - shift})
df.to_csv('keras1.csv', index=False)
