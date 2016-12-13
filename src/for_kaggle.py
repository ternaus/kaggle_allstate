"""
Based on https://www.kaggle.com/danijelk/allstate-claims-severity/keras-starter-with-bagging-lb-1120-596

 by Danijel Kivaranovic
"""

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack

from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold

import keras.backend as K
from keras.models import Sequential
from keras.optimizers import Nadam
from keras.layers import Dense, Dropout


__author__ = 'Vladimir Iglovikov'

RANDOM_SEED = 2016
np.random.seed(RANDOM_SEED)


def f_eval(y_true, y_pred):
    return K.mean(K.abs(K.exp(y_pred + y_mean) - K.exp(y_true + y_mean)))


def batch_generator(X, y, batch_size, shuffle):
    # changelog code for fitting from generator
    # (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    number_of_batches = np.ceil(X.shape[0] / batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
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


def clean_cat_non_train_test_overlap(train, test, joined, column):
    """
    Replace categorical variables in column that appear only in train or only in test by NaN
    """
    # Let's find extra categories...
    set_train = set(train[column].unique())
    set_test = set(test[column].unique())
    remove_train = set_train - set_test
    remove_test = set_test - set_train

    remove = remove_train.union(remove_test)

    def filter_cat(x):
        if x in remove:
            return np.nan
        return x

    return joined[column].apply(lambda x: filter_cat(x), 1)


def prepare_data():
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')

    num_train = train.shape[0]

    test['loss'] = np.nan

    shift = 200
    y = np.log(train['loss'].values + shift)
    y_mean = y.mean()
    y = y - y_mean

    test_ids = test['id'].values

    joined = pd.concat((train, test), axis=0)
    #
    # joined = joined.drop(['cat110', 'cat116'], 1)
    #
    # joined = joined.drop(['cat64',
    #                       'cat62',
    #                       'cat15',
    #                       'cat70',
    #                       'cat69',
    #                       'cat35',
    #                       'cat20',
    #                       'cat55',
    #                       'cat34',
    #                       'cat47',
    #                       'cat48',
    #                       'cat58',
    #                       'cat56',
    #                       'cat46',
    #                       'cat33',
    #                       'cat63',
    #                       'cat22',
    #                       'cat32'], 1)

    # Preproccesing and transforming to sparse data
    cat_columns = joined.select_dtypes(include=['object']).columns

    for column in cat_columns:
        if train[column].nunique() != test[column].nunique():
            joined[column] = clean_cat_non_train_test_overlap(train, test, joined, column)

    sparse_data = []

    for column in cat_columns:
        dummy = csr_matrix(pd.get_dummies(joined[column].astype('category'), sparse=True))
        sparse_data.append(dummy)

    num_columns = [f for f in joined.columns if 'cont' in f]

    scaler = StandardScaler()
    tmp = csr_matrix(scaler.fit_transform(joined[num_columns]))
    sparse_data.append(tmp)

    xtr_te = hstack(sparse_data, format='csr')
    X_train = xtr_te[:num_train, :]
    X_test = xtr_te[num_train:, :]

    return X_train, y, y_mean, shift, X_test, test_ids


def nn_model():
    model = Sequential()
    model.add(Dense(400, init='he_normal', activation='elu', input_dim=X_train.shape[1]))
    model.add(Dropout(0.4))
    model.add(Dense(200, init='he_normal', activation='elu'))
    # model.add(Dropout(0.4))
    # model.add(Dense(50, init='he_normal', activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    return model


if __name__ == '__main__':
    X, y, y_mean, shift, X_test, test_ids = prepare_data()
    print('size of the train set = ', X.shape)
    nfolds = 5
    folds = KFold(len(y), n_folds=nfolds, shuffle=True, random_state=2016)

    num_epoch = 15
    batch_size = 2**10
    pred_oob = np.zeros(X.shape[0])
    pred_test = np.zeros(X_test.shape[0])

    scores = []

    for i, (train_index, val_index) in enumerate(folds):
        X_train = X[train_index]
        y_train = y[train_index]
        X_val = X[val_index]
        y_val = y[val_index]
        pred_val = np.zeros(X_val.shape[0])

        model = nn_model()
        model.compile(loss='mae', optimizer=Nadam(1e-4), metrics=[f_eval])

        fit = model.fit_generator(generator=batch_generator(X_train, y_train, batch_size, True),
                                  samples_per_epoch=10000 * batch_size,
                                  nb_epoch=num_epoch,
                                  validation_data=batch_generator(X_val, y_val, batch_size, False),
                                  nb_val_samples=X_val.shape[0])

        pred_val += model.predict_generator(generator=batch_generatorp(X_val, 800), val_samples=X_val.shape[0])[:, 0]
        pred_test += model.predict_generator(generator=batch_generatorp(X_test, 800), val_samples=X_test.shape[0])[:, 0]

        score = mean_absolute_error(np.exp(y_val + y_mean), np.exp(pred_val + y_mean))
        scores += [score]
        print('Fold ', i, '- MAE:', score)

    print('Total: {mean_mae} +- {std_mae}'.format(mean_mae=np.mean(scores), std_mae=np.std(scores)))

    # test predictions
    pred_test /= nfolds
    df = pd.DataFrame({'id': test_ids, 'loss': np.exp(pred_test + y_mean) - shift})
    df.to_csv('submission_keras_v.csv', index=False)
