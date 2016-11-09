"""
Script that performs different types of Data cleaning
"""
from __future__ import division
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix, hstack
from tqdm import tqdm


def label_encode(shift=200):
    """

    :return: data with categorical encoded as factor
    """
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')
    test['loss'] = np.nan
    joined = pd.concat([train, test])
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

    train = joined[joined['loss'].notnull()]
    test = joined[joined['loss'].isnull()]

    y_train = np.log(train['loss'] + shift)
    test_ids = test['id']
    X_train = train.drop(['loss', 'id'], 1)
    X_test = test.drop(['loss', 'id'], 1)

    return X_train, y_train, X_test, test_ids


def oof_categorical(shift=200, scale=False, subtract_min=False):
    """
    Categorical variables are replced by mean of loss


    :param shift:
    :return:
    """
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')
    test['loss'] = np.nan

    joined = pd.concat([train, test])
    for column in list(train.select_dtypes(include=['object']).columns):
        g = train.groupby(column)['loss'].mean()

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

        joined[column] = joined[column].map(g)
        mean_column = joined[column].mean()
        joined[column] = joined[column].fillna(mean_column)

    if scale:
        for column in joined.columns:
            if column not in ['id', 'loss']:
                scaler = StandardScaler()
                joined[column] = scaler.fit_transform(joined[column].values.reshape(1, -1).T)

    train = joined[joined['loss'].notnull()]
    test = joined[joined['loss'].isnull()]

    y_train = np.log(train['loss'].values + shift)

    y_mean = y_train.mean()

    if subtract_min:
        y_train -= y_mean

    test_ids = test['id'].values
    train_ids = train['id'].values

    X_train = train.drop(['loss', 'id'], 1).values
    X_test = test.drop(['loss', 'id'], 1).values

    return X_train, y_train, X_test, y_mean, test_ids, train_ids


def one_hot_categorical(shift=0, subtract_mean=False):
    ## read data
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')

    ## set test loss to NaN
    test['loss'] = np.nan

    # response and IDs

    y = np.log(train['loss'].values + shift)

    y_mean = y.mean()
    if subtract_mean:
        y = y - y_mean

    id_train = train['id'].values
    id_test = test['id'].values

    # stack train test
    ntrain = train.shape[0]
    joined = pd.concat((train, test), axis=0)

    # Preprocessing and transforming to sparse data

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

    # sparse train and test data
    xtr_te = hstack(sparse_data, format='csr')
    xtrain = xtr_te[:ntrain, :]
    xtest = xtr_te[ntrain:, :]
    return xtrain, y, xtest, y_mean, id_test, id_train
