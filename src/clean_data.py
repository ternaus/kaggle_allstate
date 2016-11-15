"""
Script that performs different types of Data cleaning
"""
from __future__ import division
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix, hstack
from tqdm import tqdm
from scipy.stats import skew, boxcox
import itertools


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


def filter_cat(joined, train, test, column):
    if train[column].nunique() != test[column].nunique():
        # Let's find extra categories...
        set_train = set(train[column].unique())
        set_test = set(test[column].unique())
        remove_train = set_train - set_test
        remove_test = set_test - set_train

        remove = remove_train.union(remove_test)
        print column, remove

        # print column, remove

        def helper(x):
            if x in remove:
                return np.nan
            return x

        return joined[column].apply(lambda x: helper(x), 1)
    return joined[column]


def one_hot_categorical(shift=0, subtract_mean=False, quadratic=False):
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

    if quadratic:
        numeric_feats = [x for x in train.columns if 'cont' in x]

        joined, ntrain = mungeskewed(train, test, numeric_feats)
        COMB_FEATURE = 'cat80,cat87,cat57,cat12,cat79,cat10,cat7,cat89,cat2,cat72,cat81,cat11,cat1,cat13,cat9,cat3,cat16,cat90,cat23,cat36,cat73,cat103,cat40,cat28,cat111,cat6,cat76,cat50,cat5,cat4,cat14,cat38,cat24,cat82,cat25'.split(
        ',')
        for comb in tqdm(list(itertools.combinations(COMB_FEATURE, 2))):
            feat = comb[0] + "_" + comb[1]

            joined[feat] = joined[comb[0]] + joined[comb[1]]

        train = joined.iloc[:ntrain, :]
        test = joined.iloc[ntrain:, :]

    # Preprocessing and transforming to sparse data
    print joined.head().info()
    cat_columns = joined.select_dtypes(include=['object']).columns

    to_drop = []
    for column in list(cat_columns):
        joined[column] = filter_cat(joined, train, test, column)
        if joined[column].nunique() == 1:
            to_drop += [column]

    print 'dropping = ', to_drop
    joined = joined.drop(to_drop, 1)

    cat_columns = joined.select_dtypes(include=['object']).columns

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


def encode(charcode):
    r = 0
    ln = len(charcode)
    for i in range(ln):
        r += (ord(charcode[i]) - ord('A') + 1) * 26**(ln-i-1)
    return r


def mungeskewed(train, test, numeric_feats):
    ntrain = train.shape[0]
    test['loss'] = 0
    train_test = pd.concat((train, test)).reset_index(drop=True)
    # compute skew and do Box-Cox transformation (Tilli)
    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))
    print
    print("Skew in numeric features:")
    print(skewed_feats)
    skewed_feats = skewed_feats[skewed_feats > 0.25]
    skewed_feats = skewed_feats.index

    for feats in skewed_feats:
        train_test[feats] += 1
        train_test[feats], lam = boxcox(train_test[feats])
    return train_test, ntrain


def fancy(shift=200):
    """
    From https://www.kaggle.com/modkzs/allstate-claims-severity/lexical-encoding-feature-comb/code
    :param shift:
    :return:
    """
    COMB_FEATURE = 'cat80,cat87,cat57,cat12,cat79,cat10,cat7,cat89,cat2,cat72,cat81,cat11,cat1,cat13,cat9,cat3,cat16,cat90,cat23,cat36,cat73,cat103,cat40,cat28,cat111,cat6,cat76,cat50,cat5,cat4,cat14,cat38,cat24,cat82,cat25'.split(
        ',')
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')
    numeric_feats = [x for x in train.columns if 'cont' in x]

    joined, ntrain = mungeskewed(train, test, numeric_feats)

    # Adding quadratic features
    for comb in tqdm(list(itertools.combinations(COMB_FEATURE, 2))):
        feat = comb[0] + "_" + comb[1]

        joined[feat] = joined[comb[0]] + joined[comb[1]]
        assert joined[feat].isnull().sum() == 0

    train = joined.iloc[:ntrain, :]
    test = joined.iloc[ntrain:, :]

    # Encoding categorical features
    cats = [x for x in joined.columns if 'cat' in x]
    to_drop = []
    for column in tqdm(cats):
        joined[column] = filter_cat(joined, train, test, column)
        if joined[column].nunique() == 1:
            to_drop += [column]
            continue
        joined[column] = joined[column].fillna('unknown')

        encode_dict = {}
        for value in joined[column].unique():
            encode_dict[value] = encode(value)

        joined[column] = joined[column].map(encode_dict)
        assert joined[column].isnull().sum() == 0

    print 'dropping = ', to_drop
    joined = joined.drop(to_drop, 1)

    print 'scaling'
    ss = StandardScaler()
    joined[numeric_feats] = ss.fit_transform(joined[numeric_feats].values)
    X_train = joined.iloc[:ntrain, :]
    X_test = joined.iloc[ntrain:, :]
    y_train = np.log(X_train['loss'] + shift)
    y_mean = y_train.mean()

    return X_train.drop(['loss', 'id'], 1), y_train, X_test.drop(['loss', 'id'], 1), y_mean, X_test['id'], X_train['id']
