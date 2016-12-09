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
from scipy.stats import rankdata
from StringIO import StringIO
import string


def label_encode(shift=200):
    """

    :return: data with categorical encoded as factor
    """
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')
    test['loss'] = np.nan
    joined = pd.concat([train, test])
    for column in list(train.select_dtypes(include=['object']).columns):

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

    # Let's find extra categories...
    set_train = set(train[column].unique())
    set_test = set(test[column].unique())
    remove_train = set_train - set_test
    remove_test = set_test - set_train

    remove = remove_train.union(remove_test)
    # print column, remove

    def helper(x):
        if x in remove:
            return np.nan
        return x

    if len(remove) != 0:
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

    joined_t = joined.copy()
    cats_old = [x for x in joined_t.columns if 'cat' in x]
    for column in cats_old:
        joined_t[column] = pd.factorize(joined_t[column], sort=True)[0]

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

    joined['sum_of_cats_cont'] = (joined_t[cats_old] == 0).sum(axis=1)
    joined['sum_of_cats_0_cont'] = (joined_t[cats_old][0:71] == 0).sum(axis=1)

    print joined['sum_of_cats_cont'].value_counts()
    print
    print joined['sum_of_cats_0_cont'].value_counts()
    joined = joined.fillna(0)

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


def one_hot_categorical_sqrt(subtract_mean=False, quadratic=False):
    ## read data
    train = pd.read_csv('../data/train.csv')

    print train.shape

    temp_columns = list(train.columns)
    temp_columns.remove('id')
    temp_columns.remove('loss')

    train = train.drop_duplicates(subset=temp_columns)
    print train.shape

    test = pd.read_csv('../data/test.csv')

    ## set test loss to NaN
    test['loss'] = np.nan

    # response and IDs

    y = np.sqrt(np.sqrt(train['loss'].values))

    y_mean = y.mean()
    if subtract_mean:
        y = y - y_mean

    id_train = train['id'].values
    id_test = test['id'].values

    # stack train test
    ntrain = train.shape[0]
    joined = pd.concat((train, test), axis=0)

    joined_t = joined.copy()
    cats_old = [x for x in joined_t.columns if 'cat' in x]

    for column in cats_old:
        joined_t[column] = pd.factorize(joined_t[column], sort=True)[0]

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

    # joined['sum_of_cats_cont'] = (joined_t[cats_old] == 0).sum(axis=1)
    # joined['sum_of_cats_0_cont'] = (joined_t[cats_old][0:71] == 0).sum(axis=1)
    #
    # print joined['sum_of_cats_cont'].value_counts()
    # print
    # print joined['sum_of_cats_0_cont'].value_counts()
    joined = joined.fillna(0)

    f_num = [f for f in joined.columns if 'cont' in f]
    print len(f_num)
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
    skewed_feats = skewed_feats[np.abs(skewed_feats) > 0.25]
    skewed_feats = skewed_feats.index

    for feats in skewed_feats:
        train_test[feats] += 1
        train_test[feats], lam = boxcox(train_test[feats])
    return train_test, ntrain


def fancy(shift=200, quadratic=False, truncate=False):
    """
    From https://www.kaggle.com/modkzs/allstate-claims-severity/lexical-encoding-feature-comb/code
    :param shift:
    :return:
    """
    COMB_FEATURE = 'cat80,cat87,cat57,cat12,cat79,cat10,cat7,cat89,cat2,cat72,cat81,cat11,cat1,cat13,cat9,cat3,cat16,cat90,cat23,cat36,cat73,cat103,cat40,cat28,cat111,cat6,cat76,cat50,cat5,cat4,cat14,cat38,cat24,cat82,cat25'.split(
        ',')
    train = pd.read_csv('../data/train.csv')
    print train.shape

    temp_columns = list(train.columns)
    temp_columns.remove('id')
    temp_columns.remove('loss')

    train = train.drop_duplicates(subset=temp_columns)
    print train.shape

    if truncate:
        train = train[(train['loss'] > 200) & (train['loss'] < 30000)]

    test = pd.read_csv('../data/test.csv')
    numeric_feats = [x for x in train.columns if 'cont' in x]

    joined, ntrain = mungeskewed(train, test, numeric_feats)

    joined_t = joined.copy()
    cats_old = [x for x in joined_t.columns if 'cat' in x]
    for column in cats_old:
        joined_t[column] = pd.factorize(joined_t[column], sort=True)[0]

    # Adding extra features
    joined['state'] = joined['cat112'].map(_get_state())
    print joined['state'].unique()

    if quadratic:
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
        joined[column] = joined[column].fillna('UNKNOWN')

        encode_dict = {}
        for value in joined[column].unique():
            encode_dict[value] = encode(value)

        joined[column] = joined[column].map(encode_dict)
        assert joined[column].isnull().sum() == 0

    print 'dropping = ', to_drop
    joined = joined.drop(to_drop, 1)

    joined['sum_of_cats_0'] = (joined_t[cats_old] == 0).sum(axis=1)
    joined['sum_of_cats_0_71'] = (joined_t[cats_old][0:71] == 0).sum(axis=1)
    joined = joined.fillna(0)

    print 'scaling'
    ss = StandardScaler()
    joined[numeric_feats] = ss.fit_transform(joined[numeric_feats].values)
    X_train = joined.iloc[:ntrain, :]
    X_test = joined.iloc[ntrain:, :]
    y_train = np.log(X_train['loss'] + shift)
    y_mean = y_train.mean()

    return X_train.drop(['loss', 'id'], 1), y_train, X_test.drop(['loss', 'id'], 1), y_mean, X_test['id'], X_train['id']


def fancy_sqrt(quadratic=False, add_aggregates=False):
    COMB_FEATURE = 'cat80,cat87,cat57,cat12,cat79,cat10,cat7,cat89,cat2,cat72,cat81,cat11,cat1,cat13,cat9,cat3,cat16,cat90,cat23,cat36,cat73,cat103,cat40,cat28,cat111,cat6,cat76,cat50,cat5,cat4,cat14,cat38,cat24,cat82,cat25'.split(
        ',')
    train = pd.read_csv('../data/train.csv')
    print train.shape

    temp_columns = list(train.columns)
    temp_columns.remove('id')
    temp_columns.remove('loss')

    train = train.drop_duplicates(subset=temp_columns)

    # train = train.merge(pd.read_csv('../data/train_window.csv'), on='id')

    test = pd.read_csv('../data/test.csv')
    # test = test.merge(pd.read_csv('../data/test_window.csv'), on='id')

    # numeric_feats = [x for x in train.columns if 'cont' in x]

    # joined, ntrain = mungeskewed(train, test, numeric_feats)
    joined = pd.concat([train, test])
    ntrain = train.shape[0]

    # Adding extra features
    joined['state'] = joined['cat112'].map(_get_state())
    joined['census_region'] = joined['state'].map(_get_census_region())
    joined['timezone'] = joined['state'].map(_get_timezone())

    # joined['state'] = pd.factorize(joined['state'], sort=True)[0]
    joined = joined.drop('state', 1)
    joined['census_region'] = pd.factorize(joined['census_region'], sort=True)[0]
    joined['timezone'] = pd.factorize(joined['timezone'], sort=True)[0]

    # for column in tqdm(temp_columns):
    #     joined[column.upper() + '_count'] = joined[column].map(joined.groupby(column)[column].count())

    min_cont14 = np.diff(joined['cont14'])
    min_cont14 = min_cont14[min_cont14 > 0].min()
    joined['cont14'] = (joined['cont14'] / min_cont14).astype(int)

    # joined['cont14_7'] = joined['cont14'] % 7
    # joined['cont14_365'] = joined['cont14'] % 365
    # joined['cont14_24'] = joined['cont14'] % 24
    # joined['cont14_60'] = joined['cont14'] % 60
    # joined['cont14_5'] = joined['cont14'] % 5

    min_cont7 = np.diff(joined['cont7'])
    min_cont7 = min_cont7[min_cont7 > 0].min()
    joined['cont7'] = (joined['cont7'] / min_cont7).astype(int)

    # joined['cont7_7'] = joined['cont7'] % 7
    # joined['cont7_365'] = joined['cont7'] % 365
    # joined['cont7_24'] = joined['cont7'] % 24
    # joined['cont7_60'] = joined['cont7'] % 60
    # joined['cont7_5'] = joined['cont7'] % 5

    joined_t = joined.copy()
    cats_old = [x for x in joined_t.columns if 'cat' in x]
    for column in cats_old:
        joined_t[column] = pd.factorize(joined_t[column], sort=True)[0]

    if quadratic:
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
        joined[column] = joined[column].fillna('UNKNOWN')

        encode_dict = {}
        for value in joined[column].unique():
            encode_dict[value] = encode(value)

        joined[column] = joined[column].map(encode_dict)
        assert joined[column].isnull().sum() == 0

    print 'dropping = ', to_drop
    joined = joined.drop(to_drop, 1)

    if add_aggregates:
        for column in [x for x in joined_t.columns if 'cat' in x]:
            mean_mapping = train.groupby(column)['loss'].mean()
            median_mapping = train.groupby(column)['loss'].median()
            std_mapping = train.groupby(column)['loss'].std()
            max_mapping = train.groupby(column)['loss'].max()
            min_mapping = train.groupby(column)['loss'].min()
            joined[column + '_mean'] = joined_t[column].map(mean_mapping)
            joined[column + '_median'] = joined_t[column].map(median_mapping)
            joined[column + '_std'] = joined_t[column].map(std_mapping)
            joined[column + '_max'] = joined_t[column].map(max_mapping)
            joined[column + '_min'] = joined_t[column].map(min_mapping)

    joined['sum_of_cats_0'] = (joined_t[cats_old] == 0).sum(axis=1)
    joined['sum_of_cats_0_71'] = (joined_t[cats_old][0:71] == 0).sum(axis=1)

    joined = joined.fillna(0)

    # print 'scaling'
    # ss = StandardScaler()
    # joined[numeric_feats] = ss.fit_transform(joined[numeric_feats].values)
    X_train = joined.iloc[:ntrain, :]
    X_test = joined.iloc[ntrain:, :]
    y_train = np.sqrt(np.sqrt(X_train['loss'].values))
    y_mean = y_train.mean()

    return X_train.drop(['loss', 'id'], 1), y_train, X_test.drop(['loss', 'id'], 1), y_mean, X_test['id'], X_train['id']


def _get_state():
    POPULATION_DATA = StringIO('state_full,state,population\nAlabama,AL,' \
                               + '4779736\nAlaska,AK,710231\nArizona,AZ,6392017\nArkansas,AR,' \
                               + '2915918\nCalifornia,CA,37253956\nColorado,CO,5029196\nConnecticut,CT,' \
                               + '3574097\nDelaware,DE,897934\nDistrict of Columbia,DC,601723\nFlorida,FL,' \
                               + '18801310\nGeorgia,GA,9687653\nHawaii,HI,1360301\nIdaho,ID,' \
                               + '1567582\nIllinois,IL,12830632\nIndiana,IN,6483802\nIowa,IA,' \
                               + '3046355\nKansas,KS,2853118\nKentucky,KY,4339367\nLouisiana,LA,' \
                               + '4533372\nMaine,ME,1328361\nMaryland,MD,5773552\nMassachusetts,MA,' \
                               + '6547629\nMichigan,MI,9883640\nMinnesota,MN,5303925\nMississippi,MS,' \
                               + '2967297\nMissouri,MO,5988927\nMontana,MT,989415\nNebraska,NE,' \
                               + '1826341\nNevada,NV,2700551\nNew Hampshire,NH,1316470\nNew Jersey,NJ,' \
                               + '8791894\nNew Mexico,NM,2059179\nNew York,NY,19378102\nNorth Carolina,NC'
                               + ',9535483\nNorth Dakota,ND,672591\nOhio,OH,11536504\nOklahoma,OK,' \
                               + '3751351\nOregon,OR,3831074\nPennsylvania,PA,12702379\nRhode Island,RI,' \
                               + '1052567\nSouth Carolina,SC,4625364\nSouth Dakota,SD,814180\nTennessee,TN,' \
                               + '6346105\nTexas,TX,25145561\nUtah,UT,2763885\nVermont,VT,625741\nVirginia,VA,' \
                               + '8001024\nWashington,WA,6724540\nWest Virginia,WV,1852994\nWisconsin,WI,' \
                               + '5686986\nWyoming,WY,563626\n')

    pop = pd.read_csv(POPULATION_DATA)
    translation = list(string.ascii_uppercase)[:-1]
    for elem_i in translation[:2]:
        for elem_j in translation[:25]:
            translation.append(elem_i + elem_j)

    return dict(zip(translation[:51], pop.state))


def classes(y, bins):
    """

    :param y: list of targets
    :param bins:
    :return:
    """
    rank_y = rankdata(y, method='ordinal') / len(y)
    hist, bin_edges = np.histogram(rank_y, bins=bins)
    result = np.array(map(lambda x: np.searchsorted(bin_edges, x, side='left'), rank_y))
    result[result == 0] = 1
    result -= 1

    print min(result)
    print max(result)
    assert min(result) == 0
    assert max(result) == bins - 1
    return result


def _get_census_region():
    maps = [('TX', 'south'),
     ('VA', 'south'),
     ('AZ', 'west'),
     ('IL', 'midwest'),
     ('MS', 'south'),
     ('FL', 'south'),
     ('NY', 'northeast'),
     ('GA', 'south'),
     ('MD', 'south'),
     ('CA', 'west'),
     ('OH', 'midwest'),
     ('NC', 'south'),
     ('NH', 'northeast'),
     ('AL', 'south'),
     ('HI', 'west'),
     ('CO', 'west'),
     ('SC', 'south'),
     ('NV', 'west'),
     ('NJ', 'northeast'),
     ('OK', 'south'),
     ('PA', 'northeast'),
     ('LA', 'south'),
     ('WA', 'west'),
     ('DC', 'south'),
     ('TN', 'south'),
     ('WV', 'south'),
     ('VT', 'northeast'),
     ('SD', 'midwest'),
     ('IN', 'midwest'),
     ('RI', 'northeast'),
     ('KY', 'south'),
     ('DE', 'south'),
     ('CT', 'northeast'),
     ('NE', 'midwest'),
     ('UT', 'west'),
     ('NM', 'west'),
     ('MN', 'midwest'),
     ('MO', 'midwest'),
     ('KS', 'midwest'),
     ('WI', 'midwest'),
     ('AR', 'south'),
     ('WY', 'west'),
     ('IA', 'midwest'),
     ('AK', 'west'),
     ('OR', 'west'),
     ('ID', 'west'),
     ('ME', 'northeast'),
     ('MI', 'midwest'),
     ('MA', 'northeast'),
     ('MT', 'west'),
     ('ND', 'midwest')]

    return dict(maps)


def _get_timezone():
    maps = [('TX', 'CST'),
     ('VA', 'EST'),
     ('AZ', 'MST'),
     ('IL', 'CST'),
     ('MS', 'CST'),
     ('FL', 'EST'),
     ('NY', 'EST'),
     ('GA', 'EST'),
     ('MD', 'EST'),
     ('CA', 'PST'),
     ('OH', 'EST'),
     ('NC', 'EST'),
     ('NH', 'EST'),
     ('AL', 'CST'),
     ('HI', 'HST'),
     ('CO', 'MST'),
     ('SC', 'EST'),
     ('NV', 'PST'),
     ('NJ', 'EST'),
     ('OK', 'CST'),
     ('PA', 'EST'),
     ('LA', 'CST'),
     ('WA', 'PST'),
     ('DC', 'EST'),
     ('TN', 'CST'),
     ('WV', 'EST'),
     ('VT', 'EST'),
     ('SD', 'CST'),
     ('IN', 'EST'),
     ('RI', 'EST'),
     ('KY', 'CST'),
     ('DE', 'EST'),
     ('CT', 'EST'),
     ('NE', 'CST'),
     ('UT', 'MST'),
     ('NM', 'MST'),
     ('MN', 'CST'),
     ('MO', 'CST'),
     ('KS', 'CST'),
     ('WI', 'CST'),
     ('AR', 'CST'),
     ('WY', 'MST'),
     ('IA', 'CST'),
     ('AK', 'AKST'),
     ('OR', 'PST'),
     ('ID', 'MST'),
     ('ME', 'EST'),
     ('MI', 'EST'),
     ('MA', 'EST'),
     ('MT', 'MST'),
     ('ND', 'CST')]
    return dict(maps)
