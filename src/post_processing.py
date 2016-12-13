from __future__ import division

import pandas as pd
from sklearn.metrics import mean_absolute_error
from bayes_opt import BayesianOptimization

train = pd.read_csv('../data/train.csv')

temp_columns = list(train.columns)
temp_columns.remove('id')
temp_columns.remove('loss')

train = train.drop_duplicates(subset=temp_columns).sort_values(by='id')


lr_train = pd.read_csv('oof2/svr_train_sqrt.csv').rename(columns={'loss': 'loss'}).sort_values(by='id')
lr_test = pd.read_csv('oof2/svr_test_sqrt.csv').rename(columns={'loss': 'loss'}).sort_values(by='id')


def evaluate(multiplier, shift, threashold):
    temp = lr_train['loss'].copy()
    temp[temp > threashold] *= multiplier
    return -mean_absolute_error(train['loss'], (temp + shift))


if __name__ == '__main__':
    init_points = 10000
    num_iter = 1
    print 'Baseline = ', mean_absolute_error(train['loss'], lr_train['loss'])

    weightsBO = BayesianOptimization(evaluate, {'multiplier': (0.9, 1.1),
                                                'shift': (-40, 40),
                                                'threashold': (200, 60000)})

    weightsBO.maximize(init_points=init_points, n_iter=num_iter, acq='ucb', kappa=7)
    print(weightsBO.res['max']['max_val'])
