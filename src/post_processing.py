from __future__ import division

import pandas as pd
from sklearn.metrics import mean_absolute_error
from bayes_opt import BayesianOptimization

train = pd.read_csv('../data/train.csv').sort_values(by='id')

lr_train = pd.read_csv('oof3/lr_train.csv').rename(columns={'loss': 'loss'}).sort_values(by='id')
lr_test = pd.read_csv('oof3/lr_test.csv').rename(columns={'loss': 'loss'}).sort_values(by='id')


def evaluate(multiplier, shift):
    return -mean_absolute_error(train['loss'], (lr_train['loss'] * multiplier + shift))


if __name__ == '__main__':
    init_points = 1000
    num_iter = 0
    print 'Baseline = ', mean_absolute_error(train['loss'], lr_train['loss'])

    weightsBO = BayesianOptimization(evaluate, {'multiplier': (1, 1.01),
                                                'shift': (-40, 0)})

    weightsBO.maximize(init_points=init_points, n_iter=num_iter)
    print(weightsBO.res['max']['max_val'])
