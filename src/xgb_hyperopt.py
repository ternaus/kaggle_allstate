import pandas as pd

from sklearn.model_selection import train_test_split
import sys

sys.path += ['/home/vladimir/packages/xgboost/python-package']


import pandas as pd
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
from hyperopt import hp, tpe, STATUS_OK, Trials
import hyperopt
from sklearn.preprocessing import LabelEncoder

from pylab import *
from tqdm import tqdm

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
test['loss'] = np.nan
joined = pd.concat([train, test])


for column in tqdm(list(train.select_dtypes(include=['object']).columns)):
    le = LabelEncoder()
    joined[column] = le.fit_transform(joined[column])
    # g = train.groupby(column)['loss'].mean()
    # g = train.groupby(column)['loss'].median()
    # joined[column] = joined[column].map(g)


train = joined[joined['loss'].notnull()]
test = joined[joined['loss'].isnull()]

shift = 200

y = np.log(train['loss'] + shift)
ids = test['id']
X = train.drop(['loss', 'id'], 1)
X_test = test.drop(['loss', 'id'], 1)

num_rounds = 3000
RANDOM_STATE = 2016

print '[{datetime}] train set size = {size}'.format(datetime=str(datetime.datetime.now()), size=X.shape)

print '[{datetime}] splitting'.format(datetime=str(datetime.datetime.now()))


def xgb_eval_mae(yhat, dtrain):
    y = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(y), np.exp(yhat))


space = {
        'max_depth': hp.choice('max_depth', np.arange(5, 15, dtype=int)),
        'min_child_weight': hp.choice('min_child_weight', np.arange(1, 20, dtype=int)),
        'subsample': hp.uniform('subsample', 0.5, 1),
        # 'n_estimators': hp.choice('n_estimators', np.arange(100, 1000, 1, dtype=int)),
        'n_estimators': 1000,
        # 'learning_rate': hp.uniform('learning_rate', 0.01, 0.1),
        'learning_rate': 0.1,
        'gamma': hp.uniform('gamma', 0, 10),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1)
    }


def objective(space):
    params = {
        'min_child_weight': space['min_child_weight'],
        'eta': space['learning_rate'],
        'colsample_bytree': space['colsample_bytree'],
        'max_depth': space['max_depth'],
        'subsample': space['subsample'],
        'gamma': space['gamma'],
        'n_estimators': space['n_estimators'],
        'silent': 1,
        # 'eval_metric': ['rmse', 'mae'],
        # 'eval_metric': xgb_eval_mae,
        'verbose_eval': None,
        'seed': RANDOM_STATE
    }
    print params

    xgtrain = xgb.DMatrix(X, label=y)

    cv = xgb.cv(params,
                xgtrain,
                nfold=5,
                feval=xgb_eval_mae,
                num_boost_round=space['n_estimators'],
                early_stopping_rounds=50, as_pandas=True)

    loss = cv['test-mae-mean'].values[-1]
    print 'loss = ', loss

    return{'loss': loss, 'status': STATUS_OK}


trials = Trials()
best = hyperopt.fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=10000,
            trials=trials
            )

print(best)

fName = open('trials.pkl', 'w')
pickle.dump(trials, fName)
fName.close()