"""
Prepare file for callibration
"""

from __future__ import division
import pandas as pd


train = pd.read_csv('../data/train.csv')

pred_train = pd.read_csv('oof2/NN_train_s3.csv')
pred_test = pd.read_csv('oof2/NN_test_s3.csv')


pred_train.rename(columns={'loss': 'prediction'}).merge(train[['id', 'loss']], on='id').to_csv('calibration/trainsubmission.csv', index=False)
pred_test.rename(columns={'loss': 'prediction'}).to_csv('calibration/testsubmission.csv', index=False)