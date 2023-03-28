import pandas as pd
import numpy as np
import random
import os

from supervised.automl import AutoML

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

train_x = train_data.drop(['ID', '전화해지여부'], axis = 1)
train_y = train_data['전화해지여부']

test_x = test_data.drop('ID', axis = 1)

automl = AutoML(mode="Compete", eval_metric='f1', total_time_limit = 60 * 60 * 5)
automl.fit(train_x, train_y)

pred = automl.predict(test_x)a

submission = pd.read_csv("./data/sample_submission.csv")
submission['전화해지여부'] = pred
submission.to_csv("using_automl.csv", index=False)