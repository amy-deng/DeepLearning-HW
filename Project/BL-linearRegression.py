# -*- coding: utf-8 -*-

import time
import os
import gc
import pandas as pd
import numpy as np
from sklearn import linear_model

# use default params
regr = linear_model.LinearRegression()

orig_data = pd.read_pickle('data-all-features.pkl')
orig_data = orig_data.fillna(0)

# use all features
data = orig_data

test  = pd.read_csv('data/test.csv.gz',compression='gzip').set_index('ID')

X_train = data[data.date_block_num < 34].drop(['item_cnt_month'], axis=1)
Y_train = data[data.date_block_num < 34]['item_cnt_month']
# X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)
# Y_train = data[data.date_block_num < 33]['item_cnt_month']
# X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
# Y_valid = data[data.date_block_num == 33]['item_cnt_month']
X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)

regr.fit(X_train, Y_train)

pred = regr.predict(X_test).clip(0, 20)
submission = pd.DataFrame({
    "ID": test.index, 
    "item_cnt_month": pred
})
submission.to_csv('lr_submission.csv', index=False)

# 1.00917
