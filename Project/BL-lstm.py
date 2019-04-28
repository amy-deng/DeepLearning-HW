# -*- coding: utf-8 -*-
## import packages
import os
os.environ["CUDA_VISIBLE_DEVICES"]='2' 

import numpy as np
import pandas as pd
from itertools import product
from sklearn.preprocessing import LabelEncoder

import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

import time
import sys
import gc
import pickle

from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, ThresholdedReLU, MaxPooling2D, Embedding, Dropout, CuDNNLSTM, CuDNNGRU, Bidirectional
from keras.optimizers import Adam, SGD, RMSprop
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from keras.utils import multi_gpu_model

## import data
items = pd.read_csv('data/items.csv')
shops = pd.read_csv('data/shops.csv')
cats = pd.read_csv('data/item_categories.csv')
train = pd.read_csv('data/sales_train.csv.gz',compression='gzip',parse_dates=['date'], infer_datetime_format=True, dayfirst=True)
test  = pd.read_csv('data/test.csv.gz',compression='gzip')

## remove outliers
# plt.figure(figsize=(10,4))
# plt.xlim(-100, 3000)
# sns.boxplot(x=train.item_cnt_day)

# plt.figure(figsize=(10,4))
# plt.xlim(train.item_price.min(), train.item_price.max()*1.1)
# sns.boxplot(x=train.item_price)

train = train[train.item_price < 100000]
train = train[train.item_cnt_day < 1200]

# one item with price below zero. fill it with median
condition = (train.shop_id==32) & (train.item_id==2973) & (train.date_block_num==4) & (train.item_price > 0)
median = train[condition].item_price.median()
train.loc[train.item_price < 0, 'item_price'] = median

## shop name issues. Process items with similar shop names but different shop ids
# Якутск Орджоникидзе, 56 == !Якутск Орджоникидзе, 56 фран
train.loc[train.shop_id == 0, 'shop_id'] = 57
test.loc[test.shop_id == 0, 'shop_id'] = 57

# Якутск ТЦ "Центральный" == !Якутск ТЦ "Центральный" фран
train.loc[train.shop_id == 1, 'shop_id'] = 58
test.loc[test.shop_id == 1, 'shop_id'] = 58

# Жуковский ул. Чкалова 39м² == Жуковский ул. Чкалова 39м?
train.loc[train.shop_id == 10, 'shop_id'] = 11
test.loc[test.shop_id == 10, 'shop_id'] = 11

## process the raw data to be monthly sales by item-shop 
df = train.groupby([train.date.apply(lambda x: x.strftime('%Y-%m')),'item_id','shop_id']).sum().reset_index()
df = df[['date','item_id','shop_id','item_cnt_day']]
df["item_cnt_day"].clip(0.,20.,inplace=True)  # Competition: True target values are clipped into [0,20] range.
df = df.pivot_table(index=['item_id','shop_id'], columns='date',values='item_cnt_day',fill_value=0).reset_index()

# merge data from monthly sales to specific item-shops in test data
val = pd.merge(test,df,on=['item_id','shop_id'], how='left').fillna(0)

# only keep raw timeseries
val = val.drop(labels=['ID','item_id','shop_id'],axis=1)

## process the raw price data to be monthly average price by item-shop
scaler = MinMaxScaler(feature_range=(0, 1))
train["item_price"] = scaler.fit_transform(train["item_price"].values.reshape(-1,1))
df2 = train.groupby([train.date.apply(lambda x: x.strftime('%Y-%m')),'item_id','shop_id']).mean().reset_index()
df2 = df2[['date','item_id','shop_id','item_price']].pivot_table(index=['item_id','shop_id'], columns='date',values='item_price',fill_value=0).reset_index()

# process data from average prices to specific item-shops in test data
price = pd.merge(test,df2,on=['item_id','shop_id'], how='left').fillna(0)
price = price.drop(labels=['ID','item_id','shop_id'],axis=1)

## create x and y training sets
y_train = val['2015-10']
x_sales = val.drop(labels=['2015-10'],axis=1)
x_sales = x_sales.values.reshape((x_sales.shape[0], x_sales.shape[1], 1))
x_prices = price.drop(labels=['2015-10'],axis=1)
x_prices= x_prices.values.reshape((x_prices.shape[0], x_prices.shape[1], 1))
X = np.append(x_sales,x_prices,axis=2)

print(X.shape) # (214200, 33, 2) features are sale and price

y = y_train.values.reshape((214200, 1))
print("training X Shape: ",X.shape)
print("training y Shape: ",y.shape)
del y_train, x_sales; gc.collect()


## create x testing(val) sets
val = val.drop(labels=['2013-01'],axis=1)
x_test_sales = val.values.reshape((val.shape[0], val.shape[1], 1))
x_test_prices = price.drop(labels=['2013-01'],axis=1)
x_test_prices = x_test_prices.values.reshape((x_test_prices.shape[0], x_test_prices.shape[1], 1))

X_test = np.append(x_test_sales,x_test_prices,axis=2)
del x_test_sales,x_test_prices, price; gc.collect()
print("testing X Shape: ",X_test.shape)


## define our model
model_lstm = Sequential()
model_lstm.add(CuDNNLSTM(16, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model_lstm.add(Dropout(0.5))
model_lstm.add(CuDNNLSTM(32))
model_lstm.add(Dropout(0.5))
model_lstm.add(Dense(1,activation = 'relu'))
model_lstm.compile(optimizer="adam", loss='mse', metrics=["mse"])
print(model_lstm.summary())

PARAMS = {"batch_size":64, "verbose":2, "epochs":50}

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15, random_state=42, shuffle=False)
print("X Train Shape: ",X_train.shape)
print("X Valid Shape: ",X_valid.shape)
print("y Train Shape: ",y_train.shape)
print("y Valid Shape: ",y_valid.shape)

callbacks_list=[EarlyStopping(monitor="val_loss", min_delta=.001, patience=10, mode='auto')]

## train
hist = model_lstm.fit(X_train, y_train, validation_data=(X_valid, y_valid), 
                    callbacks=callbacks_list, **PARAMS)


## evaluation
best = np.argmin(hist.history["val_loss"])
print("Optimal Epoch: {}",best)
print("Train Score: {}, Val Score: {}".format(hist.history["loss"][best],hist.history["val_loss"][best]))
plt.plot(hist.history['loss'], label='train')
plt.plot(hist.history['val_loss'], label='val')
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.legend()
plt.show()
plt.savefig("train-val-MSE-curve.png")


pred = model_lstm.predict(X_test)

submission = pd.DataFrame(pred,columns=['item_cnt_month'])
submission.to_csv('lstm_submission.csv',index_label='ID')
