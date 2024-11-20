# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 21:18:02 2024

@author: DELL
"""

#指数平滑
import numpy as np 
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

##导入数据
data = pd.read_csv("E:/桌面/2024mathorcup/2024年MathorCup大数据挑战赛-初赛赛题/2024年MathorCup大数据挑战赛-赛道B初赛/附件/附件1.csv", encoding='gbk')
data1 = data[data['品类'] == 'category1'].copy()
data1['月份'] = pd.to_datetime(data1['月份'], errors='coerce')
data1 = data1.dropna(subset=['月份'])
data1 = data1.sort_values(by='月份', ascending=True).reset_index(drop=True)

##数据划分
train_data = data1.iloc[:-3]
test_data = data1.iloc[-3:]
train_series = train_data['库存量'].values
test_series = test_data['库存量'].values

##一次指数平滑
model1 = SimpleExpSmoothing(train_series).fit(smoothing_level=0.5, optimized=False)
forecast1 = model1.forecast(len(test_series))
mape1 = mean_absolute_percentage_error(test_series, forecast1)

##二次指数平滑
model2 = Holt(train_series).fit(smoothing_level=0.3, smoothing_trend=0.1, optimized=False)
forecast2 = model2.forecast(len(test_series))
mape2 = mean_absolute_percentage_error(test_series, forecast2)

##三次指数平滑
model3 = ExponentialSmoothing(train_series, seasonal_periods=3, trend='add', seasonal='add').fit(smoothing_level=0.3, smoothing_trend=0.1, smoothing_seasonal=0.2, optimized=False)
forecast3 = model3.forecast(len(test_series))
mape3 = mean_absolute_percentage_error(test_series, forecast3)

##最优指数平滑次数
mapes = [mape1, mape2, mape3]
models = [model1, model2, model3]
for i in range(0, len(mapes)):
    if mapes[i] == min(mapes):
        ES_model = models[i]
ES_forecast = ES_model.forecast(len(test_series))
ES_mape = mean_absolute_percentage_error(test_series, ES_forecast)
































































































