#coding:utf-8
#导入warnings包，利用过滤器来实现忽略警告语句。
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno


# 导入数据集
data_path = './data/'
train_data = pd.read_csv(data_path + 'used_car_train_20200313.csv', sep=' ')
test_data = pd.read_csv(data_path + 'used_car_testB_20200421.csv', sep=' ')
print('Train data shape:',train_data.shape)
print('TestA data shape:',test_data.shape)
print(train_data.info())

all_data = pd.concat([train_data, test_data])
all_data['notRepairedDamage'] = all_data['notRepairedDamage'].replace('-', 0).astype('float16')
# 用众数来填补null值
all_data = all_data.fillna(all_data.mode().iloc[0,:])
print('concat_data shape:',all_data.shape)