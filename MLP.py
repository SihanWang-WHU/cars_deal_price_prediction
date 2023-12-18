# https://tianchi.aliyun.com/notebook/103212

#coding:utf-8
#导入warnings包，利用过滤器来实现忽略警告语句。
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum()
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum()
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

def print_des_to_csv(df, fp = "description.csv"):
    """
    生成并将数据框的描述性统计信息写入 CSV 文件。
    """
    description = df.describe()
    description.to_csv(fp, mode='w')

# 导入数据集
data_path = './data/'
train_data = reduce_mem_usage(pd.read_csv(data_path + 'used_car_train_20200313.csv', sep=' '))
test_data = reduce_mem_usage(pd.read_csv(data_path + 'used_car_testB_20200421.csv', sep=' '))
print('Train data shape:',train_data.shape)
print('TestB data shape:',test_data.shape)
# print(train_data.info())

##################### 数据预处理 #####################
concat_data = pd.concat([train_data,test_data])
concat_data['notRepairedDamage'] = concat_data['notRepairedDamage'].replace('-',0).astype('float16')
# 用众数来填补null值
concat_data = concat_data.fillna(concat_data.mode().iloc[0,:])
print('concat_data shape:',concat_data.shape)
# log变换price值（否则是短尾分布）
concat_data['price'] = np.log(concat_data['price'])
plt.hist(concat_data['price'],color='red')
plt.title('car price distribution after log')

#截断异常值
concat_data['power'][concat_data['power']>600] = 600
concat_data['power'][concat_data['power']<1] = 1
concat_data['v_13'][concat_data['v_13']>6] = 6
concat_data['v_14'][concat_data['v_14']>4] = 4

numeric_features = ['power', 'kilometer', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6',
                    'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12', 'v_13','v_14', 'price' ]
price_numeric = concat_data[numeric_features]
correlation = price_numeric.corr()
f , ax = plt.subplots()
plt.title('Correlation of Numeric Features with Price',y=1,size=16)
sns.heatmap(correlation, square = True, vmax=0.8)

# 它将 train_data DataFrame 中的每个数值特征转换为两列：一列是变量名（'variable'），另一列是值（'value'）。
num_f = pd.melt(concat_data, value_vars=numeric_features)
# 这里使用 seaborn 的 FacetGrid 来创建一个网格，每个数值特征一个小图。col="variable" 表示每个小图对应 f DataFrame 中的一个特征
g = sns.FacetGrid(num_f, col="variable",  col_wrap=6, sharex=False, sharey=False)
# 在 FacetGrid 的每个小图中绘制一个分布图（直方图加密度曲线），每个图表示 f DataFrame 中的一个特征的分布。
g = g.map(sns.distplot, "value")
plt.show()

# 做新的交叉性的特征 其中v_i特征之内是线性相加
# v_i和其他的特征之间是相乘（numeric 和 none-numeric）
for i in ['v_' +str(i) for i in range(14)]:
    for j in ['v_' +str(i) for i in range(14)]:
        concat_data[str(i)+'+'+str(j)] = concat_data[str(i)]+concat_data[str(j)]
for i in ['model','brand', 'bodyType', 'fuelType','gearbox', 'power', 'kilometer', 'notRepairedDamage', 'regionCode']:
    for j in ['v_' +str(i) for i in range(14)]:
        concat_data[str(i)+'*'+str(j)] = concat_data[i]*concat_data[j]
print(concat_data.shape)
print_des_to_csv(concat_data)