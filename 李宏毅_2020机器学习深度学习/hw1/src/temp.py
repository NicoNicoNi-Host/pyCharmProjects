import sys
import  pprint
import  torch
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


r = range(10, )
print(list(r))
for i in range(10):
    print(i)

x = torch.empty(5, 3)
print(x)


# # 载入数据
# # data = pd.read_csv('gdrive/My Drive/hw1-regression/train.csv', header = None, encoding = 'big5')
data = pd.read_csv('../resource/train.csv', encoding='big5')
#
# # 处理训练数据
# data = data.iloc[:, 3:] # 从第四列到最后一列的所有行
# # iloc[   :  ,  : ]    前面的冒号就是取行数，后面的冒号是取列数
# data = data.iloc[1:13, :]
# data[data == 'NR'] = 0
# # raw_data = data.to_numpy()
# print(data)

sys.exit()

