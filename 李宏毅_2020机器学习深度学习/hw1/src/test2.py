# https://zhuanlan.zhihu.com/p/136329137

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 载入数据
# data = pd.read_csv('gdrive/My Drive/hw1-regression/train.csv', header = None, encoding = 'big5')
data = pd.read_csv('../resource/train.csv', encoding='big5')

# 处理训练数据
data = data.iloc[:, 3:] # 从第四列到最后一列的所有行
# iloc[   :  ,  : ]    前面的冒号就是取行数，后面的冒号是取列数
data[data == 'NR'] = 0  # 数据值为'NR'的全部替换为'0'
raw_data = data.to_numpy()

month_data = {}
for month in range(12):
    sample = np.empty([18, 480])
    for day in range(20):
        sample[:, day * 24: (day + 1) * 24] = raw_data[18 * (20 * month + day): 18 * (20 * month + day + 1), :]
    month_data[month] = sample

# 训练数据的输入x和答案y
x = np.empty([12 * 471, 18 * 9], dtype=float)
y = np.empty([12 * 471, 1], dtype=float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            x[month * 471 + day * 24 + hour, :] = month_data[month][:, day * 24 + hour: day * 24 + hour + 9].reshape(1,
                                                                                                                     -1)  # vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
            y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9]  # value

mean_x = np.mean(x, axis=0)  # 18 * 9
std_x = np.std(x, axis=0)  # 18 * 9
for i in range(len(x)):  # 12 * 471
    for j in range(len(x[0])):  # 18 * 9
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]

# 分出自测数据
import math

x_train_set = x[: math.floor(len(x) * 0.8), :]
y_train_set = y[: math.floor(len(y) * 0.8), :]
x_validation = x[math.floor(len(x) * 0.8):, :]
y_validation = y[math.floor(len(y) * 0.8):, :]

# 训练
loss_array = np.empty([4, 1000], dtype=float)


def train(x, y, learning_rate, model_id):
    dim = 18 * 9 + 1
    w = np.zeros([dim, 1])
    x = np.concatenate((np.ones([int(12 * 471 * 0.8), 1]), x), axis=1).astype(float)
    # learning_rate = 100
    iter_time = 1000
    adagrad = np.zeros([dim, 1])
    eps = 0.0000000001
    for t in range(iter_time):
        loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2)) / 471 / 12)  # rmse
        loss_array[model_id - 1, t] = loss
        if (t % 100 == 0):
            print(str(t) + ":" + str(loss))
        gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y)  # dim*1
        adagrad += gradient ** 2
        w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
    np.save('weight_md' + str(model_id) + '.npy', w)


# 处理测试数据
testdata = pd.read_csv('../resource/test.csv', header=None, encoding='big5')
test_data = testdata.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()
test_x = np.empty([240, 18 * 9], dtype=float)
for i in range(240):
    test_x[i, :] = test_data[18 * i: 18 * (i + 1), :].reshape(1, -1)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis=1).astype(float)

# 训练模型
train(x_train_set, y_train_set, 0.5, 1)
train(x_train_set, y_train_set, 0.37, 2)
train(x_train_set, y_train_set, 0.2, 3)
train(x_train_set, y_train_set, 2, 4)

# 画图
x_axis = np.linspace(0, 1000, 1000)
plt.figure()
color = ('red', 'orange', 'blue', 'green')
for i in range(4):
    plt.plot(x_axis, loss_array[i, 0:1000], color=color[i])
plt.show()

# 计算预测结果
x_validation = np.concatenate((np.ones([1131, 1]), x_validation), axis=1).astype(float)
ans_y_array = np.empty([4, int(len(x) * 0.2 + 1)], dtype=float)
for i in range(4):
    w = np.load('weight_md' + str(i + 1) + '.npy')
    ans_y = np.dot(x_validation, w)
    for j in range(len(w)):
        ans_y_array[i][j] = ans_y[j][0]

# 自测
print(y_validation.shape)
print(ans_y_array.shape)
for i in range(4):
    loss_validation = np.sqrt(np.sum(np.power(ans_y_array[i, :] - np.transpose(y_validation), 2)) / 471 / 12)
    print(str(loss_validation) + ',')

#生成提交文件
import csv
with open('submit.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
        print(row)

sys.exit()