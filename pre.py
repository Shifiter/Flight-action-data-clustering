import numpy
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

dataframe = pd.read_csv('./flight122.csv', usecols=["高度"], engine='python', encoding="ANSI")
dataset = dataframe.values
# 将整型变为float
dataset = dataset.astype('float32')
# 归一化
scaler = MinMaxScaler(feature_range=(0, 1))
testlist = scaler.fit_transform(dataset)


def create_dataset(dataset, look_back):
    # 这里的look_back与timestep相同
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return numpy.array(dataX), numpy.array(dataY)


# look_back用于设置多少个训练数据来预测后一个数
look_back = 5
testX, testY = create_dataset(testlist, look_back)

# 重新形状训练集和测试集
testX = torch.tensor(testX.reshape(testX.shape[0], testX.shape[1], 1), dtype=torch.float32)

# 加载保存的模型参数
model = torch.load('Test.pth')

model.eval()  # 设置为评估模式

# 转换测试数据为 PyTorch Tensor

testX_tensor = torch.tensor(testX, dtype=torch.float32)

# 使用模型进行预测
with torch.no_grad():
    testPredict_tensor = model(testX_tensor)

# 反归一化
testPredict = scaler.inverse_transform(testPredict_tensor.numpy())
testY = scaler.inverse_transform(testY)

# 绘制图表
plt.plot(testY, label='True Values')
plt.plot(testPredict[1:], label='Predicted Values')
plt.title('Testing Set: True vs Predicted')
plt.legend()
plt.show()
