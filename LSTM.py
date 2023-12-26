import numpy
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

dataframe = pd.read_csv('./flight97.csv', usecols=["高度"], engine='python', encoding="ANSI")
dataset = dataframe.values
# 将整型变为float
dataset = dataset.astype('float32')
# 归一化
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

train_size = int(len(dataset) * 0.65)
trainlist = dataset[:train_size]
testlist = dataset[train_size:]


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
trainX, trainY = create_dataset(trainlist, look_back)
testX, testY = create_dataset(testlist, look_back)

# 重新形状训练集和测试集
trainX = torch.tensor(trainX.reshape(trainX.shape[0], trainX.shape[1], 1), dtype=torch.float32)
testX = torch.tensor(testX.reshape(testX.shape[0], testX.shape[1], 1), dtype=torch.float32)


# 定义 LSTM 模型
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 使用最后一个时间步的输出
        return out


# 创建模型实例
input_size = 1  # 输入数据的特征数量
hidden_size = 4  # LSTM 隐藏层的单元数量
output_size = 1  # 输出数据的特征数量
model = SimpleLSTM(input_size, hidden_size, output_size)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 转换为训练模式
model.train()

# 将数据传递给模型进行训练
num_epochs = 1000
for epoch in tqdm(range(num_epochs), desc='Training', unit='epoch'):
    outputs = model(trainX)
    loss = criterion(outputs, torch.tensor(trainY, dtype=torch.float32))

    predicted_labels = torch.argmax(outputs, dim=1)
    # 计算混淆矩阵
    # conf_matrix = confusion_matrix(trainY, predicted_labels.numpy())

    # 计算准确率、查全率、F1 值
    # precision, recall, f1, _ = precision_recall_fscore_support(trainY, predicted_labels.numpy(), average='binary')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        tqdm.write(
            f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        # print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 保存模型
torch.save(model.state_dict(), 'Test.pth')

# 加载保存的模型参数
model = SimpleLSTM(input_size, hidden_size, output_size)
model.load_state_dict(torch.load('Test.pth'))
model.eval()  # 设置为评估模式

# 转换测试数据为 PyTorch Tensor
trainX_tensor = torch.tensor(trainX, dtype=torch.float32)
testX_tensor = torch.tensor(testX, dtype=torch.float32)

# 使用模型进行预测
with torch.no_grad():
    trainPredict_tensor = model(trainX_tensor)
    testPredict_tensor = model(testX_tensor)

# 反归一化
trainPredict = scaler.inverse_transform(trainPredict_tensor.numpy())
trainY = scaler.inverse_transform(trainY)
testPredict = scaler.inverse_transform(testPredict_tensor.numpy())
testY = scaler.inverse_transform(testY)

# 绘制图表
plt.plot(trainY, label='True Values')
plt.plot(trainPredict[1:], label='Predicted Values')
plt.title('Training Set: True vs Predicted')
plt.legend()
plt.show()

plt.plot(testY, label='True Values')
plt.plot(testPredict[1:], label='Predicted Values')
plt.title('Testing Set: True vs Predicted')
plt.legend()
plt.show()
