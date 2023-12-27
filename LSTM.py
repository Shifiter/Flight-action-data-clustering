import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

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
# model = SimpleLSTM(input_size, hidden_size, output_size)
#
# # 定义损失函数和优化器
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# # 转换为训练模式
# model.train()
#
# # 将数据传递给模型进行训练
# num_epochs = 1000
# for epoch in tqdm(range(num_epochs), desc='Training', unit='epoch'):
#     outputs = model(trainX)
#     loss = criterion(outputs, torch.tensor(trainY, dtype=torch.float32))
#
#     predicted_labels = torch.argmax(outputs, dim=1)
#     # 计算混淆矩阵
#     # conf_matrix = confusion_matrix(trainY, predicted_labels.numpy())
#
#     # 计算准确率、查全率、F1 值
#     # precision, recall, f1, _ = precision_recall_fscore_support(trainY, predicted_labels.numpy(), average='binary')
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     if (epoch + 1) % 10 == 0:
#         tqdm.write(
#             f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.10f}')
#         # print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
#
# # 保存模型
# torch.save(model.state_dict(), 'Test.pth')

# 加载保存的模型参数
model = SimpleLSTM(input_size, hidden_size, output_size)
model.load_state_dict(torch.load('Test.pth'))
model.eval()  # 设置为评估模式

# 转换测试数据为 PyTorch Tensor
trainX_tensor = torch.as_tensor(trainX, dtype=torch.float32)
testX_tensor = torch.as_tensor(testX, dtype=torch.float32)

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
# plt.plot(trainY, label='True Values')
# plt.plot(trainPredict[1:], label='Predicted Values')
# plt.title('Training Set: True vs Predicted')
# plt.legend()
# plt.show()
#
# plt.plot(testY, label='True Values')
# plt.plot(testPredict[1:], label='Predicted Values')
# plt.title('Testing Set: True vs Predicted')
# plt.legend()
# plt.show()

# 获取测试结果和预测结果的差值e
e = [abs(y_h - y_t[0]) for y_h, y_t in zip(testPredict, testY)]
# 对e进行加权平滑
smoothing_window = 105
e_s = list(pd.DataFrame(e).ewm(span=smoothing_window).mean().values.flatten())

batch_size = 20
window_size = 20
# 找到窗口数目
# print(testY.shape[0])
num_windows = int((testY.shape[0] - (batch_size * window_size)) / batch_size)

print(num_windows)
print(batch_size, window_size)
print(testY.shape)


# error_buffer是异常点周围被判定为异常区间的范围
def get_anomalies(window_e_s, error_buffer, inter_range, chan_std):
    mean = np.mean(window_e_s)
    sd = np.std(window_e_s)
    i_anom = []
    E_seq = []
    epsilon = mean + 2.5 * sd
    # 如果太小则忽略
    if not (sd > (.05 * chan_std) or max(window_e_s) > (.05 * inter_range)) or not max(window_e_s) > 0.05:
        return i_anom

    for x in range(0, len(window_e_s)):
        anom = True
        # 进行check  大于整体高低差的0.05
        if not window_e_s[x] > epsilon or not window_e_s[x] > 0.05 * inter_range:
            anom = False

        if anom:
            for b in range(0, error_buffer):

                if not x + b in i_anom and not x + b >= len(window_e_s):
                    i_anom.append(x + b)

                if not x - b in i_anom and not x - b < 0:
                    i_anom.append(x - b)
    # 进行序列转换
    i_anom = sorted(list(set(i_anom)))
    # groups = [list(group) for group in mit.consecutive_groups(i_anom)]
    # E_seq = [(g[0], g[-1]) for g in groups if not g[0] == g[-1]]
    return i_anom


# 画出原始曲线
plt.plot(e_s, label='Original Data', color='blue')
all_original_indices = []
# 得到窗口e_s
for i in range(1, num_windows + 2):
    prior_idx = (i - 1) * (batch_size)
    # 前面有i-1个batch size
    idx = (window_size * batch_size) + ((i - 1) * batch_size)

    if i == num_windows + 1:
        # 因为最后一个加的幅度不满于config.batchsize
        idx = testY.shape[0]
    window_e_s = e_s[prior_idx:idx]

    perc_high, perc_low = np.percentile(window_e_s, [95, 5])
    # window_e_s的高低差和方差
    inter_range = perc_high - perc_low
    chan_std = np.std(window_e_s)

    # 对当前窗口进行异常检测
    anomalies = get_anomalies(window_e_s, 400, inter_range, chan_std)

    # 使用黄色填充 window_e_s 的范围
    plt.fill_between(range(prior_idx, idx), window_e_s, color='yellow', alpha=0.3)

    # 使用红色标注异常点
    # 使用红色标注异常点
    original_indices = [prior_idx + anomaly for anomaly in anomalies]
    all_original_indices.extend(original_indices)
    plt.scatter(original_indices, [window_e_s[i] for i in anomalies], color='red')

# 设置标签并显示图
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Original Data with Window and Anomaly Regions')
plt.legend()
plt.show()

plt.plot(testY, label='testY', color='blue')
plt.scatter(all_original_indices, [testY[i] for i in all_original_indices], color='red')
plt.title('TestY Data with Anomalies')
plt.legend()
plt.show()
