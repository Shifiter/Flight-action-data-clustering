import torch
import torch.nn as nn


# 定义生成网络（Generator）
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(Generator, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.relu(out)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out


# 定义判别网络（Discriminator）
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(Discriminator, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.relu(out)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out


# 定义整个 LSTM-GAN 模型
class LSTM_GAN(nn.Module):
    def __init__(self, latent_dim, gen_hidden_size, dis_hidden_size, window_size):
        super(LSTM_GAN, self).__init__()
        self.generator = Generator(latent_dim, gen_hidden_size, window_size, num_layers=20)
        self.discriminator = Discriminator(window_size, dis_hidden_size, 1, num_layers=20)
        self.mask_layer = nn.Dropout(0.5)

    def forward(self, z):
        generated_sequence = self.generator(z)
        masked_sequence = self.mask_layer(generated_sequence)
        real_or_fake = self.discriminator(masked_sequence)
        return generated_sequence, real_or_fake


# 定义测试函数
def evaluate_model(lstm_gan, test_data, ground_truth):
    lstm_gan.eval()

    # 生成器的输出
    with torch.no_grad():
        generated_data, _ = lstm_gan(z)  # 假设 z 是一些随机潜在向量

    # 定义二元分类阈值（可根据具体情况调整）
    threshold = 0.5

    # 将生成的数据与阈值进行比较，以获得二元分类结果
    binary_predictions = (generated_data > threshold).float()

    # 计算混淆矩阵
    confusion_matrix = torch.zeros(2, 2)
    confusion_matrix[0, 0] = torch.sum((binary_predictions == 0) & (ground_truth == 0)).item()  # 真负
    confusion_matrix[1, 0] = torch.sum((binary_predictions == 0) & (ground_truth == 1)).item()  # 假负
    confusion_matrix[0, 1] = torch.sum((binary_predictions == 1) & (ground_truth == 0)).item()  # 假正
    confusion_matrix[1, 1] = torch.sum((binary_predictions == 1) & (ground_truth == 1)).item()  # 真正

    # 计算性能指标
    accuracy = (confusion_matrix[0, 0] + confusion_matrix[1, 1]) / confusion_matrix.sum()
    precision = confusion_matrix[1, 1] / (confusion_matrix[0, 1] + confusion_matrix[1, 1])
    recall = confusion_matrix[1, 1] / (confusion_matrix[1, 0] + confusion_matrix[1, 1])
    f1_score = 2 * (precision * recall) / (precision + recall)

    # 打印性能指标
    print(
        f'Test - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}')


# 示例用法
latent_dim = 20
gen_hidden_size = 50
dis_hidden_size = 50
window_size = 30

# 创建模型
lstm_gan = LSTM_GAN(latent_dim, gen_hidden_size, dis_hidden_size, window_size)

# 定义优化器和损失函数
optimizer_gen = torch.optim.Adam(lstm_gan.generator.parameters(), lr=0.001)
optimizer_dis = torch.optim.Adam(lstm_gan.discriminator.parameters(), lr=0.001)

# 训练示例（需要适应您的数据和训练过程）
num_epochs = 2000
test_interval = 50  # 每训练50次进行一次测试

for epoch in range(num_epochs):
    # 生成随机潜在向量
    z = torch.randn(100, latent_dim)

    # 计算生成器的 Wasserstein 距离损失并优化生成器
    optimizer_gen.zero_grad()
    generated_sequence, real_or_fake = lstm_gan(z)
    gen_loss = -torch.mean(real_or_fake)
    gen_loss.backward()
    optimizer_gen.step()

    # 计算判别器的 Wasserstein 距离损失并优化判别器
    optimizer_dis.zero_grad()
    real_data = torch.randn(100, window_size)
    real_or_fake_real = lstm_gan(real_data)[1]
    dis_loss_real = -torch.mean(real_or_fake_real)

    fake_data = generated_sequence.detach()
    real_or_fake_fake = lstm_gan(fake_data)[1]
    dis_loss_fake = torch.mean(real_or_fake_fake)

    dis_loss = dis_loss_real + dis_loss_fake
    dis_loss.backward()
    optimizer_dis.step()

    # 每训练50次进行一次测试
    if (epoch + 1) % test_interval == 0:
        evaluate_model(lstm_gan, test_data, ground_truth)

    # 打印训练信息
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Gen Loss: {gen_loss.item():.4f}, Dis Loss: {dis_loss.item():.4f}')

# 在训练结束后，您可以使用生成器生成新的序列
with torch.no_grad():
    new_z = torch.randn(1, latent_dim)
    generated_sequence, _ = lstm_gan(new_z)
    print("Generated Sequence:", generated_sequence.numpy())
