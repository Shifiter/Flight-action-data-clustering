import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pandas import DataFrame
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

# 读取三个CSV文件
df1 = pd.read_csv('data2/flight97_sampled.csv').T
df2 = pd.read_csv('data2/flight98_sampled.csv').T
df3 = pd.read_csv('data2/flight104_sampled.csv').T
df4 = pd.read_csv('data2/flight111_sampled.csv').T
df5 = pd.read_csv('data2/flight112_sampled.csv').T
df6 = pd.read_csv('data2/flight121_sampled.csv').T
df7 = pd.read_csv('data2/flight122_sampled.csv').T

all_data = pd.concat([df1, df2, df3, df4, df5, df6, df7], ignore_index=True)  # 7 × 3000

X = all_data.values
# 存储每个K值对应的损失函数值
cost = []

# K-means聚类
kms = KMeans(n_clusters=5, init='k-means++')
data_fig = kms.fit(all_data)  # 模型拟合
centers = kms.cluster_centers_  # 计算聚类中心
labs = kms.labels_  # 为数据打标签
df_labels = DataFrame(kms.labels_)  # 将标签存放为DataFrame
# df_labels.to_excel('datalabels.xlsx')  # 输出数据标签，其实输出可有可无

# 将聚类结果为 0，1,2,3,4 的数据筛选出来 并打上标签
df_A_0 = all_data[kms.labels_ == 0]
df_A_1 = all_data[kms.labels_ == 1]
df_A_2 = all_data[kms.labels_ == 2]
df_A_3 = all_data[kms.labels_ == 3]
df_A_4 = all_data[kms.labels_ == 4]
m = np.shape(df_A_0)[1]
df_A_0.insert(df_A_0.shape[1], 'label', 0)  # 打标签
df_A_1.insert(df_A_1.shape[1], 'label', 1)
df_A_2.insert(df_A_2.shape[1], 'label', 2)
df_A_3.insert(df_A_3.shape[1], 'label', 3)
df_A_4.insert(df_A_4.shape[1], 'label', 4)
df_labels_data = pd.concat([df_A_0, df_A_1, df_A_2, df_A_3, df_A_4])  # 数据融合
df_labels_data.to_excel('data_labeled.xlsx')  # 输出带有标签的数据

# 输出最终聚类中心
df_centers = DataFrame(centers)
df_centers.to_excel('data_final_center.xlsx')

# 首先，对原数据进行 PCA 降维处理，获得散点图的横纵坐标轴数据
pca = PCA(n_components=2)  # 提取两个主成分，作为坐标轴
pca.fit(all_data)
data_pca = pca.transform(all_data)
data_pca = pd.DataFrame(data_pca, columns=['1', '2'])
data_pca.insert(data_pca.shape[1], 'labels', labs)

# centers pca 对 K-means 的聚类中心降维，对应到散点图的二维坐标系中
pca = PCA(n_components=2)
pca.fit(centers)
data_pca_centers = pca.transform(centers)
data_pca_centers = pd.DataFrame(data_pca_centers, columns=['1', '2'])

# Visualize it:
plt.figure(figsize=(8, 6))
plt.scatter(data_pca.values[:, 0], data_pca.values[:, 1], s=3, c=data_pca.values[:, 2], cmap='Accent')
plt.scatter(data_pca_centers.values[:, 0], data_pca_centers.values[:, 1], marker='o', s=55, c='#8E00FF')

# Scatter plot for each cluster center
plt.scatter(data_pca_centers.values[:, 0], data_pca_centers.values[:, 1], marker='o', s=55, c='#8E00FF')

# Annotate cluster centers
for i, center in enumerate(data_pca_centers.values):
    plt.annotate(f'Cluster {i}', (center[0], center[1]), fontsize=10, color='red')

# 添加文件名注释
for i, txt in enumerate(all_data.index):
    plt.annotate(txt, (data_pca.values[i, 0], data_pca.values[i, 1]), fontsize=6)

# 使用 KMeans 模型的 transform 方法获取每个样本到聚类中心的距离
distances_to_centers = kms.transform(all_data)
print(distances_to_centers)
# 找到每个簇中离中心点最近的点的索引
closest_point_indices = np.argmin(distances_to_centers, axis=0)

# 找到每个簇中离中心点最近的点的标签
closest_point_labels = labs[closest_point_indices]

# 打印结果
for cluster_label, closest_point_label in zip(range(5), closest_point_labels):
    print(f'Cluster {cluster_label} - Closest Point Label: {closest_point_label}')

plt.show()
