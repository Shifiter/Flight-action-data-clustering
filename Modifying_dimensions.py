import pandas as pd
import math
import os
import chardet

# 定义目标提取的行数
target_rows = 500

# 你的文件列表
file_list = ["flight97.csv", "flight98.csv", "flight104.csv", "flight111.csv", "flight112.csv", "flight121.csv",
             "flight122.csv"]
# 目标表头
target_headers = ["高度", "攻角", "法向加速度", "空速", "滚转角", "俯仰角"]

# 用于存储提取的数据的空DataFrame
result_df = pd.DataFrame()

# 从每个文件中按步长提取行
for file in file_list:
    # 读取CSV文件
    df = pd.read_csv(file, encoding='ANSI')

    # 获取文件中的行数
    total_rows = df.shape[0]

    # 计算步长，确保最终提取的行数为 target_rows
    step_size = max(1, math.floor(total_rows / target_rows))

    # 根据步长提取行
    sampled_df = df.iloc[::step_size]

    # 截取前 target_rows 行以确保最终行数相同
    sampled_df = sampled_df.head(target_rows)

    # 创建新的DataFrame，将每一列合并为一列
    merged_column = pd.concat([sampled_df[col] for col in target_headers], ignore_index=True)

    # 创建包含合并列的新DataFrame
    result_df = pd.DataFrame({'Flight': merged_column})

    # 构造新文件名（例如，原文件名加上“_sampled”后缀）
    new_file_name = os.path.join("data2", file.replace(".csv", "_sampled.csv"))

    # 将提取的数据保存到新的CSV文件
    result_df.to_csv(new_file_name, index=False, encoding='ANSI')





