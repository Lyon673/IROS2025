import matplotlib.pyplot as plt
import numpy as np
import json
import os

x = np.arange(1, 16) # x轴数据

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
file_path = os.path.join(current_dir, '../BayesianLog','scores.json')

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
except FileNotFoundError:
    print(f"错误: 文件 '{file_path}' 未找到。请确保 JSON 文件在正确的路径下。")
    exit() # 如果文件未找到，则退出程序
except json.JSONDecodeError:
    print(f"错误: 无法解析 '{file_path}' 中的 JSON 数据。请检查文件内容是否为有效的 JSON 格式。")
    exit()

# 获取所有可能的键（数据类型）
all_keys = json_data[0].keys()

# 创建一个字典来存储每个键对应的数值列表
data_dict = {key: [] for key in all_keys}

# 遍历 JSON 数据，将每个键的数值收集到对应的列表中
for item in json_data:
    for key in all_keys:
        data_dict[key].append(item[key])

# 将每个列表转换为 NumPy 数组
numpy_arrays = {key: np.array(values) for key, values in data_dict.items()}

# --- 绘制子图 ---

fig, axes = plt.subplots(3, 2, figsize=(10, 8)) # 创建2x2的子图布局，并设置图表大小
fig.patch.set_facecolor('white') # 设置整个图的背景为白色

# 左上图：I_G
axes[0, 0].plot(x, numpy_arrays['gracefulness'], color='purple', marker='o')
axes[0, 0].set_ylabel('Gracefulness') # 设置y轴标签，使用LaTeX格式
# 调整y轴范围以更好地展示数据
axes[0, 0].set_ylim(1.55, 2.0)
axes[0, 0].set_xticks([]) # 如果不需要x轴刻度，可以取消注释

# 右上图：Number of clutching
axes[0, 1].bar(x, numpy_arrays['clutch_times'], color='lightsteelblue')
axes[0, 1].set_ylabel('Number of clutching')
axes[0, 1].set_ylim(0, 12) # 调整y轴范围
axes[0, 1].set_xticks([]) # 如果不需要x轴刻度，可以取消注释


subjective_score = (numpy_arrays['total_score'] - numpy_arrays['clutch_times_score'] - numpy_arrays['gracefulness_score'] \
    - numpy_arrays['smoothness_score'] - numpy_arrays['total_distance_score'] - numpy_arrays['total_time_score'])

# 左下图：I_S
axes[1, 0].plot(x, numpy_arrays['smoothness'], color='cornflowerblue', marker='o')
axes[1, 0].set_ylabel('Smoothness') # 设置y轴标签，使用LaTeX格式
#axes[1, 0].set_ylim(3.8, 9.8) # 调整y轴范围
axes[1, 0].set_ylim(5.5, 9.5) # 调整y轴范围
axes[1, 0].set_xticks([]) # 如果不需要x轴刻度，可以取消注释

# 右下图：Completion time
axes[1, 1].plot(x, numpy_arrays['total_time'], color='steelblue', marker='o')
axes[1, 1].set_ylabel('Completion time')
axes[1, 1].set_ylim(30, 110) # 调整y轴范围
axes[1, 1].set_xticks([]) # 如果不需要x轴刻度，可以取消注释

# 左下图: d_plane^min
axes[2, 0].plot(x, subjective_score, color='lightseagreen', marker='o', markersize=5)
axes[2, 0].set_ylabel('Subjective score')
axes[2, 0].set_ylim(5, 50) # 可选：根据数据调整Y轴范围
axes[2, 0].set_xticks([]) # 如果不需要x轴刻度，可以取消注释

# 右下图: d_plane^max
axes[2, 1].plot(x, numpy_arrays['total_score'], color='seagreen', marker='o', markersize=5)
axes[2, 1].set_ylabel('Final score')
axes[2, 1].set_ylim(15, 90) # 可选：根据数据调整Y轴范围
axes[2, 1].set_xticks([]) # 如果不需要x轴刻度，可以取消注释

# 调整子图之间的间距
plt.tight_layout()

# 显示图表
plt.show()