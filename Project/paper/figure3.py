import matplotlib.pyplot as plt
import numpy as np
import json
import os

x = np.arange(1, 16) # x轴数据

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
file_path = os.path.join(current_dir, '../BayesianLog','figure.json')

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
except FileNotFoundError:
    print(f"错误: 文件 '{file_path}' 未找到。请确保 JSON 文件在正确的路径下。")
    exit() # 如果文件未找到，则退出程序
except json.JSONDecodeError:
    print(f"错误: 无法解析 '{file_path}' 中的 JSON 数据。请检查文件内容是否为有效的 JSON 格式。")
    exit()

_json_data = [json_data[i]['params'] for i in range(len(json_data)) if 'params' in json_data[i]]
json_data = _json_data

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

# 创建一个3行2列的子图布局，并设置图表大小
fig, axes = plt.subplots(3, 2, figsize=(10, 8)) # 调整figsize以适应六个图

# 设置整个图的背景为白色（如果需要的话，默认通常是透明）
fig.patch.set_facecolor('white')

# 左上图: k_S
axes[0, 0].plot(x, numpy_arrays['W_d'], color='purple', marker='o', markersize=5)
axes[0, 0].set_ylabel('$W_d$')
axes[0, 0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) # 隐藏X轴刻度及标签
# axes[0, 0].set_ylim(1.5, 8.5) # 可选：根据数据调整Y轴范围

# 右上图: d_GIP^max
axes[0, 1].plot(x, numpy_arrays['W_p'], color='royalblue', marker='o', markersize=5)
axes[0, 1].set_ylabel('$W_p$')
axes[0, 1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) # 隐藏X轴刻度及标签
# axes[0, 1].set_ylim(0, 0.16) # 可选：根据数据调整Y轴范围

# 左中图: v_N^max
axes[1, 0].plot(x, numpy_arrays['W_v'], color='cornflowerblue', marker='o', markersize=5)
axes[1, 0].set_ylabel('$W_v$')
axes[1, 0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) # 隐藏X轴刻度及标签
# axes[1, 0].set_ylim(0, 270) # 可选：根据数据调整Y轴范围

# 右中图: l_PA_max
axes[1, 1].plot(x, numpy_arrays['W_dp'], color='steelblue', marker='o', markersize=5)
axes[1, 1].set_ylabel('$W_dp$')
axes[1, 1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) # 隐藏X轴刻度及标签
# axes[1, 1].set_ylim(0.2, 0.5) # 可选：根据数据调整Y轴范围

# 左下图: d_plane^min
axes[2, 0].plot(x, numpy_arrays['W_dv'], color='lightseagreen', marker='o', markersize=5)
axes[2, 0].set_ylabel('$W_dv$')
#axes[2, 0].set_ylim(0, 0.0035) # 可选：根据数据调整Y轴范围

# 右下图: d_plane^max
axes[2, 1].plot(x, numpy_arrays['W_pv'], color='seagreen', marker='o', markersize=5)
axes[2, 1].set_ylabel('$W_pv$')
#axes[2, 1].set_ylim(0.015, 0.08) # 可选：根据数据调整Y轴范围

# 调整子图之间的间距
plt.tight_layout()

# 显示图表
plt.show()
