import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import socket
import Project.ipa as ipa
import pywt

_ipa = np.load('/home/lambda/Documents/experiment_data/useful/4/ipaL_data.npy', allow_pickle=True)[90:500]

for i in range(len(_ipa)):
    if _ipa[i] < 0.5:
        _ipa[i] = _ipa[i-1]
        
_v = np.load('/home/lambda/Documents/experiment_data/useful/4/psm_velocity_data.npy', allow_pickle=True)[90:500]
_d = np.load('/home/lambda/Documents/experiment_data/useful/4/distance_data.npy', allow_pickle=True)[90:500]
_s = np.load('/home/lambda/Documents/experiment_data/useful/4/scale_data.npy', allow_pickle=True)[90:500]

# for i in range(len(_ipa)):
#     if _ipa[i] != 1 and _ipa[i-1] == 1:
#         print(f"index:{i}")

timestamps = range(len(_ipa))
print(len(timestamps))

data_sets = [_d, _ipa, _v, _s]



# 创建 2×2 子图布局
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# 将 axes 展平为 1D 数组以便迭代
axes = axes.flatten()

# 四组数据的名称（可以根据实际数据修改）
data_names = ['$d_{GP}$', '$IPA$', '$v_M$', '$scale$']

# 定义需要标记的 time 区间（以索引为单位）

approaching_regions = [
    (10, 100),  # 第一个区间
    (248, 340),   # 第二个区间
]

needle_pick_regions = [
    (105, 180),   # 第二个区间
    (353, 385),
]

# 在每个子图中绘制折线图并添加矩形区域
#colors = [ '#66B2FF', '#FF6666',  '#66CCAA',  '#FFAA66' ]
colors = [ '#0089ba' , '#4e8397', '#008e9b',  '#008f7a' ]
for i in range(4):
    # 绘制折线图
    if i != 3 and i != 1:  
        n_samples = len(data_sets[i])  
        t = np.arange(n_samples)/60  # 时间轴，单位：ms（根据你的需求改为 *5）
        axes[i].plot(t, data_sets[i], label=data_names[i], color=colors[i], linewidth=2)
    else:
        n_samples = len(data_sets[i])  
        t = np.arange(n_samples)/60   # 时间轴，单位：ms（根据你的需求改为 *5）
        axes[i].scatter(t, data_sets[i], label=data_names[i], color=colors[i], s=2)

    # 为每个子图添加浅黄色矩形区域
    for start_idx, end_idx in approaching_regions:
        # 将索引转换为时间（t）
        start_time = start_idx/60  # 假设时间单位是 ms
        end_time = end_idx/60
        # 添加矩形（fill_between 或 axvspan）
        axes[i].axvspan(start_time, end_time, facecolor='gray', alpha=0.1, edgecolor='none')

    # 为每个子图添加浅黄色矩形区域
    for start_idx, end_idx in needle_pick_regions:
        # 将索引转换为时间（t）
        start_time = start_idx/60  # 假设时间单位是 ms
        end_time = end_idx/60 
        # 添加矩形（fill_between 或 axvspan）
        axes[i].axvspan(start_time, end_time, facecolor='orange', alpha=0.1, edgecolor='none')

    # 设置标题和标签
    # axes[i].set_title(data_names[i])  # 设置标题（已注释）
    axes[i].set_xlabel('$Time(s)$')   # 设置横轴标签
    axes[i].set_ylabel(data_names[i], fontsize = 16)       # 设置纵轴标签


# 调整布局，避免重叠
plt.tight_layout()

# 显示图像
plt.show()

