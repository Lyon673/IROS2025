import numpy as np
import matplotlib.pyplot as plt

# 文件路径（根据实际路径修改）
_y1 = np.load('/home/lambda/surgical_robotics_challenge/scripts/surgical_robotics_challenge/iros/data/distance_data.npy', allow_pickle=True)
_y2 = np.load('/home/lambda/surgical_robotics_challenge/scripts/surgical_robotics_challenge/iros/data/ipaL_data.npy', allow_pickle=True)
_y3 = np.load('/home/lambda/surgical_robotics_challenge/scripts/surgical_robotics_challenge/iros/data/psm_twist.npy', allow_pickle=True)
_y4 = np.load('/home/lambda/surgical_robotics_challenge/scripts/surgical_robotics_challenge/iros/data/distance_plane_data.npy', allow_pickle=True)
__y3 = [(x[0].x**2+x[0].y**2+x[0].z**2)**0.5 for x in _y3]
"""y1 = _y1[95:]
y3 = __y3[95:]
__y2 = [x for x in _y2 if (x <0.3 and x != 0.23)]
y2 = __y2+[0.23708368105606598]*63
y4 = _y4[95:]"""

# 读取四个 .npy 文件
data_sets = [_y1, _y2, __y3, _y4]

# 假设所有数据的行数（长度）相同，计算时间轴（索引 × 5 ms）
n_samples = len(data_sets[0])  # 取第一个文件的数据长度
t = np.arange(n_samples)  # 时间轴，单位：ms（根据你的需求改为 *5）

# 创建 2×2 子图布局
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# 将 axes 展平为 1D 数组以便迭代
axes = axes.flatten()

# 四组数据的名称（可以根据实际数据修改）
data_names = ['$d_{GP}$', '$IPA_L$', '$v_M$', '$d_{PL}$']

# 定义需要标记的 time 区间（以索引为单位）
"""approaching_regions = [
    (20, 80),  # 第一个区间
    (245, 305),   # 第二个区间
    (440, 500)
]
needle_pick_regions = [
    (170, 200),   # 第二个区间
    (350, 380),
]"""
"""
approaching_regions = [
    (20, 80),  # 第一个区间
    (340, 400),   # 第二个区间
    (535, 595)
]
needle_pick_regions = [
    (265, 295),   # 第二个区间
    (445, 475),
]"""
approaching_regions = [
    (10, 60),  # 第一个区间
    (150, 210),   # 第二个区间
    (345, 405)
]
needle_pick_regions = [
    (75, 105),   # 第二个区间
    (255, 285),
]

# 在每个子图中绘制折线图并添加矩形区域
#colors = [ '#66B2FF', '#FF6666',  '#66CCAA',  '#FFAA66' ]
colors = [ '#0089ba' , '#4e8397', '#008e9b',  '#008f7a' ]
for i in range(4):
    # 绘制折线图
    if i != 1:
        axes[i].plot(t, data_sets[i], label=data_names[i], color=colors[i], linewidth=2)
    else:
        axes[i].scatter(t, data_sets[i], label=data_names[i], color=colors[i], s=2)

    # 为每个子图添加浅黄色矩形区域
    for start_idx, end_idx in approaching_regions:
        # 将索引转换为时间（t）
        start_time = start_idx  # 假设时间单位是 ms
        end_time = end_idx
        # 添加矩形（fill_between 或 axvspan）
        axes[i].axvspan(start_time, end_time, facecolor='gray', alpha=0.1, edgecolor='none')

    # 为每个子图添加浅黄色矩形区域
    for start_idx, end_idx in needle_pick_regions:
        # 将索引转换为时间（t）
        start_time = start_idx  # 假设时间单位是 ms
        end_time = end_idx
        # 添加矩形（fill_between 或 axvspan）
        axes[i].axvspan(start_time, end_time, facecolor='orange', alpha=0.1, edgecolor='none')

    # 设置标题和标签
    # axes[i].set_title(data_names[i])  # 设置标题（已注释）
    axes[i].set_xlabel('Time')   # 设置横轴标签
    axes[i].set_ylabel(data_names[i], fontsize = 16)       # 设置纵轴标签

    if i == 1:
        axes[i].set_ylim(0.22, 0.25)
    
    # # 添加图例（仅在第一个子图显示，以避免重复）
    # if i == 0:
    #     axes[i].legend()

# 调整布局，避免重叠
plt.tight_layout()

# 保存图像（可选）
plt.savefig('plot_2x2_highlighted.png', bbox_inches='tight')

# 显示图像
plt.show()