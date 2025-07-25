import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 假设你的数组是 arr（加载数据部分保持不变）
_y1 = np.load('/home/lambda/surgical_robotics_challenge/scripts/surgical_robotics_challenge/iros/data/distance_data.npy', allow_pickle=True)
_y2 = np.load('/home/lambda/surgical_robotics_challenge/scripts/surgical_robotics_challenge/iros/data/ipaL_data.npy', allow_pickle=True)
_y3 = np.load('/home/lambda/surgical_robotics_challenge/scripts/surgical_robotics_challenge/iros/data/psm_twist.npy', allow_pickle=True)
_y4 = np.load('/home/lambda/surgical_robotics_challenge/scripts/surgical_robotics_challenge/iros/data/distance_plane_data.npy', allow_pickle=True)
__y3 = [(x[0].x**2 + x[0].y**2 + x[0].z**2)**0.5 for x in _y3]

# 数据处理（与你原代码一致）
for i in range(len(_y2)):
    if _y2[i] > 0.3:
        _y2[i] = _y2[i-1]
    
__y2 = _y2.copy()

for i in range(len(_y2)):
    if i-35 >= 0:
        __y2[i] = np.sum(_y2[i-35:i-15]) / 20

y1 = _y1[180:]
y2 = np.array(__y2)[180:]
y3 = __y3[180:]
y4 = _y4[180:]

n_sample = len(y1)
x1 = np.arange(len(y1))/n_sample*32
x2 = np.arange(len(y2))/n_sample*32
x3 = np.arange(len(y3))/n_sample*32
x4 = np.arange(len(y4))/n_sample*32

print(y2[-1])
print(len(y1), len(y2), len(y3), len(y4))

# 创建图形和子图
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# 初始化每个子图的线条对象
line1, = axs[0, 0].plot([], [], c='blue', alpha=0.5)
line2, = axs[0, 1].plot([], [], c='red', alpha=0.5)
line3, = axs[1, 0].plot([], [], c='green', alpha=0.5)
line4, = axs[1, 1].plot([], [], c='purple', alpha=0.5)

# 设置坐标轴标签
axs[0, 0].set_xlabel('t')
axs[0, 0].set_ylabel('distance_GP')
axs[0, 1].set_xlabel('t')
axs[0, 1].set_ylabel('ipa')
axs[1, 0].set_xlabel('t')
axs[1, 0].set_ylabel('velocity')
axs[1, 1].set_xlabel('t')
axs[1, 1].set_ylabel('distance_plane')

# 设置每个子图的范围（防止动画中坐标轴跳动）
axs[0, 0].set_xlim(0, len(x1)/n_sample*32)
axs[0, 0].set_ylim(np.min(y1), np.max(y1))
axs[0, 1].set_xlim(0, len(x2)/n_sample*32)
axs[0, 1].set_ylim(np.min(y2), np.max(y2))
axs[1, 0].set_xlim(0, len(x3)/n_sample*32)
axs[1, 0].set_ylim(np.min(y3), np.max(y3))
axs[1, 1].set_xlim(0, len(x4)/n_sample*32)
axs[1, 1].set_ylim(np.min(y4), np.max(y4))

# 初始化函数（动画开始前调用）
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    line4.set_data([], [])
    return line1, line2, line3, line4

# 更新函数（每一帧调用）
def update(frame):
    line1.set_data(x1[:frame], y1[:frame])
    line2.set_data(x2[:frame], y2[:frame])
    line3.set_data(x3[:frame], y3[:frame])
    line4.set_data(x4[:frame], y4[:frame])
    return line1, line2, line3, line4

# 创建动画
ani = FuncAnimation(fig, update, frames=len(x1), init_func=init, blit=True, interval=50)

# 调整布局
plt.tight_layout()

ani.save('features_fixed.mp4', writer='ffmpeg', fps=19)

# 显示动画
plt.show()

