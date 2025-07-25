import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 加载 gazepoint 数据
_gazepoint = np.load('./Project/data/gazepoint_data.npy', allow_pickle=True)

# 提取坐标和时间戳
coordinates = np.array([point[:2] for point in _gazepoint])  # 提取二维坐标
timestamps = np.array([point[2] for point in _gazepoint])    # 提取时间戳

# 计算帧间隔（以毫秒为单位）
time_deltas = np.diff(timestamps, prepend=timestamps[0]) * 1000  # 转换为毫秒

# 创建画布
fig, ax = plt.subplots()
ax.set_xlim(0, 640)  # 设置 x 轴范围
ax.set_ylim(0, 360)  # 设置 y 轴范围
ax.set_title("Gaze Point Trajectory")
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")

# 初始化轨迹线和当前点
trajectory, = ax.plot([], [], 'b-', lw=2, label="Trajectory")  # 蓝色轨迹线
current_point, = ax.plot([], [], 'ro', label="Current Point")  # 红色当前点
ax.legend()

# 初始化函数
def init():
    trajectory.set_data([], [])
    current_point.set_data([], [])
    return trajectory, current_point

# 更新函数
def update(frame):
    # 更新轨迹和当前点
    x_data = coordinates[:frame+1, 0]
    y_data = coordinates[:frame+1, 1]
    trajectory.set_data(x_data, y_data)
    current_point.set_data(coordinates[frame, 0], coordinates[frame, 1])
    
    # 如果是最后一帧，停止动画
    if frame == len(coordinates) - 1:
        ani.event_source.stop()
    
    return trajectory, current_point

# 自定义帧生成器
def frame_generator():
    for i in range(len(coordinates)):
        yield i

# 创建动画
ani = FuncAnimation(fig, update, frames=frame_generator(), init_func=init, blit=True, interval=1)

# 动态调整帧间隔
def update_interval():
    current_frame = ani.frame_seq.__next__()
    ani.event_source.interval = time_deltas[current_frame]
    ani._draw_next_frame(current_frame, blit=True)

ani.event_source.add_callback(update_interval)

# 显示动画
plt.show()