import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# 1. 定义我们想要学习的“真实函数”
def true_function(X):
    return (2 - X.ravel()) * np.sin(1.5 * X.ravel()) * 3 - (X.ravel() - 1)**2 + 1

# 2. 定义绘图的X轴范围和真实函数的值
X_plot = np.linspace(-2, 7, 500).reshape(-1, 1)
y_true = true_function(X_plot)

# 3. 初始采样点和之后要添加的点
initial_point = np.array([[0.]])
points_to_add = [np.array([[6.0]]), np.array([[3.0]])]

# 初始化训练数据
X_train = initial_point
y_train = true_function(X_train)

# 4. 设置高斯过程的核函数
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))

# 5. 创建绘图窗口
fig, axes = plt.subplots(1, 3, figsize=(21, 6))
plot_titles = ["Initial Sample", "Added Second Sample", "Added Third Sample"]

# 6. 循环生成三个子图
for i in range(3):
    ax = axes[i]

    # 初始化并训练高斯过程模型
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1, random_state=42)
    gp.fit(X_train, y_train)

    # 在整个X轴范围内进行预测，获取均值和标准差
    y_mean, y_std = gp.predict(X_plot, return_std=True)

    # --- 开始绘图 ---
    ax.plot(X_plot, y_true, 'r--', linewidth=2, label='True Function')
    ax.plot(X_plot, y_mean, 'b-', linewidth=2, label='GP Mean')
    ax.fill_between(X_plot.ravel(), y_mean - 1.96 * y_std, y_mean + 1.96 * y_std,
                    alpha=0.3, color='green', label='Confidence Interval')
    ax.axhline(y=0, color='r', linestyle='-', alpha=0.5, linewidth=1)
    ax.plot(X_train, y_train, 'ko', markersize=8, label='Sample Points')

    if i < 2:
        next_sample_x = points_to_add[i]
        next_sample_y = true_function(next_sample_x)
        ax.plot(next_sample_x, next_sample_y, 'r*', markersize=15, label='Next Sample given')

    # --- 格式化图表 ---
    ax.set_title(plot_titles[i], fontsize=25)
    if i == 0:
        ax.set_ylabel("Final Scores", fontsize=25)
    ax.grid(False)
    ax.set_ylim(-50, 45)
    ax.set_xlim(-2.5, 7.5)
    
    # 为下一次循环准备数据
    if i < 2:
        X_train = np.vstack([X_train, points_to_add[i]])
        y_train = true_function(X_train)

# --- 创建共享图例 ---
# 从最后一个子图中获取图例句柄和标签
handles, labels = axes[0].get_legend_handles_labels()

# 在整个图的顶部中心创建共享图例，并设置更大的字体
fig.legend(handles, labels, loc='lower center', ncol=10, fontsize=25, frameon=False)

# 调整布局，为顶部的图例留出空间 (rect=[left, bottom, right, top])
plt.tight_layout(rect=[0, 0.1, 1, 1])

plt.show()