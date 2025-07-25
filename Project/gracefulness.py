import numpy as np
import os

def calculate_gracefulness(data):
    # 计算λ(t)，即位置向量
    positions = data[:, :3]
    
    # 计算速度向量 (一阶导数)
    velocities = np.gradient(positions, axis=0)
    
    # 计算加速度向量 (二阶导数)
    accelerations = np.gradient(velocities, axis=0)
    
    # 计算曲率 κ
    # κ = ||λ'(t) × λ''(t)|| / ||λ'(t)||^3
    cross_products = np.cross(velocities, accelerations)
    numerator = np.linalg.norm(cross_products, axis=1)
    denominator = np.power(np.linalg.norm(velocities, axis=1), 3)
    
    # 避免除以0
    denominator = np.where(denominator == 0, np.inf, denominator)
    curvature = numerator / denominator
    
    # 计算G值 (使用log10的中位数)
    G = np.median(np.log10(curvature + 1e-10))  # 添加小量避免log(0)
    
    return G

def calculate_smoothness(data):
    # 计算位置向量的三阶导数
    positions = data[:, :3]
    first_deriv = np.gradient(positions, axis=0)
    second_deriv = np.gradient(first_deriv, axis=0)
    third_deriv = np.gradient(second_deriv, axis=0)
    
    # 计算三阶导数的平方范数
    jerk_squared = np.sum(np.square(third_deriv), axis=1)
    
    # 计算时间间隔
    time = data[:, 3]
    dt = time[1] - time[0]
    
    # 计算积分（使用梯形法则）
    integral = np.trapz(jerk_squared, dx=dt)
    
    # 根据公式中的参数计算
    duration = time[-1] - time[0]
    peak_velocity = np.max(np.linalg.norm(first_deriv, axis=1))
    
    # 计算 φ
    phi = (np.power(duration, 5) / np.square(peak_velocity)) * integral
    
    # 计算S值
    S = np.median(np.log10(phi + 1e-10))  # 添加小量避免log(0)
    
    return S

def calculate_metrics(data):
    G = calculate_gracefulness(data)
    S = calculate_smoothness(data)
    return G, S

def cal_GS():
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    data = np.load(os.path.join(current_dir, 'data', 'psm_ghost_pose.npy'), allow_pickle=True)
    #clutch = np.load(os.path.join(current_dir, 'data', 'clutch_times.npy'), allow_pickle=True)
    G, S = calculate_metrics(data)

    return G, S

print(cal_GS())