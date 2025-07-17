#!/usr/bin/env python3
"""
PD参数影响可视化脚本
展示不同Kp和Kd值对系统响应的影响
"""

import numpy as np
import matplotlib.pyplot as plt


def simulate_pd_response(kp, kd, target_angle=1.0, duration=5.0, dt=0.01):
    """
    模拟PD控制器的阶跃响应

    Args:
        kp: 位置增益
        kd: 速度增益
        target_angle: 目标角度
        duration: 仿真时长
        dt: 时间步长

    Returns:
        time, angle, velocity, torque
    """
    time_steps = int(duration / dt)
    time = np.linspace(0, duration, time_steps)

    # 初始状态
    angle = 0.0
    velocity = 0.0

    # 存储结果
    angles = []
    velocities = []
    torques = []

    # 简化的关节动力学：J*ddθ + B*dθ = τ
    J = 1.0  # 惯性
    B = 0.1  # 摩擦阻尼

    for t in time:
        # PD控制律
        position_error = target_angle - angle
        velocity_error = 0 - velocity  # 目标速度为0
        torque = kp * position_error + kd * velocity_error

        # 简化动力学更新
        acceleration = (torque - B * velocity) / J
        velocity += acceleration * dt
        angle += velocity * dt

        # 存储数据
        angles.append(angle)
        velocities.append(velocity)
        torques.append(torque)

    return time, np.array(angles), np.array(velocities), np.array(torques)


def plot_pd_comparison():
    """绘制不同PD参数的响应对比"""

    # 测试参数组合
    test_cases = [
        # (kp, kd, label, color)
        (50, 2.0, "原始参数 (Kp=50, Kd=2.0)", "blue"),
        (120, 2.0, "仅提高Kp (Kp=120, Kd=2.0)", "red"),
        (50, 5.0, "仅提高Kd (Kp=50, Kd=5.0)", "green"),
        (120, 5.0, "优化参数 (Kp=120, Kd=5.0)", "purple"),
    ]

    plt.figure(figsize=(15, 12))

    # 子图1：角度响应
    plt.subplot(3, 2, 1)
    for kp, kd, label, color in test_cases:
        time, angles, _, _ = simulate_pd_response(kp, kd)
        plt.plot(time, angles, color=color, linewidth=2, label=label)

    plt.axhline(y=1.0, color="black", linestyle="--", alpha=0.5, label="目标位置")
    plt.xlabel("时间 (秒)")
    plt.ylabel("角度 (弧度)")
    plt.title("阶跃响应 - 角度变化")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图2：速度响应
    plt.subplot(3, 2, 2)
    for kp, kd, label, color in test_cases:
        time, _, velocities, _ = simulate_pd_response(kp, kd)
        plt.plot(time, velocities, color=color, linewidth=2, label=label)

    plt.xlabel("时间 (秒)")
    plt.ylabel("角速度 (弧度/秒)")
    plt.title("阶跃响应 - 角速度变化")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图3：控制扭矩
    plt.subplot(3, 2, 3)
    for kp, kd, label, color in test_cases:
        time, _, _, torques = simulate_pd_response(kp, kd)
        plt.plot(time, torques, color=color, linewidth=2, label=label)

    plt.xlabel("时间 (秒)")
    plt.ylabel("控制扭矩 (N⋅m)")
    plt.title("阶跃响应 - 控制扭矩")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图4：Kp影响分析
    plt.subplot(3, 2, 4)
    kp_values = [20, 50, 100, 150, 200]
    kd_fixed = 2.0

    for kp in kp_values:
        time, angles, _, _ = simulate_pd_response(kp, kd_fixed, duration=3.0)
        plt.plot(time, angles, linewidth=2, label=f"Kp={kp}")

    plt.axhline(y=1.0, color="black", linestyle="--", alpha=0.5)
    plt.xlabel("时间 (秒)")
    plt.ylabel("角度 (弧度)")
    plt.title(f"Kp影响分析 (Kd固定={kd_fixed})")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图5：Kd影响分析
    plt.subplot(3, 2, 5)
    kd_values = [0.5, 1.0, 2.0, 5.0, 8.0]
    kp_fixed = 100

    for kd in kd_values:
        time, angles, _, _ = simulate_pd_response(kp_fixed, kd, duration=3.0)
        plt.plot(time, angles, linewidth=2, label=f"Kd={kd}")

    plt.axhline(y=1.0, color="black", linestyle="--", alpha=0.5)
    plt.xlabel("时间 (秒)")
    plt.ylabel("角度 (弧度)")
    plt.title(f"Kd影响分析 (Kp固定={kp_fixed})")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图6：性能指标对比
    plt.subplot(3, 2, 6)

    # 计算性能指标
    metrics = []
    labels = []

    for kp, kd, label, color in test_cases:
        time, angles, velocities, _ = simulate_pd_response(kp, kd, duration=10.0)

        # 计算超调量
        overshoot = max(angles) - 1.0

        # 计算稳定时间（2%误差带）
        steady_state_error = 0.02
        steady_indices = np.where(np.abs(angles - 1.0) < steady_state_error)[0]
        settling_time = time[steady_indices[0]] if len(steady_indices) > 0 else 10.0

        # 计算稳态误差
        final_error = abs(angles[-1] - 1.0)

        metrics.append([overshoot, settling_time, final_error])
        labels.append(f'{label.split("(")[0]}')

    metrics = np.array(metrics)

    x = np.arange(len(labels))
    width = 0.25

    plt.bar(x - width, metrics[:, 0], width, label="超调量", alpha=0.7)
    plt.bar(x, metrics[:, 1], width, label="稳定时间(s)", alpha=0.7)
    plt.bar(x + width, metrics[:, 2] * 100, width, label="稳态误差×100", alpha=0.7)

    plt.xlabel("参数配置")
    plt.ylabel("性能指标")
    plt.title("PD参数性能对比")
    plt.xticks(x, labels, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "/home/yuxin/unitree_rl_gym/deploy/deploy_mujoco/pd_parameters_analysis.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()


def analyze_boxing_parameters():
    """分析拳击动作中的参数效果"""
    print("=== PD参数在拳击动作中的影响分析 ===")
    print()

    # 原始参数
    kp_orig, kd_orig = 50, 2.0
    time_orig, angles_orig, velocities_orig, torques_orig = simulate_pd_response(
        kp_orig, kd_orig
    )

    # 优化参数
    kp_opt, kd_opt = 120, 5.0
    time_opt, angles_opt, velocities_opt, torques_opt = simulate_pd_response(
        kp_opt, kd_opt
    )

    # 性能对比
    print("1. 响应速度对比：")
    # 达到90%目标值的时间
    t90_orig = time_orig[np.where(angles_orig >= 0.9)[0][0]]
    t90_opt = time_opt[np.where(angles_opt >= 0.9)[0][0]]
    print(f"   原始参数达到90%目标用时: {t90_orig:.3f}s")
    print(f"   优化参数达到90%目标用时: {t90_opt:.3f}s")
    print(f"   速度提升: {(t90_orig - t90_opt) / t90_orig * 100:.1f}%")
    print()

    print("2. 超调和振荡分析：")
    overshoot_orig = max(angles_orig) - 1.0
    overshoot_opt = max(angles_opt) - 1.0
    print(f"   原始参数超调量: {overshoot_orig:.3f} 弧度")
    print(f"   优化参数超调量: {overshoot_opt:.3f} 弧度")
    print()

    print("3. 稳定性分析：")
    # 计算稳定后的振荡程度（方差）
    stable_start = int(len(angles_orig) * 0.8)  # 后20%时间段
    var_orig = np.var(angles_orig[stable_start:])
    var_opt = np.var(angles_opt[stable_start:])
    print(f"   原始参数稳态振荡方差: {var_orig:.6f}")
    print(f"   优化参数稳态振荡方差: {var_opt:.6f}")
    print(f"   稳定性提升: {(var_orig - var_opt) / var_orig * 100:.1f}%")
    print()

    print("4. 能耗分析：")
    energy_orig = np.sum(np.abs(torques_orig) * np.abs(velocities_orig)) * 0.01
    energy_opt = np.sum(np.abs(torques_opt) * np.abs(velocities_opt)) * 0.01
    print(f"   原始参数总能耗: {energy_orig:.3f} J")
    print(f"   优化参数总能耗: {energy_opt:.3f} J")
    print(f"   能耗变化: {(energy_opt - energy_orig) / energy_orig * 100:.1f}%")
    print()

    print("5. 拳击动作的具体改善：")
    print("   ✅ 出拳响应更快，减少延迟")
    print("   ✅ 到达目标位置后振荡减少")
    print("   ✅ 更稳定地保持出拳姿态")
    print("   ✅ 回收动作更平滑")
    print("   ⚠️  控制扭矩和能耗适度增加")


if __name__ == "__main__":
    print("开始PD参数分析...")

    # 绘制对比图
    plot_pd_comparison()

    # 分析拳击参数效果
    analyze_boxing_parameters()

    print("\n分析完成！图表已保存到 pd_parameters_analysis.png")
