#!/usr/bin/env python3
"""
分析训练过程中的奖励变化
记录每个checkpoint对应的奖励值
"""

import os

# import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


def read_tensorboard_data(log_dir):
    """读取TensorBoard数据"""
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )

        # 创建事件累积器
        ea = EventAccumulator(log_dir)
        ea.Reload()

        # 获取所有标量标签
        scalar_tags = ea.Tags()["scalars"]
        print(f"找到 {len(scalar_tags)} 个标量指标")

        # 提取奖励相关数据
        reward_data = {}

        for tag in scalar_tags:
            if "rew_" in tag or "reward" in tag.lower():
                scalar_events = ea.Scalars(tag)
                steps = [event.step for event in scalar_events]
                values = [event.value for event in scalar_events]
                reward_data[tag] = {"steps": steps, "values": values}
                print(f"  {tag}: {len(steps)} 个数据点")

        return reward_data

    except ImportError:
        print("TensorBoard未安装，尝试其他方法...")
        return None


def analyze_checkpoints(log_dir):
    """分析checkpoint对应的奖励"""

    # 1. 读取TensorBoard数据
    reward_data = read_tensorboard_data(log_dir)

    if reward_data is None:
        print("无法读取TensorBoard数据")
        return

    # 2. 查找模型文件
    model_files = []
    for file in os.listdir(log_dir):
        if file.startswith("model_") and file.endswith(".pt"):
            iteration = int(file.replace("model_", "").replace(".pt", ""))
            model_files.append(iteration)

    model_files.sort()
    print(f"\n找到 {len(model_files)} 个checkpoint:")
    print(f"迭代次数: {model_files}")

    # 3. 创建checkpoint奖励记录
    checkpoint_rewards = []

    # 获取总奖励数据（如果存在）
    total_reward_key = None
    for key in reward_data.keys():
        if (
            "Mean reward" in key
            or "total_reward" in key
            or key == "Episode/mean_reward"
        ):
            total_reward_key = key
            break

    if total_reward_key:
        steps = reward_data[total_reward_key]["steps"]
        values = reward_data[total_reward_key]["values"]

        for checkpoint in model_files:
            # 找到最接近checkpoint的数据点
            closest_idx = min(
                range(len(steps)), key=lambda i: abs(steps[i] - checkpoint)
            )

            checkpoint_rewards.append(
                {
                    "checkpoint": checkpoint,
                    "iteration": steps[closest_idx],
                    "mean_reward": values[closest_idx],
                }
            )

    # 4. 输出结果
    print(f"\n=== Checkpoint 奖励记录 ===")
    print(f"{'Checkpoint':<12} {'Iteration':<12} {'Mean Reward':<15}")
    print("-" * 40)

    for record in checkpoint_rewards:
        print(
            f"{record['checkpoint']:<12} {record['iteration']:<12} {record['mean_reward']:<15.4f}"
        )

    # 5. 保存到CSV文件
    if checkpoint_rewards:
        df = pd.DataFrame(checkpoint_rewards)
        csv_file = os.path.join(log_dir, "checkpoint_rewards.csv")
        df.to_csv(csv_file, index=False)
        print(f"\n结果已保存到: {csv_file}")

    # 6. 生成详细的奖励分解报告
    print(f"\n=== 最新checkpoint ({model_files[-1]}) 奖励分解 ===")

    latest_checkpoint = model_files[-1]
    for key, data in reward_data.items():
        if "rew_" in key:
            steps = data["steps"]
            values = data["values"]

            # 找到最接近最新checkpoint的值
            closest_idx = min(
                range(len(steps)), key=lambda i: abs(steps[i] - latest_checkpoint)
            )

            reward_name = key.replace("Train/", "").replace("Episode/", "")
            print(f"  {reward_name:<25}: {values[closest_idx]:<10.4f}")

    return checkpoint_rewards


def plot_reward_evolution(log_dir):
    """绘制奖励演化图"""
    reward_data = read_tensorboard_data(log_dir)

    if not reward_data:
        return

    plt.figure(figsize=(15, 10))

    # 绘制总奖励
    total_reward_key = None
    for key in reward_data.keys():
        if "Mean reward" in key or "total_reward" in key:
            total_reward_key = key
            break

    if total_reward_key:
        steps = reward_data[total_reward_key]["steps"]
        values = reward_data[total_reward_key]["values"]

        plt.subplot(2, 1, 1)
        plt.plot(steps, values, "b-", linewidth=2, label="Total Reward")
        plt.xlabel("Training Iteration")
        plt.ylabel("Mean Reward")
        plt.title("Training Progress: Total Reward")
        plt.grid(True, alpha=0.3)
        plt.legend()

    # 绘制主要奖励分量
    plt.subplot(2, 1, 2)
    important_rewards = [
        "rew_tracking_lin_vel",
        "rew_alive",
        "rew_contact",
        "rew_base_height",
        "rew_orientation",
    ]

    colors = ["red", "green", "blue", "orange", "purple"]

    for i, reward_name in enumerate(important_rewards):
        full_key = None
        for key in reward_data.keys():
            if reward_name in key:
                full_key = key
                break

        if full_key:
            steps = reward_data[full_key]["steps"]
            values = reward_data[full_key]["values"]
            plt.plot(
                steps,
                values,
                color=colors[i % len(colors)],
                label=reward_name.replace("rew_", ""),
                alpha=0.8,
            )

    plt.xlabel("Training Iteration")
    plt.ylabel("Reward Value")
    plt.title("Training Progress: Reward Components")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plot_file = os.path.join(log_dir, "reward_evolution.png")
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"奖励演化图已保存到: {plot_file}")


if __name__ == "__main__":
    # 分析指定的日志目录
    log_dir = "/home/yuxin/unitree_rl_gym/logs/h1_2_fullbody_obs/Jul16_20-35-55_"

    print(f"分析日志目录: {log_dir}")
    print(f"日志创建时间: {datetime.fromtimestamp(1752723356)}")

    # 分析checkpoint奖励
    checkpoint_rewards = analyze_checkpoints(log_dir)

    # 绘制奖励演化图
    try:
        plot_reward_evolution(log_dir)
    except Exception as e:
        print(f"绘图失败: {e}")
