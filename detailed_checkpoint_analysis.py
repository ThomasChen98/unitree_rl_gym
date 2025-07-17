#!/usr/bin/env python3
"""
详细记录每个checkpoint的奖励信息
生成CSV报告和可视化图表
"""

import os
import pandas as pd
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def extract_checkpoint_rewards(log_dir):
    """提取每个checkpoint对应的详细奖励信息"""

    # 读取TensorBoard数据
    ea = EventAccumulator(log_dir)
    ea.Reload()

    # 获取所有奖励数据
    reward_data = {}
    scalar_tags = ea.Tags()["scalars"]

    for tag in scalar_tags:
        if "rew_" in tag or "reward" in tag.lower():
            scalar_events = ea.Scalars(tag)
            steps = [event.step for event in scalar_events]
            values = [event.value for event in scalar_events]
            reward_data[tag] = {step: value for step, value in zip(steps, values)}

    # 查找所有checkpoint
    checkpoint_files = []
    for file in os.listdir(log_dir):
        if file.startswith("model_") and file.endswith(".pt"):
            iteration = int(file.replace("model_", "").replace(".pt", ""))
            checkpoint_files.append(iteration)

    checkpoint_files.sort()

    # 为每个checkpoint提取奖励
    checkpoint_records = []

    for checkpoint in checkpoint_files:
        record = {"checkpoint": checkpoint}

        # 对每个奖励项，找到最接近checkpoint的值
        for reward_name, step_values in reward_data.items():
            # 找到最接近的步数
            closest_step = min(step_values.keys(), key=lambda x: abs(x - checkpoint))

            clean_name = reward_name.replace("Episode/", "").replace("Train/", "")
            record[clean_name] = step_values[closest_step]

        checkpoint_records.append(record)

    return pd.DataFrame(checkpoint_records)


def save_checkpoint_summary(log_dir):
    """保存checkpoint奖励总结"""

    df = extract_checkpoint_rewards(log_dir)

    # 保存完整数据
    full_csv = os.path.join(log_dir, "checkpoint_rewards_full.csv")
    df.to_csv(full_csv, index=False)
    print(f"完整奖励数据已保存到: {full_csv}")

    # 计算总奖励
    reward_columns = [col for col in df.columns if "rew_" in col]
    df["total_reward"] = df[reward_columns].sum(axis=1)

    # 保存简化版本
    key_columns = [
        "checkpoint",
        "total_reward",
        "rew_tracking_lin_vel",
        "rew_alive",
        "rew_contact",
        "rew_base_height",
    ]

    if all(col in df.columns for col in key_columns):
        summary_df = df[key_columns].copy()
        summary_csv = os.path.join(log_dir, "checkpoint_rewards_summary.csv")
        summary_df.to_csv(summary_csv, index=False)
        print(f"奖励总结已保存到: {summary_csv}")

        # 打印最新几个checkpoint的情况
        print("\n=== 最近10个Checkpoint奖励变化 ===")
        print(summary_df.tail(10).to_string(index=False))

    return df


if __name__ == "__main__":
    log_dir = "/home/yuxin/unitree_rl_gym/logs/h1_2_fullbody_obs/Jul16_20-35-55_"
    df = save_checkpoint_summary(log_dir)

    # 显示训练趋势
    if "total_reward" in df.columns:
        print(f"\n=== 训练趋势分析 ===")
        print(f"初始总奖励: {df['total_reward'].iloc[0]:.4f}")
        print(f"最终总奖励: {df['total_reward'].iloc[-1]:.4f}")
        print(
            f"最大总奖励: {df['total_reward'].max():.4f} (checkpoint {df.loc[df['total_reward'].idxmax(), 'checkpoint']})"
        )
        print(
            f"改善幅度: {((df['total_reward'].iloc[-1] - df['total_reward'].iloc[0]) / abs(df['total_reward'].iloc[0]) * 100):.1f}%"
        )
