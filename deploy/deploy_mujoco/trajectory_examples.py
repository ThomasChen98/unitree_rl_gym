#!/usr/bin/env python3
"""
示例：不同上半身轨迹的实现
这个文件展示了如何修改 deploy_mujoco3.py 中的轨迹生成函数来实现不同的动作
"""

import math
import numpy as np


def trajectory_arm_circles(time_sim, config):
    """双臂圆周运动 - 原始实现"""
    traj_amp = config["trajectory_amplitude"]
    traj_freq = config["trajectory_frequency"]
    torso_amp = config["torso_bend_amplitude"]
    torso_freq = config["torso_bend_frequency"]
    arm_offset = config["arm_forward_offset"]

    phase = 2 * math.pi * traj_freq * time_sim
    torso_phase = 2 * math.pi * torso_freq * time_sim

    upper_body_targets = np.zeros(15)

    # 躯干前弯
    upper_body_targets[0] = torso_amp * math.sin(torso_phase)

    # 左臂圆周运动
    upper_body_targets[1] = arm_offset + traj_amp * math.sin(phase)
    upper_body_targets[2] = 0.3 * math.sin(phase)
    upper_body_targets[3] = 0.0
    upper_body_targets[4] = 1.2 + 0.3 * math.cos(phase)
    upper_body_targets[5] = 0.0
    upper_body_targets[6] = 0.0
    upper_body_targets[7] = 0.0

    # 右臂对称运动
    upper_body_targets[8] = arm_offset + traj_amp * math.sin(phase)
    upper_body_targets[9] = -0.3 * math.sin(phase)
    upper_body_targets[10] = 0.0
    upper_body_targets[11] = 1.2 + 0.3 * math.cos(phase)
    upper_body_targets[12] = 0.0
    upper_body_targets[13] = 0.0
    upper_body_targets[14] = 0.0

    return upper_body_targets


def trajectory_waving_hello(time_sim, config):
    """挥手打招呼"""
    wave_freq = 2.0  # 2Hz 挥手频率
    wave_phase = 2 * math.pi * wave_freq * time_sim

    upper_body_targets = np.zeros(15)

    # 躯干保持直立
    upper_body_targets[0] = 0.0

    # 左臂自然下垂
    upper_body_targets[1] = 0.0
    upper_body_targets[2] = 0.0
    upper_body_targets[3] = 0.0
    upper_body_targets[4] = 0.0
    upper_body_targets[5] = 0.0
    upper_body_targets[6] = 0.0
    upper_body_targets[7] = 0.0

    # 右臂挥手动作
    upper_body_targets[8] = 1.2  # 肩膀抬起
    upper_body_targets[9] = -0.8  # 肩膀外展
    upper_body_targets[10] = 0.0
    upper_body_targets[11] = 1.0  # 肘部弯曲
    upper_body_targets[12] = 0.0
    upper_body_targets[13] = 0.3 * math.sin(wave_phase)  # 手腕左右摆动
    upper_body_targets[14] = 0.0

    return upper_body_targets


def trajectory_tai_chi(time_sim, config):
    """太极推手动作"""
    slow_freq = 0.15  # 很慢的频率
    phase = 2 * math.pi * slow_freq * time_sim

    upper_body_targets = np.zeros(15)

    # 躯干轻微转动
    upper_body_targets[0] = 0.1 * math.sin(phase * 0.5)

    # 左右臂交替推拉
    push_amplitude = 0.6

    # 左臂
    upper_body_targets[1] = 0.8 + push_amplitude * math.sin(phase)
    upper_body_targets[2] = 0.3
    upper_body_targets[3] = 0.2 * math.sin(phase)
    upper_body_targets[4] = 1.0 - 0.3 * math.sin(phase)
    upper_body_targets[5] = 0.0
    upper_body_targets[6] = 0.0
    upper_body_targets[7] = 0.0

    # 右臂（相位相反）
    upper_body_targets[8] = 0.8 + push_amplitude * math.sin(phase + math.pi)
    upper_body_targets[9] = -0.3
    upper_body_targets[10] = -0.2 * math.sin(phase + math.pi)
    upper_body_targets[11] = 1.0 - 0.3 * math.sin(phase + math.pi)
    upper_body_targets[12] = 0.0
    upper_body_targets[13] = 0.0
    upper_body_targets[14] = 0.0

    return upper_body_targets


def trajectory_boxing(time_sim, config):
    """拳击动作"""
    punch_freq = 1.5  # 1.5Hz 出拳频率
    phase = 2 * math.pi * punch_freq * time_sim

    upper_body_targets = np.zeros(15)

    # 躯干稍微前倾
    upper_body_targets[0] = 0.1

    # 拳击姿势：交替出拳
    if math.sin(phase) > 0:
        # 左拳出击
        upper_body_targets[1] = 0.5 + 0.4 * math.sin(phase)  # 快速前伸
        upper_body_targets[2] = 0.2
        upper_body_targets[3] = 0.0
        upper_body_targets[4] = 1.5 - 0.5 * math.sin(phase)  # 肘部伸展
        upper_body_targets[5] = 0.0
        upper_body_targets[6] = 0.0
        upper_body_targets[7] = 0.0

        # 右拳防守
        upper_body_targets[8] = 0.3
        upper_body_targets[9] = -0.2
        upper_body_targets[10] = 0.0
        upper_body_targets[11] = 1.3
        upper_body_targets[12] = 0.0
        upper_body_targets[13] = 0.0
        upper_body_targets[14] = 0.0
    else:
        # 右拳出击
        upper_body_targets[1] = 0.3
        upper_body_targets[2] = 0.2
        upper_body_targets[3] = 0.0
        upper_body_targets[4] = 1.3
        upper_body_targets[5] = 0.0
        upper_body_targets[6] = 0.0
        upper_body_targets[7] = 0.0

        # 左拳防守
        upper_body_targets[8] = 0.5 - 0.4 * math.sin(phase)  # 快速前伸
        upper_body_targets[9] = -0.2
        upper_body_targets[10] = 0.0
        upper_body_targets[11] = 1.5 + 0.5 * math.sin(phase)  # 肘部伸展
        upper_body_targets[12] = 0.0
        upper_body_targets[13] = 0.0
        upper_body_targets[14] = 0.0

    return upper_body_targets


def trajectory_dancing(time_sim, config):
    """舞蹈动作"""
    dance_freq = 0.8  # 0.8Hz 舞蹈频率
    phase = 2 * math.pi * dance_freq * time_sim

    upper_body_targets = np.zeros(15)

    # 躯干左右摆动
    upper_body_targets[0] = 0.2 * math.sin(phase * 0.5)

    # 双臂协调摆动
    arm_swing = 0.8

    # 左臂
    upper_body_targets[1] = 0.5 + arm_swing * math.sin(phase)
    upper_body_targets[2] = 0.5 + 0.3 * math.cos(phase)
    upper_body_targets[3] = 0.3 * math.sin(phase * 2)
    upper_body_targets[4] = 0.5 + 0.5 * math.sin(phase + math.pi / 4)
    upper_body_targets[5] = 0.0
    upper_body_targets[6] = 0.2 * math.sin(phase * 3)
    upper_body_targets[7] = 0.0

    # 右臂（稍有不同的相位）
    upper_body_targets[8] = 0.5 + arm_swing * math.sin(phase + math.pi / 3)
    upper_body_targets[9] = -0.5 - 0.3 * math.cos(phase + math.pi / 3)
    upper_body_targets[10] = -0.3 * math.sin(phase * 2 + math.pi / 3)
    upper_body_targets[11] = 0.5 + 0.5 * math.sin(phase + math.pi / 4 + math.pi / 3)
    upper_body_targets[12] = 0.0
    upper_body_targets[13] = -0.2 * math.sin(phase * 3 + math.pi / 3)
    upper_body_targets[14] = 0.0

    return upper_body_targets


def trajectory_stretching(time_sim, config):
    """拉伸动作"""
    stretch_freq = 0.1  # 很慢的拉伸频率
    phase = 2 * math.pi * stretch_freq * time_sim

    upper_body_targets = np.zeros(15)

    # 根据时间切换不同的拉伸动作
    cycle_time = time_sim % 20.0  # 20秒一个循环

    if cycle_time < 5.0:
        # 双臂向上拉伸
        stretch_amount = 0.5 * (1 + math.sin(phase))
        upper_body_targets[0] = -0.1  # 躯干稍微后仰
        upper_body_targets[1] = -1.0 + stretch_amount * 2.0
        upper_body_targets[8] = -1.0 + stretch_amount * 2.0
        upper_body_targets[2] = stretch_amount * 0.3
        upper_body_targets[9] = -stretch_amount * 0.3

    elif cycle_time < 10.0:
        # 左右侧弯
        side_bend = 0.3 * math.sin(phase * 2)
        upper_body_targets[0] = side_bend
        upper_body_targets[1] = 1.2
        upper_body_targets[8] = 1.2
        upper_body_targets[2] = 0.5 + side_bend
        upper_body_targets[9] = -0.5 - side_bend

    elif cycle_time < 15.0:
        # 前后拉伸
        forward_back = 0.4 * math.sin(phase)
        upper_body_targets[0] = forward_back
        upper_body_targets[1] = 0.8 + forward_back
        upper_body_targets[8] = 0.8 + forward_back

    else:
        # 放松姿势
        upper_body_targets[0] = 0.0
        upper_body_targets[1] = 0.2
        upper_body_targets[8] = 0.2
        upper_body_targets[4] = 0.3
        upper_body_targets[11] = 0.3

    return upper_body_targets


# 使用示例：
# 在 deploy_mujoco3.py 中，替换 generate_upper_body_trajectory 函数的内容：

"""
def generate_upper_body_trajectory(time_sim, config):
    # 选择你想要的轨迹类型
    
    # 1. 原始圆周运动
    # return trajectory_arm_circles(time_sim, config)
    
    # 2. 挥手打招呼
    # return trajectory_waving_hello(time_sim, config)
    
    # 3. 太极推手
    # return trajectory_tai_chi(time_sim, config)
    
    # 4. 拳击动作
    # return trajectory_boxing(time_sim, config)
    
    # 5. 舞蹈动作
    # return trajectory_dancing(time_sim, config)
    
    # 6. 拉伸动作
    return trajectory_stretching(time_sim, config)
"""

if __name__ == "__main__":
    print("这是一个示例文件，展示了不同的上半身轨迹实现")
    print("请将想要的轨迹函数复制到 deploy_mujoco3.py 中使用")
