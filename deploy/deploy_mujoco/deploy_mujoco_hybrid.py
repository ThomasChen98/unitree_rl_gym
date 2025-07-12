import time
import numpy as np
import mujoco.viewer
import mujoco
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml
import math


def get_gravity_orientation(quaternion):
    """Extract gravity vector from quaternion orientation"""
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


def trajectory_2arms_circles(time_sim, config):
    """Dual arm simple forward-backward swinging"""
    traj_amp = 1.2  # Swing amplitude
    traj_freq = 0.8  # Swing frequency
    arm_offset = 0.0  # Center position

    # Calculate swing angle: center position ± amplitude
    swing_angle = traj_amp * math.sin(traj_freq * time_sim)

    upper_body_targets = np.zeros(15)

    # Torso stays upright
    upper_body_targets[0] = 0.0

    # Left arm: shoulder forward-backward swing, other joints fixed
    upper_body_targets[1] = arm_offset - swing_angle  # Shoulder swing
    upper_body_targets[2] = 0.3  # Shoulder side fixed angle
    upper_body_targets[3:8] = 0.0  # Other joints stay neutral

    # Right arm: synchronized swing with left arm
    upper_body_targets[8] = arm_offset - swing_angle  # 肩部前后摆动
    upper_body_targets[9] = -0.3  # 肩部侧向固定角度（镜像）
    upper_body_targets[10:15] = 0.0  # 其他关节保持中性

    return upper_body_targets


def trajectory_1arm_circles(time_sim, config):
    """Single arm circular motion"""
    traj_amp = 1.2  # Swing amplitude
    traj_freq = 0.8  # Swing frequency
    arm_offset = 0.0  # Center position

    # Calculate swing angle: center position ± amplitude
    swing_angle = traj_amp * math.sin(traj_freq * time_sim)

    upper_body_targets = np.zeros(15)

    # Torso stays upright
    upper_body_targets[0] = 0.0

    # Left arm: shoulder forward-backward swing, other joints fixed
    upper_body_targets[1] = arm_offset - swing_angle  # Shoulder swing
    upper_body_targets[2] = 0.3  # Shoulder side fixed angle
    upper_body_targets[3:8] = 0.0  # Other joints stay neutral

    # Right arm: stays still
    upper_body_targets[8] = 0.0  # Shoulder stays neutral
    upper_body_targets[9] = -0.3  # Shoulder side fixed angle (mirror)
    upper_body_targets[10:15] = 0.0  # Other joints stay neutral

    return upper_body_targets


def trajectory_2arms_waving(time_sim, config):
    """Dual arm waving motion"""
    wave_amp = 2.0  # Wave amplitude
    wave_freq = 0.5  # 2Hz wave frequency
    wave_offset = 0.3

    upper_body_targets = np.zeros(15)

    # Torso stays upright
    upper_body_targets[0] = 0.0

    upper_body_targets[1] = 0.0
    upper_body_targets[2] = wave_offset + wave_amp * abs(
        math.sin(wave_freq * time_sim)
    )  # Left shoulder swing
    upper_body_targets[3:8] = 0.0  # Left shoulder pitch

    # Right arm waving motion
    upper_body_targets[8] = 0.0  # Shoulder raise
    upper_body_targets[9] = -wave_offset - wave_amp * abs(
        math.sin(wave_freq * time_sim)
    )  # Right shoulder swing
    upper_body_targets[10] = 0.0
    upper_body_targets[11] = 0.0  # Elbow bend
    upper_body_targets[12] = 0.0
    upper_body_targets[13] = 0.0  # Wrist swing
    upper_body_targets[14] = 0.0

    return upper_body_targets


def trajectory_1arm_waving(time_sim, config):
    """Single arm waving motion"""
    wave_amp = 2.0  # Wave amplitude
    wave_freq = 0.5  # 2Hz wave frequency
    wave_offset = 0.0

    upper_body_targets = np.zeros(15)

    # Torso stays upright
    upper_body_targets[0] = 0.0

    upper_body_targets[1] = 0.0
    upper_body_targets[2] = wave_offset + wave_amp * abs(
        math.sin(wave_freq * time_sim)
    )  # Left shoulder swing
    upper_body_targets[3:8] = 0.0  # Left shoulder pitch

    # Right arm stays neutral
    upper_body_targets[8] = 0.0  # Shoulder neutral
    upper_body_targets[9] = -wave_offset
    upper_body_targets[10] = 0.0
    upper_body_targets[11] = 0.0  # Elbow neutral
    upper_body_targets[12] = 0.0
    upper_body_targets[13] = 0.0  # Wrist neutral
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
    upper_body_targets[1] = -0.8 - push_amplitude * math.sin(phase)
    upper_body_targets[2] = 0.3
    upper_body_targets[3] = 0.2 * math.sin(phase)
    upper_body_targets[4] = -1.0 + 0.5 * math.sin(phase)
    upper_body_targets[5:8] = 0.0

    # 右臂（相位相反）
    upper_body_targets[8] = -0.8 - push_amplitude * math.sin(phase + math.pi)
    upper_body_targets[9] = -0.3
    upper_body_targets[10] = -0.2 * math.sin(phase + math.pi)
    upper_body_targets[11] = -1.0 + 0.5 * math.sin(phase + math.pi)
    upper_body_targets[12:15] = 0.0

    return upper_body_targets


def trajectory_boxing(time_sim, config):
    """拳击动作 - 快速交替出拳，出拳后停顿，带平滑过渡"""
    punch_cycle_time = 4.0  # 每个出拳周期4秒
    punch_duration = 0.6  # 出拳持续时间0.6秒
    hold_duration = 0.0  # 出拳后停顿1.0秒
    transition_duration = 0.4  # 过渡时间0.4秒

    # 计算当前在周期中的位置
    cycle_pos = time_sim % punch_cycle_time
    half_cycle = punch_cycle_time / 2  # 2秒一次出拳切换

    upper_body_targets = np.zeros(15)

    # 躯干稍微前倾，保持拳击姿态
    upper_body_targets[0] = 0.15

    # 拳击姿势参数 - 减小角度，增加稳定性
    punch_extension = -1.8  # 出拳时胳膊前伸角度（负值）
    guard_position = 0.2  # 防守时胳膊位置
    elbow_bend_punch = -0.6  # 出拳时肘部角度
    elbow_bend_guard = -2.5  # 防守时肘部弯曲

    def smooth_transition(from_val, to_val, progress):
        """平滑插值函数，使用余弦插值"""
        smooth_progress = 0.5 * (1 - math.cos(math.pi * progress))
        return from_val + (to_val - from_val) * smooth_progress

    # 判断是左拳还是右拳的回合
    if cycle_pos < half_cycle:
        # 左拳回合
        local_time = cycle_pos

        if local_time < punch_duration:
            # 左拳出击阶段 - 平滑过渡到出拳位置
            progress = local_time / punch_duration
            progress = min(1.0, max(0.0, progress))

            # 左拳平滑出击
            left_shoulder = smooth_transition(guard_position, punch_extension, progress)
            left_elbow = smooth_transition(elbow_bend_guard, elbow_bend_punch, progress)

            upper_body_targets[1] = left_shoulder  # 左肩前伸
            upper_body_targets[2] = 0.1
            upper_body_targets[3] = 0.0
            upper_body_targets[4] = left_elbow  # 左肘伸展出拳
            upper_body_targets[5:8] = 0.0

            # 右拳防守姿态
            upper_body_targets[8] = guard_position  # 右肩防守位置
            upper_body_targets[9] = -0.1
            upper_body_targets[10] = 0.0
            upper_body_targets[11] = elbow_bend_guard  # 右肘弯曲防守
            upper_body_targets[12:15] = 0.0

        elif local_time < (punch_duration + hold_duration):
            # 左拳保持阶段 - 增加更强的阻尼效果
            upper_body_targets[1] = punch_extension  # 左肩保持前伸
            upper_body_targets[2] = 0.1
            upper_body_targets[3] = 0.0
            upper_body_targets[4] = elbow_bend_punch  # 左肘保持伸展
            upper_body_targets[5:8] = 0.0

            # 右拳防守姿态
            upper_body_targets[8] = guard_position  # 右肩防守位置
            upper_body_targets[9] = -0.1
            upper_body_targets[10] = 0.0
            upper_body_targets[11] = elbow_bend_guard  # 右肘弯曲防守
            upper_body_targets[12:15] = 0.0
        else:
            # 左拳回收阶段 - 平滑过渡回防守位置
            transition_time = local_time - punch_duration - hold_duration
            progress = transition_time / transition_duration
            progress = min(1.0, max(0.0, progress))

            left_shoulder = smooth_transition(punch_extension, guard_position, progress)
            left_elbow = smooth_transition(elbow_bend_punch, elbow_bend_guard, progress)

            upper_body_targets[1] = left_shoulder
            upper_body_targets[2] = 0.1
            upper_body_targets[3] = 0.0
            upper_body_targets[4] = left_elbow
            upper_body_targets[5:8] = 0.0

            upper_body_targets[8] = guard_position
            upper_body_targets[9] = -0.1
            upper_body_targets[10] = 0.0
            upper_body_targets[11] = elbow_bend_guard
            upper_body_targets[12:15] = 0.0
    else:
        # 右拳回合
        local_time = cycle_pos - half_cycle

        if local_time < punch_duration:
            # 右拳出击阶段 - 平滑过渡到出拳位置
            progress = local_time / punch_duration
            progress = min(1.0, max(0.0, progress))

            # 左拳防守
            upper_body_targets[1] = guard_position  # 左肩防守位置
            upper_body_targets[2] = 0.1
            upper_body_targets[3] = 0.0
            upper_body_targets[4] = elbow_bend_guard  # 左肘弯曲防守
            upper_body_targets[5:8] = 0.0

            # 右拳平滑出击
            right_shoulder = smooth_transition(
                guard_position, punch_extension, progress
            )
            right_elbow = smooth_transition(
                elbow_bend_guard, elbow_bend_punch, progress
            )

            upper_body_targets[8] = right_shoulder  # 右肩前伸
            upper_body_targets[9] = -0.1  # 右肩抬起
            upper_body_targets[10] = 0.0
            upper_body_targets[11] = right_elbow  # 右肘伸展出拳
            upper_body_targets[12:15] = 0.0

        elif local_time < (punch_duration + hold_duration):
            # 右拳保持阶段
            upper_body_targets[1] = guard_position
            upper_body_targets[2] = 0.1
            upper_body_targets[3] = 0.0
            upper_body_targets[4] = elbow_bend_guard
            upper_body_targets[5:8] = 0.0

            upper_body_targets[8] = punch_extension  # 右肩保持前伸
            upper_body_targets[9] = -0.1
            upper_body_targets[10] = 0.0
            upper_body_targets[11] = elbow_bend_punch  # 右肘保持伸展
            upper_body_targets[12:15] = 0.0
        else:
            # 右拳回收阶段 - 平滑过渡回防守位置
            transition_time = local_time - punch_duration - hold_duration
            progress = transition_time / transition_duration
            progress = min(1.0, max(0.0, progress))

            upper_body_targets[1] = guard_position
            upper_body_targets[2] = 0.1
            upper_body_targets[3] = 0.0
            upper_body_targets[4] = elbow_bend_guard
            upper_body_targets[5:8] = 0.0

            right_shoulder = smooth_transition(
                punch_extension, guard_position, progress
            )
            right_elbow = smooth_transition(
                elbow_bend_punch, elbow_bend_guard, progress
            )

            upper_body_targets[8] = right_shoulder
            upper_body_targets[9] = -0.1
            upper_body_targets[10] = 0.0
            upper_body_targets[11] = right_elbow
            upper_body_targets[12:15] = 0.0

    return upper_body_targets


def trajectory_random_motion(time_sim, config):
    """随机动作 - 15个上身关节的随机角度变化"""
    import random

    # 设置随机种子，基于时间但变化较慢，避免过于频繁的变化
    random.seed(int(time_sim * 2) % 1000)  # 每0.5秒更新一次随机种子

    upper_body_targets = np.zeros(15)

    # 定义每个关节的随机范围和特性
    joint_ranges = [
        # 躯干关节 (索引0)
        [-0.5, 0.5],  # 躯干前后倾斜
        # 左臂关节 (索引1-7)
        [-2.0, 1.0],  # 左肩前后 (负值为前伸)
        [-0.5, 1.5],  # 左肩侧向 (正值为抬起)
        [-1.0, 1.0],  # 左肩转动
        [-2.0, 0.0],  # 左肘弯曲 (负值为弯曲)
        [-1.0, 1.0],  # 左腕转动1
        [-0.5, 0.5],  # 左腕转动2
        [-0.5, 0.5],  # 左腕转动3
        # 右臂关节 (索引8-14) - 镜像左臂
        [-2.0, 1.0],  # 右肩前后 (负值为前伸)
        [-1.5, 0.5],  # 右肩侧向 (负值为抬起)
        [-1.0, 1.0],  # 右肩转动
        [-2.0, 0.0],  # 右肘弯曲 (负值为弯曲)
        [-1.0, 1.0],  # 右腕转动1
        [-0.5, 0.5],  # 右腕转动2
        [-0.5, 0.5],  # 右腕转动3
    ]

    # 为每个关节生成随机角度
    for i in range(15):
        min_angle, max_angle = joint_ranges[i]
        # 生成随机角度，添加一些时间相关的平滑变化
        base_random = random.uniform(min_angle, max_angle)
        time_variation = 0.2 * math.sin(0.5 * time_sim + i)  # 每个关节不同相位

        upper_body_targets[i] = base_random + time_variation

        # 确保角度在安全范围内
        upper_body_targets[i] = max(min_angle, min(max_angle, upper_body_targets[i]))

    return upper_body_targets


# ========== 静止动作函数 ==========


def pose_arms_forward(time_sim, config):
    """静止动作1: 双臂前伸"""
    upper_body_targets = np.zeros(15)

    # 躯干保持直立
    upper_body_targets[0] = 0.0

    # 左臂前伸
    upper_body_targets[1] = -2.0  # 肩膀前伸
    upper_body_targets[2] = 0.5  # 肩膀内外旋
    upper_body_targets[3] = 0.0  # 肩膀俯仰
    upper_body_targets[4] = 0.0  # 肘部伸直
    upper_body_targets[5:8] = 0.0  # 手腕保持中性

    # 右臂前伸（镜像）
    upper_body_targets[8] = -2.0  # 肩膀前伸
    upper_body_targets[9] = -0.5  # 肩膀内外旋
    upper_body_targets[10] = 0.0  # 肩膀俯仰
    upper_body_targets[11] = 0.0  # 肘部伸直
    upper_body_targets[12:15] = 0.0  # 手腕保持中性

    return upper_body_targets


def pose_left_down_right_forward(time_sim, config):
    """静止动作2: 左臂下垂，右臂前伸"""
    upper_body_targets = np.zeros(15)

    # 躯干保持直立
    upper_body_targets[0] = 0.0

    # 左臂自然下垂
    upper_body_targets[1:8] = 0.0

    # 右臂前伸
    upper_body_targets[8] = -2.0  # 肩膀前伸
    upper_body_targets[9] = -0.5  # 肩膀内外旋
    upper_body_targets[10] = 0.0  # 肩膀俯仰
    upper_body_targets[11] = 0.0  # 肘部伸直
    upper_body_targets[12:15] = 0.0  # 手腕保持中性

    return upper_body_targets


def pose_t_shape(time_sim, config):
    """静止动作3: 双臂侧向张开，成十字架状"""
    upper_body_targets = np.zeros(15)

    # 躯干保持直立
    upper_body_targets[0] = 0.0

    # 左臂侧向张开
    upper_body_targets[1] = 0.0  # 肩膀不前伸
    upper_body_targets[2] = 1.57  # 肩膀外展90度
    upper_body_targets[3] = 0.0  # 肩膀俯仰
    upper_body_targets[4] = 0.0  # 肘部伸直
    upper_body_targets[5:8] = 0.0  # 手腕保持中性

    # 右臂侧向张开（镜像）
    upper_body_targets[8] = 0.0  # 肩膀不前伸
    upper_body_targets[9] = -1.57  # 肩膀外展90度
    upper_body_targets[10] = 0.0  # 肩膀俯仰
    upper_body_targets[11] = 0.0  # 肘部伸直
    upper_body_targets[12:15] = 0.0  # 手腕保持中性

    return upper_body_targets


def pose_left_down_right_side(time_sim, config):
    """静止动作4: 左臂下垂，右臂侧向张开"""
    upper_body_targets = np.zeros(15)

    # 躯干保持直立
    upper_body_targets[0] = 0.0

    # 左臂自然下垂
    upper_body_targets[1:8] = 0.0

    # 右臂侧向张开
    upper_body_targets[8] = 0.0  # 肩膀不前伸
    upper_body_targets[9] = -1.57  # 肩膀外展90度
    upper_body_targets[10] = 0.0  # 肩膀俯仰
    upper_body_targets[11] = 0.0  # 肘部伸直
    upper_body_targets[12:15] = 0.0  # 手腕保持中性

    return upper_body_targets


def pose_torso_side_twist(time_sim, config):
    """静止动作5: 躯干向前扭转（绕z轴）"""
    upper_body_targets = np.zeros(15)

    # 躯干向前扭转30度
    upper_body_targets[0] = 1.57  # 绕y轴前倾

    # 双臂自然下垂
    upper_body_targets[1:8] = 0.0
    upper_body_targets[8:15] = 0.0

    return upper_body_targets


def pose_arms_up(time_sim, config):
    """静止动作6: 双臂上举"""
    upper_body_targets = np.zeros(15)

    # 躯干保持直立
    upper_body_targets[0] = 0.0

    upper_body_targets[1] = -3.0  # 肩膀后伸（向上举）
    upper_body_targets[2] = 0.5  # 肩膀内外旋
    upper_body_targets[3] = 0.0  # 肩膀俯仰
    upper_body_targets[4] = 0.0  # 肘部伸直
    upper_body_targets[5:8] = 0.0  # 手腕保持中性

    upper_body_targets[8] = -3.0
    upper_body_targets[9] = -0.5
    upper_body_targets[10:15] = 0.0

    return upper_body_targets


def generate_upper_body_trajectory(time_sim, config, trajectory_type="circles"):
    """Generate predefined trajectories for upper body joints"""

    # Trajectory selection dictionary
    trajectory_functions = {
        # Dynamic trajectories - dual arm
        "2arms_circles": trajectory_2arms_circles,
        "2arms_waving": trajectory_2arms_waving,
        # Dynamic trajectories - single arm
        "1arm_circles": trajectory_1arm_circles,
        "1arm_waving": trajectory_1arm_waving,
        # Complex motion trajectories
        "taichi": trajectory_tai_chi,
        "boxing": trajectory_boxing,
        "random": trajectory_random_motion,
        # Static upper body poses
        "pose_arms_forward": pose_arms_forward,
        "pose_left_down_right_forward": pose_left_down_right_forward,
        "pose_t_shape": pose_t_shape,
        "pose_left_down_right_side": pose_left_down_right_side,
        "pose_torso_side_twist": pose_torso_side_twist,
        "pose_arms_up": pose_arms_up,
    }

    # Get corresponding trajectory function
    if trajectory_type in trajectory_functions:
        return trajectory_functions[trajectory_type](time_sim, config)
    else:
        print(
            f"Warning: Unknown trajectory type '{trajectory_type}', "
            f"using default '2arms_circles'"
        )
        return trajectory_2arms_circles(time_sim, config)


def extract_observations(d, default_angles, config, action, time_sim, cmd_values=None):
    """Extract observations for the policy"""
    # Joint positions and velocities (lower body only)
    qj = d.qpos[7:19]  # Lower body joints
    dqj = d.qvel[6:18]  # Lower body joint velocities
    quat = d.qpos[3:7]  # Base orientation
    omega = d.qvel[3:6]  # Angular velocity

    # Scale observations
    qj = (qj - default_angles) * config["dof_pos_scale"]
    dqj = dqj * config["dof_vel_scale"]
    gravity_orientation = get_gravity_orientation(quat)
    omega = omega * config["ang_vel_scale"]

    # Gait phase
    period = 0.8
    count = time_sim
    phase = count % period / period
    sin_phase = np.sin(2 * np.pi * phase)
    cos_phase = np.cos(2 * np.pi * phase)

    # Command - use provided cmd_values or fall back to config default
    if cmd_values is not None:
        cmd = np.array(cmd_values, dtype=np.float32)
    else:
        cmd = np.array(config["cmd_init"], dtype=np.float32)
    cmd_scaled = cmd * np.array(config["cmd_scale"], dtype=np.float32)

    # Assemble observations
    obs = np.zeros(config["num_obs"], dtype=np.float32)
    obs[:3] = omega
    obs[3:6] = gravity_orientation
    obs[6:9] = cmd_scaled
    obs[9 : 9 + config["num_actions"]] = qj
    obs[9 + config["num_actions"] : 9 + 2 * config["num_actions"]] = dqj
    obs[9 + 2 * config["num_actions"] : 9 + 3 * config["num_actions"]] = action
    obs[9 + 3 * config["num_actions"] : 9 + 3 * config["num_actions"] + 2] = np.array(
        [sin_phase, cos_phase]
    )

    return obs


if __name__ == "__main__":
    # 解析命令行参数
    import argparse

    parser = argparse.ArgumentParser(
        description="H1_2 混合控制系统部署工具 - 下半身策略控制(12 DOF) + 上半身轨迹控制(15 DOF)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available trajectory types:
    Static poses: pose_arms_forward, pose_t_shape, etc.
    Dual arm motions: 2arms_circles, 2arms_waving
    Single arm motions: 1arm_circles, 1arm_waving
    Complex motions: taichi, boxing, random
        """,
    )
    parser.add_argument(
        "config_file", type=str, help="配置文件名（位于 configs/ 目录）"
    )
    parser.add_argument(
        "--trajectory",
        "-t",
        type=str,
        default="2arms_circles",
        choices=[
            # Static poses
            "pose_arms_forward",
            "pose_t_shape",
            "pose_arms_up",
            "pose_left_down_right_forward",
            "pose_left_down_right_side",
            "pose_torso_side_twist",
            # Dynamic trajectories - dual arm
            "2arms_circles",
            "2arms_waving",
            # Dynamic trajectories - single arm
            "1arm_circles",
            "1arm_waving",
            # Complex motion trajectories
            "taichi",
            "boxing",
            "random",
        ],
        help="Upper body trajectory type (default: 2arms_circles)",
    )
    parser.add_argument(
        "--cmd",
        "-c",
        type=str,
        default="0.0,0.0,0.0",
        help="Motion command: 'fwd,side,turn' (default: 0.0,0.0,0.0)",
    )
    args = parser.parse_args()
    config_file = args.config_file
    trajectory_type = args.trajectory
    cmd_input = args.cmd

    # Parse command input
    try:
        cmd_values = [float(x.strip()) for x in cmd_input.split(",")]
        if len(cmd_values) != 3:
            raise ValueError("Command must have exactly 3 values")
        cmd_forward, cmd_sideward, cmd_turning = cmd_values
    except (ValueError, IndexError) as e:
        print(f"Error parsing command '{cmd_input}': {e}")
        print("Command format: 'forward,sideward,turn' (e.g., '1.0,0.0,0.5')")
        exit(1)

    print(f"Starting H1_2 hybrid control, trajectory: '{trajectory_type}'")
    print(f"Motion command: [{cmd_forward}, {cmd_sideward}, {cmd_turning}]")
    print("Available trajectory types:")
    print("  Static poses: pose_arms_forward, pose_t_shape, etc.")
    print("  Dual arm motions: 2arms_circles, 2arms_waving")
    print("  Single arm motions: 1arm_circles, 1arm_waving")
    print("  Complex motions: taichi, boxing, random")
    print(f"Config file: {config_file}")
    print("=" * 60)

    # Load configuration
    config_path = (
        f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/" f"{config_file}"
    )
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace(
            "{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR
        )
        xml_path = config["xml_path"].replace(
            "{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR
        )

        sim_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        # Lower body PD gains and defaults
        lower_kps = np.array(config["lower_body_kps"], dtype=np.float32)
        lower_kds = np.array(config["lower_body_kds"], dtype=np.float32)
        lower_default_angles = np.array(
            config["lower_body_default_angles"], dtype=np.float32
        )

        # Upper body PD gains and defaults
        upper_kps = np.array(config["upper_body_kps"], dtype=np.float32)
        upper_kds = np.array(config["upper_body_kds"], dtype=np.float32)
        upper_default_angles = np.array(
            config["upper_body_default_angles"], dtype=np.float32
        )

        # Combine PD gains for all joints (27 DOF total)
        all_kps = np.concatenate([lower_kps, upper_kps])
        all_kds = np.concatenate([lower_kds, upper_kds])

    print(f"Policy path: {policy_path}")
    print(f"XML path: {xml_path}")
    print(f"Lower body DOF: {len(lower_kps)}")
    print(f"Upper body DOF: {len(upper_kps)}")
    print(f"Total DOF: {len(all_kps)}")

    # Initialize control variables
    lower_action = np.zeros(config["num_actions"], dtype=np.float32)
    lower_target_dof_pos = lower_default_angles.copy()
    upper_target_dof_pos = upper_default_angles.copy()
    obs = np.zeros(config["num_obs"], dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    print(f"Model has {m.nq} position DOF and {m.nv} velocity DOF")
    print(f"Model has {m.nu} actuators")

    # Load policy
    policy = torch.jit.load(policy_path)
    print("Policy loaded successfully")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration.
        start = time.time()

        while viewer.is_running() and (time.time() - start) < sim_duration:
            step_start = time.time()
            current_time = time.time() - start

            # Combine target positions for all joints
            all_target_dof_pos = np.concatenate(
                [lower_target_dof_pos, upper_target_dof_pos]
            )

            # Current joint pos
            # Current joint positions (27 DOF: 12 lower + 15 upper)
            current_joint_pos = d.qpos[7:34]  # Skip floating base (7 DOF)
            current_joint_vel = d.qvel[6:33]  # Skip floating base (6 DOF)

            # Compute control torques using PD control
            tau = pd_control(
                all_target_dof_pos,
                current_joint_pos,
                all_kps,
                np.zeros_like(all_kds),
                current_joint_vel,
                all_kds,
            )

            # Apply control torques
            d.ctrl[:] = tau

            # Step physics
            mujoco.mj_step(m, d)
            counter += 1
            # Update control at decimated frequency
            if counter % control_decimation == 0:
                # Generate upper body trajectory
                upper_target_dof_pos = generate_upper_body_trajectory(
                    current_time, config, trajectory_type
                )
                upper_target_dof_pos += upper_default_angles

                # Create observations for policy (lower body only)
                cmd_values = [cmd_forward, cmd_sideward, cmd_turning]
                obs = extract_observations(
                    d,
                    lower_default_angles,
                    config,
                    lower_action,
                    current_time,
                    cmd_values,
                )

                # Get action from policy
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                lower_action = policy(obs_tensor).detach().numpy().squeeze()
                lower_action = lower_action[: config["num_actions"]]

                # Transform action to target positions for lower body
                action_scaled = lower_action * config["action_scale"]
                lower_target_dof_pos = action_scaled + lower_default_angles

                # Debug output
                debug_freq = control_decimation * 10
                if counter % debug_freq == 0:  # Print every 10 control steps
                    print(f"Time: {current_time:.2f}s")
                    print(f"Lower body targets: {lower_target_dof_pos[:6]}")
                    print(f"Upper body targets: {upper_target_dof_pos[:4]}")
                    print("---")

            # Sync viewer
            viewer.sync()

            # Time keeping
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    print("Simulation completed!")
