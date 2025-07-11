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

    # Torso joint: forward bending motion
    upper_body_targets[0] = torso_amp * math.sin(torso_phase)

    # Left arm joints
    upper_body_targets[1] = arm_offset + traj_amp * math.sin(phase)
    upper_body_targets[2] = 0.3 * math.sin(phase)
    upper_body_targets[3] = 0.0
    upper_body_targets[4] = 1.2 + 0.3 * math.cos(phase)
    upper_body_targets[5] = 0.0
    upper_body_targets[6] = 0.0
    upper_body_targets[7] = 0.0

    # Right arm joints - mirror left arm
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
    upper_body_targets[1:8] = 0.0

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
    upper_body_targets[5:8] = 0.0

    # 右臂（相位相反）
    upper_body_targets[8] = 0.8 + push_amplitude * math.sin(phase + math.pi)
    upper_body_targets[9] = -0.3
    upper_body_targets[10] = -0.2 * math.sin(phase + math.pi)
    upper_body_targets[11] = 1.0 - 0.3 * math.sin(phase + math.pi)
    upper_body_targets[12:15] = 0.0

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
        upper_body_targets[5:8] = 0.0

        # 右拳防守
        upper_body_targets[8] = 0.3
        upper_body_targets[9] = -0.2
        upper_body_targets[10] = 0.0
        upper_body_targets[11] = 1.3
        upper_body_targets[12:15] = 0.0
    else:
        # 右拳出击
        upper_body_targets[1] = 0.3
        upper_body_targets[2] = 0.2
        upper_body_targets[3] = 0.0
        upper_body_targets[4] = 1.3
        upper_body_targets[5:8] = 0.0

        # 左拳防守
        upper_body_targets[8] = 0.5 - 0.4 * math.sin(phase)  # 快速前伸
        upper_body_targets[9] = -0.2
        upper_body_targets[10] = 0.0
        upper_body_targets[11] = 1.5 + 0.5 * math.sin(phase)  # 肘部伸展
        upper_body_targets[12:15] = 0.0

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


def trajectory_random_motion(time_sim, config):
    """随机动作 - 每个关节随机小幅度变化"""
    import random

    upper_body_targets = np.zeros(15)

    # 随机动作的参数
    base_amplitude = 0.3  # 基础随机幅度
    freq_variation = 0.5  # 频率变化范围

    # 为每个关节生成不同的随机种子（基于时间）
    random.seed(int(time_sim * 1000) % 10000)

    # 躯干随机轻微摆动
    upper_body_targets[0] = base_amplitude * 0.5 * (random.random() - 0.5)

    # 左臂随机动作
    for i in range(1, 8):
        # 每个关节有不同的随机变化
        random_factor = random.random() - 0.5  # -0.5 到 0.5
        time_factor = math.sin(
            2 * math.pi * (0.2 + freq_variation * random.random()) * time_sim
        )

        if i == 1:  # 左肩pitch - 更大范围
            upper_body_targets[i] = base_amplitude * 2.0 * random_factor * time_factor
        elif i == 2:  # 左肩roll - 中等范围
            upper_body_targets[i] = base_amplitude * 1.5 * random_factor * time_factor
        elif i == 4:  # 左肘 - 保持正值避免过度伸展
            upper_body_targets[i] = (
                abs(base_amplitude * random_factor * time_factor) + 0.1
            )
        else:  # 其他关节
            upper_body_targets[i] = base_amplitude * random_factor * time_factor

    # 右臂随机动作（使用不同的随机种子）
    for i in range(8, 15):
        random_factor = random.random() - 0.5
        time_factor = math.sin(
            2 * math.pi * (0.15 + freq_variation * random.random()) * time_sim
        )

        if i == 8:  # 右肩pitch - 更大范围
            upper_body_targets[i] = base_amplitude * 2.0 * random_factor * time_factor
        elif i == 9:  # 右肩roll - 中等范围，负值
            upper_body_targets[i] = -base_amplitude * 1.5 * random_factor * time_factor
        elif i == 11:  # 右肘 - 保持正值避免过度伸展
            upper_body_targets[i] = (
                abs(base_amplitude * random_factor * time_factor) + 0.1
            )
        else:  # 其他关节
            upper_body_targets[i] = base_amplitude * random_factor * time_factor

    return upper_body_targets


# ========== 静止动作函数 ==========


def pose_arms_forward(time_sim, config):
    """静止动作1: 双臂前伸"""
    upper_body_targets = np.zeros(15)

    # 躯干保持直立
    upper_body_targets[0] = 0.0

    # 左臂前伸
    upper_body_targets[1] = -2.0  # 肩膀前伸
    upper_body_targets[2] = 0.0  # 肩膀内外旋
    upper_body_targets[3] = 0.0  # 肩膀俯仰
    upper_body_targets[4] = 0.0  # 肘部伸直
    upper_body_targets[5:8] = 0.0  # 手腕保持中性

    # 右臂前伸（镜像）
    upper_body_targets[8] = -2.0  # 肩膀前伸
    upper_body_targets[9] = 0.0  # 肩膀内外旋
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
    upper_body_targets[9] = 0.0  # 肩膀内外旋
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


def pose_left_up_right_down(time_sim, config):
    """静止动作6: 左臂上举，右臂下垂"""
    upper_body_targets = np.zeros(15)

    # 躯干保持直立
    upper_body_targets[0] = 0.0

    # 左臂上举
    upper_body_targets[1] = -3.0  # 肩膀后伸（向上举）
    upper_body_targets[2] = 0.0  # 肩膀内外旋
    upper_body_targets[3] = 0.0  # 肩膀俯仰
    upper_body_targets[4] = 0.0  # 肘部伸直
    upper_body_targets[5:8] = 0.0  # 手腕保持中性

    # 右臂自然下垂
    upper_body_targets[8:15] = 0.0

    return upper_body_targets


def generate_upper_body_trajectory(time_sim, config, trajectory_type="circles"):
    """Generate predefined trajectories for upper body joints"""

    # 轨迹选择字典
    trajectory_functions = {
        # 动态轨迹
        "circles": trajectory_arm_circles,
        "waving": trajectory_waving_hello,
        "taichi": trajectory_tai_chi,
        "boxing": trajectory_boxing,
        "dancing": trajectory_dancing,
        "stretching": trajectory_stretching,
        "random": trajectory_random_motion,
        # 静止上身动作
        "pose_arms_forward": pose_arms_forward,
        "pose_left_down_right_forward": pose_left_down_right_forward,
        "pose_t_shape": pose_t_shape,
        "pose_left_down_right_side": pose_left_down_right_side,
        "pose_torso_side_twist": pose_torso_side_twist,
        "pose_left_up_right_down": pose_left_up_right_down,
    }

    # 获取对应的轨迹函数
    if trajectory_type in trajectory_functions:
        return trajectory_functions[trajectory_type](time_sim, config)
    else:
        print(
            f"Warning: Unknown trajectory type '{trajectory_type}', "
            f"using default 'circles'"
        )
        return trajectory_arm_circles(time_sim, config)


def extract_observations(d, default_angles, config, action, time_sim):
    """Extract observations for the policy (same as original)"""
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

    # Command
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
    # Parse command line arguments
    import argparse

    parser = argparse.ArgumentParser(
        description="H1_2 Hybrid Control Deployment with Trajectory Selection"
    )
    parser.add_argument(
        "config_file", type=str, help="config file name in the config folder"
    )
    parser.add_argument(
        "--trajectory",
        "-t",
        type=str,
        default="circles",
        choices=[
            # 动态轨迹
            "circles",
            "waving",
            "taichi",
            "boxing",
            "dancing",
            "stretching",
            "random",
            # 静止上身动作
            "pose_arms_forward",
            "pose_left_down_right_forward",
            "pose_t_shape",
            "pose_left_down_right_side",
            "pose_torso_side_twist",
            "pose_left_up_right_down",
        ],
        help="Upper body trajectory type (default: circles)",
    )
    args = parser.parse_args()
    config_file = args.config_file
    trajectory_type = args.trajectory

    print(f"Starting H1_2 hybrid control with '{trajectory_type}' " f"trajectory...")
    print(
        "Available trajectories: circles, waving, taichi, boxing, "
        "dancing, stretching"
    )
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

        simulation_duration = config["simulation_duration"]
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

        while viewer.is_running() and time.time() - start < simulation_duration:
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
                obs = extract_observations(
                    d, lower_default_angles, config, lower_action, current_time
                )

                # Get action from policy
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                lower_action = policy(obs_tensor).detach().numpy().squeeze()
                lower_action = lower_action[: config["num_actions"]]

                # Transform action to target positions for lower body
                lower_target_dof_pos = (
                    lower_action * config["action_scale"] + lower_default_angles
                )

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
