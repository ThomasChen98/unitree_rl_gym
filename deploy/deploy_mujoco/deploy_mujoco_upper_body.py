import time
import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml


def get_gravity_orientation(quaternion):
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


def generate_upper_body_motion(time_step, period=4.0):
    """
    生成上身有规律的运动
    返回上身关节的目标角度

    假设关节顺序 (基于27DOF模型):
    - torso_joint (1个)
    - 上身关节 (14个):
      - left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw
      - left_elbow_pitch, left_elbow_roll
      - left_wrist_pitch, left_wrist_yaw
      - right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw
      - right_elbow_pitch, right_elbow_roll
      - right_wrist_pitch, right_wrist_yaw
    """
    t = time_step
    phase = (t % period) / period * 2 * np.pi
    
    # 创建上身15个关节的目标角度 (torso + 14个上身关节)
    upper_body_targets = np.zeros(15)
    
    # Torso保持直立
    upper_body_targets[0] = 0.0  # torso_joint
    
    # 双臂圆周运动参数
    arm_radius = 0.5  # 圆周运动半径(弧度)
    arm_center_pitch = 0.2  # 肩膀俯仰中心位置
    arm_center_roll = 0.3   # 肩膀横滚中心位置
    
    # 左臂圆周运动 (绕Y轴)
    left_shoulder_pitch = arm_center_pitch + arm_radius * np.sin(phase)
    left_shoulder_roll = arm_center_roll + arm_radius * np.cos(phase)
    left_shoulder_yaw = 0.1 * np.sin(phase * 2)  # 轻微摆动
    left_elbow_pitch = 0.5 + 0.3 * np.sin(phase * 1.5)  # 肘部弯曲
    left_elbow_roll = 0.0
    left_wrist_pitch = 0.2 * np.sin(phase * 3)
    left_wrist_yaw = 0.1 * np.cos(phase * 2)
    
    # 右臂圆周运动 (与左臂相位相反)
    phase_right = phase + np.pi
    right_shoulder_pitch = (arm_center_pitch +
                            arm_radius * np.sin(phase_right))
    right_shoulder_roll = (-arm_center_roll -
                           arm_radius * np.cos(phase_right))  # 右臂相反方向
    right_shoulder_yaw = 0.1 * np.sin(phase_right * 2)
    right_elbow_pitch = 0.5 + 0.3 * np.sin(phase_right * 1.5)
    right_elbow_roll = 0.0
    right_wrist_pitch = 0.2 * np.sin(phase_right * 3)
    right_wrist_yaw = 0.1 * np.cos(phase_right * 2)
    
    # 填入目标角度
    upper_body_targets[1] = left_shoulder_pitch
    upper_body_targets[2] = left_shoulder_roll
    upper_body_targets[3] = left_shoulder_yaw
    upper_body_targets[4] = left_elbow_pitch
    upper_body_targets[5] = left_elbow_roll
    upper_body_targets[6] = left_wrist_pitch
    upper_body_targets[7] = left_wrist_yaw
    upper_body_targets[8] = right_shoulder_pitch
    upper_body_targets[9] = right_shoulder_roll
    upper_body_targets[10] = right_shoulder_yaw
    upper_body_targets[11] = right_elbow_pitch
    upper_body_targets[12] = right_elbow_roll
    upper_body_targets[13] = right_wrist_pitch
    upper_body_targets[14] = right_wrist_yaw
    
    return upper_body_targets


if __name__ == "__main__":
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file", type=str, help="config file name in the config folder"
    )
    args = parser.parse_args()
    config_file = args.config_file

    config_path = (f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/"
                   f"configs/{config_file}")
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace(
            "{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR
        )
        # 使用27DOF模型而不是12DOF
        xml_path = f"{LEGGED_GYM_ROOT_DIR}/resources/robots/h1_2/h1_2_27dof.xml"

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        # 下身12个关节的PD参数
        leg_kps = np.array(config["kps"], dtype=np.float32)
        leg_kds = np.array(config["kds"], dtype=np.float32)
        
        # 上身15个关节的PD参数 (torso + 14个上身关节)
        upper_kps = np.array([100] + [50]*14, dtype=np.float32)  # 上身用较小的增益
        upper_kds = np.array([5] + [2]*14, dtype=np.float32)
        
        # 组合所有关节的PD参数 (下身12 + 上身15 = 27)
        all_kps = np.concatenate([leg_kps, upper_kps])
        all_kds = np.concatenate([leg_kds, upper_kds])

        default_angles = np.array(config["default_angles"], dtype=np.float32)
        # 为上身添加默认角度 (torso + 14个上身关节)
        upper_default = np.array([0.0,  # torso
                                 0.4, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0,  # 左臂
                                 0.4, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0]) # 右臂
        all_default_angles = np.concatenate([default_angles, upper_default])

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]  # 下身12个关节
        print(f"Number of leg actions: {num_actions}")
        num_obs = config["num_obs"]
        print(f"Number of observations: {num_obs}")

        cmd = np.array(config["cmd_init"], dtype=np.float32)

    # define context variables
    leg_action = np.zeros(num_actions, dtype=np.float32)  # 下身动作
    target_dof_pos = all_default_angles.copy()  # 所有27个关节的目标位置
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model (27 DOF)
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    
    print(f"Total DOF in model: {m.nq - 7}")  # 减去浮动基座的7个DOF
    print(f"Actuated DOF: {m.nu}")

    # load policy (for lower body only)
    policy = torch.jit.load(policy_path)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            
            # PD控制所有27个关节
            tau = pd_control(
                target_dof_pos, d.qpos[7:], all_kps, 
                np.zeros_like(all_kds), d.qvel[6:], all_kds
            )
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # 1. 生成上身的有规律运动
                current_time = counter * simulation_dt
                upper_body_targets = generate_upper_body_motion(current_time)
                
                # 2. 为下身策略创建观测 (只使用前12个关节)
                qj_legs = d.qpos[7:7+12]  # 前12个关节 (下身)
                dqj_legs = d.qvel[6:6+12]  # 前12个关节速度
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]

                # 标准化下身关节数据
                qj_legs_normalized = (qj_legs - default_angles) * dof_pos_scale
                dqj_legs_normalized = dqj_legs * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                omega_normalized = omega * ang_vel_scale

                # 步态相位
                period = 0.8
                count = counter * simulation_dt
                phase = count % period / period
                sin_phase = np.sin(2 * np.pi * phase)
                cos_phase = np.cos(2 * np.pi * phase)

                # 构建观测向量 (只包含下身信息)
                obs[:3] = omega_normalized
                obs[3:6] = gravity_orientation
                obs[6:9] = cmd * cmd_scale
                obs[9 : 9 + num_actions] = qj_legs_normalized
                obs[9 + num_actions : 9 + 2 * num_actions] = dqj_legs_normalized
                obs[9 + 2 * num_actions : 9 + 3 * num_actions] = leg_action
                obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array(
                    [sin_phase, cos_phase]
                )
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)

                # 3. 策略推理 (只针对下身)
                leg_action = policy(obs_tensor).detach().numpy().squeeze()

                # 4. 更新目标位置
                # 下身12个关节使用策略输出
                target_dof_pos[:12] = leg_action * action_scale + default_angles
                # 上身15个关节使用有规律运动
                target_dof_pos[12:] = upper_body_targets

                # 打印调试信息
                if counter % (control_decimation * 50) == 0:  # 每秒打印一次
                    print(f"Time: {current_time:.2f}s, "
                          f"Leg action range: [{leg_action.min():.3f}, {leg_action.max():.3f}], "
                          f"Upper body range: [{upper_body_targets.min():.3f}, {upper_body_targets.max():.3f}]")

            viewer.sync()

            # 时间同步
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
