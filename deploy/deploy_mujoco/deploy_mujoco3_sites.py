#!/usr/bin/env python3
"""
H1_2混合控制部署脚本 - 使用sites可视化重心轨迹
基于测试结果，使用可行的sites方法显示重心轨迹
"""

import os
import sys
import time
import yaml
import numpy as np
import mujoco
import mujoco.viewer
import torch

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "..")
sys.path.append(project_root)

# 注释掉有问题的导入，直接在需要时加载
# from legged_gym.envs.h1_2.h1_2_env import H1_2Env


def generate_upper_body_trajectory(current_time, upper_body_dof, trajectory_type):
    """生成上半身轨迹"""
    trajectory = np.zeros(upper_body_dof)

    if trajectory_type == "boxing":
        # 拳击动作
        freq = 1.5
        cycle_time = current_time * freq
        phase = cycle_time % 1.0

        if phase < 0.3:  # 出拳阶段
            progress = phase / 0.3
            smoothed = 0.5 * (1 - np.cos(progress * np.pi))
            punch_amplitude = 0.8
            trajectory[0] = smoothed * punch_amplitude
            trajectory[2] = smoothed * punch_amplitude
        elif phase < 0.5:  # 保持阶段
            trajectory[0] = 0.8
            trajectory[2] = 0.8
        else:  # 回收阶段
            progress = (phase - 0.5) / 0.5
            smoothed = 0.5 * (1 + np.cos(progress * np.pi))
            punch_amplitude = 0.8
            trajectory[0] = smoothed * punch_amplitude
            trajectory[2] = smoothed * punch_amplitude

    elif trajectory_type == "2arm_circles":
        # 双臂画圆
        freq = 0.5
        angle = current_time * freq * 2 * np.pi
        amplitude = 0.6
        trajectory[0] = amplitude * np.sin(angle)
        trajectory[1] = amplitude * np.cos(angle)
        trajectory[3] = amplitude * np.sin(angle + np.pi)
        trajectory[4] = amplitude * np.cos(angle + np.pi)

    elif trajectory_type == "random":
        # 随机运动
        np.random.seed(int(current_time * 10))
        for i in range(upper_body_dof):
            freq = 0.8 + 0.4 * np.random.random()
            phase = np.random.random() * 2 * np.pi
            amplitude = 0.3 + 0.4 * np.random.random()
            trajectory[i] = amplitude * np.sin(current_time * freq + phase)

    return trajectory


def extract_observations(d, default_dof_pos, lower_body_dof, action, current_time):
    """提取观测数据"""
    obs = np.zeros(48)

    obs[:lower_body_dof] = d.qpos[7 : 7 + lower_body_dof] - default_dof_pos
    obs[lower_body_dof : 2 * lower_body_dof] = d.qvel[6 : 6 + lower_body_dof]
    obs[2 * lower_body_dof : 3 * lower_body_dof] = action

    return obs


def setup_com_visualization(m, d):
    """设置重心轨迹可视化

    使用现有的IMU site作为重心标记
    修改其属性使其更适合重心可视化
    """
    # 查找IMU site
    imu_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "imu")

    if imu_site_id >= 0:
        print(f"✅ 找到IMU site，ID: {imu_site_id}")

        # 修改IMU site的外观，使其更适合作为重心标记
        # 设置为球形
        m.site_type[imu_site_id] = mujoco.mjtGeom.mjGEOM_SPHERE

        # 设置大小（半径）
        m.site_size[imu_site_id] = [0.03, 0.03, 0.03]  # 3cm半径的球

        # 设置为亮绿色，表示当前重心
        m.site_rgba[imu_site_id] = [0.0, 1.0, 0.0, 0.9]

        print(f"✅ IMU site已配置为重心标记")
        return imu_site_id
    else:
        print(f"❌ 未找到IMU site")
        return -1


def update_com_visualization(m, d, imu_site_id, com_pos):
    """更新重心可视化

    将IMU site移动到当前重心位置
    """
    if imu_site_id >= 0:
        # 更新site位置为当前重心位置
        m.site_pos[imu_site_id] = com_pos[:3]

        # 重新计算相关变换
        mujoco.mj_forward(m, d)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="H1_2混合控制部署 - Sites重心轨迹可视化版本"
    )
    parser.add_argument("config_file", type=str, help="配置文件名")
    parser.add_argument(
        "--trajectory",
        "-t",
        type=str,
        default="boxing",
        choices=["boxing", "2arm_circles", "random", "pose_arms_forward"],
        help="轨迹类型",
    )
    parser.add_argument(
        "--duration", "-d", type=float, default=15.0, help="仿真时长（秒）"
    )

    args = parser.parse_args()

    print(f"=== H1_2混合控制部署 - Sites重心轨迹可视化 ===")
    print(f"轨迹类型: {args.trajectory}")
    print(f"仿真时长: {args.duration}秒")

    # 配置文件路径
    config_path = os.path.join(current_dir, "configs", args.config_file)
    if not os.path.exists(config_path):
        print(f"错误：配置文件不存在 {config_path}")
        return

    # 加载配置
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # XML和策略路径
    xml_path = os.path.join(project_root, "resources", "robots", "h1_2", "scene.xml")
    policy_path = os.path.join(
        project_root, "logs", "h1_2", "exported", "policies", "policy_lstm_1.pt"
    )

    if not os.path.exists(xml_path):
        print(f"错误：XML文件不存在 {xml_path}")
        return
    if not os.path.exists(policy_path):
        print(f"错误：策略文件不存在 {policy_path}")
        return

    # 加载模型和策略
    print("加载MuJoCo模型...")
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)

    print("加载强化学习策略...")
    policy = torch.jit.load(policy_path, map_location="cpu")
    policy.eval()

    # 初始化
    mujoco.mj_resetData(m, d)

    # 设置重心可视化
    imu_site_id = setup_com_visualization(m, d)

    # 默认关节角度
    lower_body_dof = config["num_actions"]  # 下半身DOF数量
    total_dof = 27  # H1_2机器人总DOF
    upper_body_dof = total_dof - lower_body_dof  # 上半身DOF数量

    lower_default_angles = np.array(config["lower_body_default_angles"])
    upper_default_angles = np.array(config["upper_body_default_angles"])

    # 设置初始姿态
    d.qpos[7 : 7 + lower_body_dof] = lower_default_angles
    d.qpos[7 + lower_body_dof :] = upper_default_angles

    mujoco.mj_forward(m, d)

    # 控制参数
    control_decimation = 4
    trajectory_update_freq = 10  # 每10步更新重心显示

    # 重心轨迹记录
    com_trajectory = []
    max_trajectory_points = 100

    # 仿真参数
    max_steps = int(args.duration / m.opt.timestep)
    lower_action = np.zeros(config["num_actions"])

    print(f"开始仿真... 总步数: {max_steps}")
    print(f"重心标记: 绿色球体会跟随机器人重心移动")

    # 启动viewer
    with mujoco.viewer.launch_passive(m, d) as viewer:
        print("✅ Viewer启动成功")

        # 确保sites显示
        if hasattr(viewer.opt, "sitegroup"):
            for i in range(len(viewer.opt.sitegroup)):
                viewer.opt.sitegroup[i] = 1

        # 主仿真循环
        counter = 0
        for step in range(max_steps):
            step_start = time.time()
            current_time = step * m.opt.timestep

            # 计算控制扭矩
            lower_kps = np.array(config["lower_body_kps"])
            lower_kds = np.array(config["lower_body_kds"])
            upper_kps = np.array(config["upper_body_kps"])
            upper_kds = np.array(config["upper_body_kds"])

            # 合并PD增益
            all_kps = np.concatenate([lower_kps, upper_kps])
            all_kds = np.concatenate([lower_kds, upper_kds])

            # 上半身和下半身目标位置
            if counter % control_decimation == 0:
                # 生成上半身轨迹
                upper_target_dof_pos = generate_upper_body_trajectory(
                    current_time, upper_body_dof, args.trajectory
                )
                upper_target_dof_pos += upper_default_angles

                # 创建下半身观测并获取动作
                obs = extract_observations(
                    d, lower_default_angles, lower_body_dof, lower_action, current_time
                )
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                lower_action = policy(obs_tensor).detach().numpy().squeeze()
                lower_action = lower_action[: config["num_actions"]]

                # 转换动作为目标位置
                lower_target_dof_pos = (
                    lower_action * config["action_scale"] + lower_default_angles
                )

            # 合并目标位置
            all_target_dof_pos = np.concatenate(
                [lower_target_dof_pos, upper_target_dof_pos]
            )

            # 计算PD控制
            current_joint_pos = d.qpos[7:]
            current_joint_vel = d.qvel[6:]

            tau = (
                all_kps * (all_target_dof_pos - current_joint_pos)
                - all_kds * current_joint_vel
            )

            # 应用控制
            d.ctrl[:] = tau

            # 步进物理
            mujoco.mj_step(m, d)
            counter += 1

            # 更新重心轨迹显示
            if counter % trajectory_update_freq == 0:
                # 计算重心
                mujoco.mj_rnePostConstraint(m, d)
                com_pos = d.subtree_com[0].copy()

                if not np.isnan(com_pos).any():
                    com_trajectory.append(com_pos.copy())

                    # 限制轨迹点数量
                    if len(com_trajectory) > max_trajectory_points:
                        com_trajectory.pop(0)

                    # 更新重心可视化
                    update_com_visualization(m, d, imu_site_id, com_pos)

                    if counter % 200 == 0:  # 每200步输出一次
                        print(
                            f"Step {counter}: CoM = [{com_pos[0]:.3f}, "
                            f"{com_pos[1]:.3f}, {com_pos[2]:.3f}]"
                        )
                        print(f"轨迹点数: {len(com_trajectory)}")

            # 同步viewer
            viewer.sync()

            # 控制仿真速度
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            # 进度输出
            if counter % 1000 == 0:
                progress = (step / max_steps) * 100
                print(f"仿真进度: {progress:.1f}%")

        print("\n仿真完成！")
        print(f"最终轨迹点数: {len(com_trajectory)}")

        if len(com_trajectory) > 0:
            com_array = np.array(com_trajectory)
            print(f"重心轨迹统计:")
            print(
                f"  X范围: {com_array[:, 0].min():.3f} ~ "
                f"{com_array[:, 0].max():.3f} m"
            )
            print(
                f"  Y范围: {com_array[:, 1].min():.3f} ~ "
                f"{com_array[:, 1].max():.3f} m"
            )
            print(
                f"  Z范围: {com_array[:, 2].min():.3f} ~ "
                f"{com_array[:, 2].max():.3f} m"
            )

        print(f"\n✅ 重心轨迹可视化成功！")
        print(f"您应该能看到绿色球体跟随机器人重心移动")

        # 保持窗口开启
        print(f"按Ctrl+C退出...")
        try:
            while viewer.is_running():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("退出程序")


if __name__ == "__main__":
    main()
