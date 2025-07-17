# 上身随机干扰训练策略实现示例

import torch
import numpy as np


class UpperBodyDisturbanceStrategy:
    """
    在27DOF训练中实现上身随机运动干扰的策略
    目标: 训练下身在上身随机运动下仍能稳定行走
    """

    def __init__(self, cfg):
        self.cfg = cfg
        # 上身关节索引 (基于27DOF配置)
        self.upper_body_indices = list(range(12, 27))  # 第12-26个关节为上身
        self.torso_index = 12  # 躯干关节
        self.arm_indices = list(range(13, 27))  # 手臂关节

        # 干扰参数
        self.disturbance_frequency = 0.5  # 干扰频率 (Hz)
        self.disturbance_amplitude = {
            "torso": 0.2,  # 躯干摆动幅度 (rad)
            "arm": 0.5,  # 手臂运动幅度 (rad)
        }

    def add_upper_body_disturbance(self, env):
        """
        方法1: 在动作输出后添加上身随机干扰
        """
        # 1. 只训练下身12DOF策略
        lower_body_actions = env.actions[:, :12]  # 策略输出的下身动作

        # 2. 生成上身随机目标位置
        upper_body_targets = self._generate_random_upper_targets(env)

        # 3. 合并动作
        full_actions = torch.cat([lower_body_actions, upper_body_targets], dim=1)

        return full_actions

    def _generate_random_upper_targets(self, env):
        """生成上身关节的随机目标位置"""
        num_envs = env.num_envs
        time = env.episode_length_buf * env.dt

        # 不同频率的正弦波组合
        freq1 = self.disturbance_frequency
        freq2 = self.disturbance_frequency * 1.3
        freq3 = self.disturbance_frequency * 0.7

        upper_targets = torch.zeros(num_envs, 15, device=env.device)

        # 躯干随机摆动
        upper_targets[:, 0] = self.disturbance_amplitude["torso"] * torch.sin(
            2 * np.pi * freq1 * time
        ) + 0.5 * self.disturbance_amplitude["torso"] * torch.sin(
            2 * np.pi * freq2 * time
        )

        # 左臂随机运动 (肩膀、肘部、腕部)
        for i in range(1, 8):  # 左臂7个关节
            phase_offset = i * 0.3  # 不同关节的相位差
            upper_targets[:, i] = self.disturbance_amplitude["arm"] * torch.sin(
                2 * np.pi * freq1 * time + phase_offset
            ) + 0.3 * self.disturbance_amplitude["arm"] * torch.cos(
                2 * np.pi * freq3 * time + phase_offset
            )

        # 右臂随机运动 (镜像但不同步)
        for i in range(8, 15):  # 右臂7个关节
            phase_offset = (i - 8) * 0.3 + np.pi  # 与左臂相位差π
            upper_targets[:, i] = self.disturbance_amplitude["arm"] * torch.sin(
                2 * np.pi * freq1 * time + phase_offset
            ) + 0.3 * self.disturbance_amplitude["arm"] * torch.cos(
                2 * np.pi * freq3 * time + phase_offset
            )

        return upper_targets


# 在环境配置中的实现方式
class H1_2_27DofRobustCfg:
    """
    专门针对上身干扰鲁棒性的配置
    """

    class env:
        num_observations = 47  # 只观测下身相关信息
        num_actions = 12  # 只输出下身动作
        upper_body_disturbance = True  # 启用上身干扰

    class rewards:
        class scales:
            # 增强稳定性相关奖励权重
            orientation = -2.0  # 姿态稳定 (增强)
            base_height = -15.0  # 高度稳定 (增强)
            ang_vel_xy = -0.1  # 角速度稳定 (增强)
            tracking_lin_vel = 1.5  # 速度跟踪 (增强)

            # 新增上身运动适应性奖励
            upper_body_balance = 1.0  # 上身运动下的平衡能力
            disturbance_rejection = 0.5  # 干扰拒绝能力

    class control:
        # 下身采用更强的PD参数提高抗干扰能力
        stiffness = {
            "hip_yaw_joint": 250.0,  # 增强髋部刚度
            "hip_roll_joint": 250.0,
            "hip_pitch_joint": 250.0,
            "knee_joint": 350.0,  # 增强膝部刚度
            "ankle_pitch_joint": 60.0,  # 增强踝部刚度
            "ankle_roll_joint": 60.0,
        }
        damping = {
            "hip_yaw_joint": 4.0,
            "hip_roll_joint": 4.0,
            "hip_pitch_joint": 4.0,
            "knee_joint": 6.0,  # 增强阻尼
            "ankle_pitch_joint": 3.0,
            "ankle_roll_joint": 3.0,
        }


# 实现新的奖励函数
def _reward_upper_body_balance(self):
    """
    奖励在上身运动下保持平衡的能力
    """
    # 考虑上身运动对重心的影响
    upper_body_com_offset = self._compute_upper_body_com_offset()
    balance_effort = torch.norm(self.base_ang_vel[:, :2], dim=1)  # 平衡所需的角速度

    # 奖励小的平衡调整
    return torch.exp(-balance_effort * 2.0)


def _reward_disturbance_rejection(self):
    """
    奖励对上身运动干扰的拒绝能力
    """
    # 上身运动幅度
    upper_motion = torch.norm(self.dof_vel[:, 12:], dim=1)
    # 下身稳定程度
    lower_stability = 1.0 / (
        1.0 + torch.norm(self.base_lin_vel[:, :2] - self.commands[:, :2], dim=1)
    )

    # 奖励在上身大幅运动时下身仍保持稳定
    return lower_stability * torch.tanh(upper_motion)
