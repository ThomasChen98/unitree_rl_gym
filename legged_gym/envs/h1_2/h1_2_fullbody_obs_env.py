from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch
import numpy as np

class H1_2FullBodyObsRobot(LeggedRobot):
    """
    全身观测 + 下半身策略环境
    - 观测包含全身27DOF信息
    - 策略只输出下半身12DOF动作
    - 上半身每episode随机固定角度
    """
    
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        # 上半身关节索引 (12-26为上半身15个关节)
        self.upper_body_indices = list(range(12, 27))
        # 下半身关节索引 (0-11为下半身12个关节) 
        self.lower_body_indices = list(range(0, 12))
        
        # 存储每个环境的上半身随机目标角度
        self.upper_body_targets = torch.zeros(self.num_envs, 15, device=self.device)
        
        # 初始化上半身随机目标
        self._randomize_upper_body_targets(torch.arange(self.num_envs, device=self.device))
    
    def _get_noise_scale_vec(self, cfg):
        """ 设置噪声缩放向量 - 适配77维观测 """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        
        # 观测结构: ang_vel(3) + gravity(3) + commands(3) + dof_pos(27) + dof_vel(27) + actions(12) + phase(2)
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0.  # commands
        noise_vec[9:36] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[36:63] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[63:75] = 0.  # actions (12)
        noise_vec[75:77] = 0.  # phase (2)
        
        return noise_vec

    def _init_foot(self):
        """初始化脚部状态跟踪"""
        self.feet_num = len(self.feet_indices)
        
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _init_buffers(self):
        super()._init_buffers()
        self._init_foot()

    def update_feet_state(self):
        """更新脚部状态"""
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _post_physics_step_callback(self):
        """物理步后回调 - 更新状态和相位"""
        self.update_feet_state()

        # 步态相位计算
        period = 0.8
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)
        
        return super()._post_physics_step_callback()
    
    def _randomize_upper_body_targets(self, env_ids):
        """为指定环境随机化上半身目标角度"""
        if len(env_ids) == 0:
            return
            
        # 从配置中获取随机范围
        angle_range = self.cfg.env.upper_body_angle_range
        
        # 为每个重置的环境生成新的随机上半身目标
        random_angles = torch.rand(len(env_ids), 15, device=self.device) * (angle_range[1] - angle_range[0]) + angle_range[0]
        
        # 可以为不同关节设置不同的随机范围
        # 躯干: 较小摆动
        random_angles[:, 0] *= 0.3  # torso_joint
        
        # 肩部: 中等范围
        random_angles[:, 1:4] *= 0.8   # left shoulder
        random_angles[:, 8:11] *= 0.8  # right shoulder
        
        # 肘部和腕部: 较大范围
        random_angles[:, 4:8] *= 1.0   # left elbow+wrist
        random_angles[:, 11:15] *= 1.0 # right elbow+wrist
        
        self.upper_body_targets[env_ids] = random_angles
        
    def _compute_torques(self, actions):
        """计算力矩 - 扩展为27DOF控制
        actions: 12维策略输出 (下半身)
        返回: 27维力矩输出 (全身)
        """
        # 确保输入动作是12维
        assert actions.shape[1] == 12, f"Expected 12 actions, got {actions.shape[1]}"
        
        # 下半身12DOF使用策略输出
        lower_body_actions = actions * self.cfg.control.action_scale
        
        # 上半身15DOF使用固定目标角度 (相对于default角度的偏移)
        upper_body_actions = self.upper_body_targets
        
        # 合并为27DOF动作
        full_actions = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        full_actions[:, :12] = lower_body_actions  # 下半身12DOF
        full_actions[:, 12:] = upper_body_actions  # 上半身15DOF
        
        # 计算PD控制力矩
        control_type = self.cfg.control.control_type
        if control_type == "P":
            torques = (self.p_gains * (full_actions + self.default_dof_pos - self.dof_pos) 
                      - self.d_gains * self.dof_vel)
        elif control_type == "V":
            torques = (self.p_gains * (full_actions - self.dof_vel) 
                      - self.d_gains * (self.dof_vel - self.last_dof_vel) / self.sim_params.dt)
        elif control_type == "T":
            torques = full_actions
        else:
            raise NameError(f"Unknown controller type: {control_type}")
            
        return torch.clip(torques, -self.torque_limits, self.torque_limits)
    
    def compute_observations(self):
        """计算观测 - 包含全身27DOF信息"""
        sin_phase = torch.sin(2 * np.pi * self.phase).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase).unsqueeze(1)
        
        # 全身观测: 包含所有27个关节的位置和速度
        self.obs_buf = torch.cat((
            self.base_ang_vel * self.obs_scales.ang_vel,                                    # 3维
            self.projected_gravity,                                                         # 3维
            self.commands[:, :3] * self.commands_scale,                                     # 3维
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,               # 27维 (全身)
            self.dof_vel * self.obs_scales.dof_vel,                                        # 27维 (全身)
            self.actions,                                                                   # 12维 (只有下半身动作)
            sin_phase,                                                                      # 1维
            cos_phase                                                                       # 1维
        ), dim=-1)  # 总计: 3+3+3+27+27+12+1+1 = 77维
        
        # 特权观测额外包含线速度
        self.privileged_obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,                                   # 3维
            self.base_ang_vel * self.obs_scales.ang_vel,                                   # 3维
            self.projected_gravity,                                                        # 3维
            self.commands[:, :3] * self.commands_scale,                                    # 3维
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,              # 27维
            self.dof_vel * self.obs_scales.dof_vel,                                       # 27维
            self.actions,                                                                  # 12维
            sin_phase,                                                                     # 1维
            cos_phase                                                                      # 1维
        ), dim=-1)  # 总计: 3+3+3+3+27+27+12+1+1 = 80维
        
        # 添加噪声
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
    
    def reset_idx(self, env_ids):
        """重置环境时重新随机化上半身目标"""
        # 先调用父类重置
        super().reset_idx(env_ids)
        
        # 为重置的环境重新随机化上半身目标角度
        self._randomize_upper_body_targets(env_ids)

    # ==================== 奖励函数 ====================
    
    def _reward_contact(self):
        """基于相位的接触奖励"""
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
            res += ~(contact ^ is_stance)
        return res
    
    def _reward_feet_swing_height(self):
        """摆动腿高度控制奖励"""
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        pos_error = torch.square(self.feet_pos[:, :, 2] - 0.08) * ~contact
        return torch.sum(pos_error, dim=(1))
    
    def _reward_alive(self):
        """存活奖励"""
        return 1.0
    
    def _reward_contact_no_vel(self):
        """接触时脚部静止奖励"""
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1, 2))
    
    def _reward_hip_pos(self):
        """髋关节位置奖励"""
        return torch.sum(torch.square(self.dof_pos[:, [0, 2, 6, 8]]), dim=1)
    
    def _reward_upper_body_tracking(self):
        """奖励上半身跟踪目标角度"""
        upper_body_pos_error = torch.square(
            self.dof_pos[:, 12:] - (self.default_dof_pos[12:] + self.upper_body_targets)
        )
        return torch.sum(upper_body_pos_error, dim=1)
