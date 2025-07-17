from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class H1_2FullBodyObsCfg(LeggedRobotCfg):
    """
    全身观测 + 下半身策略配置
    - 观测: 包含全身27DOF信息
    - 动作: 只输出下半身12DOF
    - 上半身: 每episode随机固定角度
    """

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 1.05]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            # 下半身12DOF (与原始h1_2相同)
            "left_hip_yaw_joint": 0,
            "left_hip_pitch_joint": -0.16,
            "left_hip_roll_joint": 0,
            "left_knee_joint": 0.36,
            "left_ankle_pitch_joint": -0.2,
            "left_ankle_roll_joint": 0.0,
            "right_hip_yaw_joint": 0,
            "right_hip_pitch_joint": -0.16,
            "right_hip_roll_joint": 0,
            "right_knee_joint": 0.36,
            "right_ankle_pitch_joint": -0.2,
            "right_ankle_roll_joint": 0.0,
            # 上半身15DOF (将被随机化)
            "torso_joint": 0,
            "left_shoulder_pitch_joint": 0.4,
            "left_shoulder_roll_joint": 0,
            "left_shoulder_yaw_joint": 0,
            "left_elbow_pitch_joint": 0.3,
            "left_elbow_roll_joint": 0,
            "left_wrist_pitch_joint": 0,
            "left_wrist_yaw_joint": 0,
            "right_shoulder_pitch_joint": 0.4,
            "right_shoulder_roll_joint": 0,
            "right_shoulder_yaw_joint": 0,
            "right_elbow_pitch_joint": 0.3,
            "right_elbow_roll_joint": 0,
            "right_wrist_pitch_joint": 0,
            "right_wrist_yaw_joint": 0,
        }

    class env(LeggedRobotCfg.env):
        # 全身观测: 3(ang_vel) + 3(gravity) + 3(commands) + 27(dof_pos) + 27(dof_vel) + 12(actions) + 2(phase) = 77
        num_observations = 77
        # 特权观测额外包含: 3(lin_vel) = 80
        num_privileged_obs = 80
        # 只输出下半身动作
        num_actions = 12

        # 上半身随机化配置
        upper_body_randomization = True
        upper_body_angle_range = [-0.5, 0.5]  # 上半身关节随机范围 [rad]

    class control(LeggedRobotCfg.control):
        control_type = "P"

        # 下半身PD参数 (与原始相同)
        stiffness = {
            "hip_yaw_joint": 200.0,
            "hip_roll_joint": 200.0,
            "hip_pitch_joint": 200.0,
            "knee_joint": 300.0,
            "ankle_pitch_joint": 40.0,
            "ankle_roll_joint": 40.0,
            # 上半身PD参数 (用于维持固定姿态)
            "torso_joint": 160.0,
            "shoulder_pitch_joint": 60.0,
            "shoulder_roll_joint": 60.0,
            "shoulder_yaw_joint": 60.0,
            "elbow_pitch_joint": 30.0,
            "elbow_roll_joint": 30.0,
            "wrist_pitch_joint": 15.0,
            "wrist_yaw_joint": 15.0,
        }

        damping = {
            "hip_yaw_joint": 2.5,
            "hip_roll_joint": 2.5,
            "hip_pitch_joint": 2.5,
            "knee_joint": 4,
            "ankle_pitch_joint": 2.0,
            "ankle_roll_joint": 2.0,
            # 上半身阻尼
            "torso_joint": 3.0,
            "shoulder_pitch_joint": 2.5,
            "shoulder_roll_joint": 5.0,
            "shoulder_yaw_joint": 5.0,
            "elbow_pitch_joint": 1.0,
            "elbow_roll_joint": 1.0,
            "wrist_pitch_joint": 1.0,
            "wrist_yaw_joint": 1.0,
        }

        action_scale = 0.25
        decimation = 8

    class sim(LeggedRobotCfg.sim):
        dt = 0.0025

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1.0, 3.0]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.5

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/h1_2/h1_2_27dof.urdf"  # 使用27DOF模型
        name = "h1_2_fullbody_obs"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0
        flip_visual_attachments = False
        armature = 1e-3

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 1.0

        class scales(LeggedRobotCfg.rewards.scales):
            # 增强稳定性权重 (因为上半身随机姿态会增加难度)
            tracking_lin_vel = 1.2  # 稍微增强速度跟踪
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.08  # 增强角速度稳定
            orientation = -1.5  # 增强姿态稳定
            base_height = -12.0  # 增强高度稳定
            dof_acc = -2.5e-7
            dof_vel = -1e-3
            feet_air_time = 0.0
            collision = 0.0
            action_rate = -0.01
            dof_pos_limits = -5.0
            alive = 0.15
            hip_pos = -1.0
            contact_no_vel = -0.2
            feet_swing_height = -20.0
            contact = 0.18

            # 新增奖励: 惩罚上半身偏离随机目标角度
            upper_body_tracking = -0.1


class H1_2FullBodyObsCfgPPO(LeggedRobotCfgPPO):
    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [64, 32]  # 稍微增大网络 (因为观测维度增加)
        critic_hidden_dims = [64, 32]
        activation = "elu"
        # 使用LSTM处理序列信息
        rnn_type = "lstm"
        rnn_hidden_size = 64
        rnn_num_layers = 1

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = "ActorCriticRecurrent"
        max_iterations = 15000  # 增加训练轮数
        run_name = ""
        experiment_name = "h1_2_fullbody_obs"
