# H1_2 混合控制部署说明

## 概述
这个项目实现了H1_2人形机器人的混合控制方案：
- **下半身（12 DOF）**：使用训练好的强化学习策略控制
- **上半身（15 DOF）**：使用预定义轨迹控制

## 文件结构

### 配置文件
- `configs/h1_2_hybrid.yaml`: 混合控制的配置文件
- `configs/h1_2.yaml`: 原始的下半身控制配置文件

### 部署脚本
- `deploy_mujoco3.py`: 新的混合控制部署脚本
- `deploy_mujoco2.py`: 原始的下半身控制脚本
- `run_hybrid_deploy.sh`: 运行脚本

### 机器人模型
- `scene.xml`: 场景配置文件（已修改为使用27DOF模型）
- `h1_2_27dof.xml`: 完整的27DOF机器人模型
- `h1_2_12dof.xml`: 原始的12DOF机器人模型

## 关节映射

### 下半身关节（12 DOF）- 策略控制
```
索引 0-5:  左腿
  0: left_hip_yaw_joint
  1: left_hip_pitch_joint  
  2: left_hip_roll_joint
  3: left_knee_joint
  4: left_ankle_pitch_joint
  5: left_ankle_roll_joint

索引 6-11: 右腿
  6: right_hip_yaw_joint
  7: right_hip_pitch_joint
  8: right_hip_roll_joint  
  9: right_knee_joint
  10: right_ankle_pitch_joint
  11: right_ankle_roll_joint
```

### 上半身关节（15 DOF）- 轨迹控制
```
索引 12:    躯干
  12: torso_joint

索引 13-19: 左臂
  13: left_shoulder_pitch_joint
  14: left_shoulder_roll_joint
  15: left_shoulder_yaw_joint
  16: left_elbow_pitch_joint
  17: left_elbow_roll_joint
  18: left_wrist_pitch_joint
  19: left_wrist_yaw_joint

索引 20-26: 右臂
  20: right_shoulder_pitch_joint
  21: right_shoulder_roll_joint
  22: right_shoulder_yaw_joint
  23: right_elbow_pitch_joint
  24: right_elbow_roll_joint
  25: right_wrist_pitch_joint
  26: right_wrist_yaw_joint
```

## 轨迹设计

### 当前实现的轨迹
1. **双臂圆周运动**：
   - 肩关节pitch做圆周运动，模拟前伸并画圆
   - 肩关节roll做对称的左右摆动
   - 肘关节pitch保持弯曲状态并有轻微变化

2. **躯干前弯运动**：
   - 躯干关节做正弦波前弯运动

### 轨迹参数（在配置文件中可调）
- `trajectory_amplitude`: 手臂圆周运动幅度 (0.5 rad)
- `trajectory_frequency`: 圆周运动频率 (0.5 Hz)
- `torso_bend_amplitude`: 躯干弯曲幅度 (0.2 rad)
- `torso_bend_frequency`: 躯干弯曲频率 (0.3 Hz)
- `arm_forward_offset`: 手臂前伸基础角度 (0.5 rad)

## 运行方法

### 方法1：使用便捷脚本
```bash
cd /home/yuxin/unitree_rl_gym/deploy/deploy_mujoco

# 使用默认轨迹（圆周运动）
./run_hybrid_deploy.sh

# 使用指定轨迹
./run_hybrid_deploy.sh waving     # 挥手打招呼
./run_hybrid_deploy.sh taichi     # 太极推手
./run_hybrid_deploy.sh boxing     # 拳击动作
./run_hybrid_deploy.sh dancing    # 舞蹈动作
./run_hybrid_deploy.sh stretching # 拉伸动作
```

### 方法2：直接运行Python
```bash
cd /home/yuxin/unitree_rl_gym/deploy/deploy_mujoco

# 使用默认轨迹
python deploy_mujoco3.py h1_2_hybrid.yaml

# 使用指定轨迹
python deploy_mujoco3.py h1_2_hybrid.yaml --trajectory waving
python deploy_mujoco3.py h1_2_hybrid.yaml -t taichi
```

### 方法3：轨迹演示模式
```bash
cd /home/yuxin/unitree_rl_gym/deploy/deploy_mujoco

# 依次演示所有轨迹（每个10秒）
./demo_all_trajectories.sh
```

## 可用轨迹类型

### 1. `circles` - 双臂圆周运动（默认）
- 手臂做圆周运动，配合躯干前弯
- 参数可在配置文件中调节

### 2. `waving` - 挥手打招呼
- 右臂抬起挥手，左臂自然下垂
- 适合测试单臂动作

### 3. `taichi` - 太极推手
- 缓慢的推拉动作，左右臂交替
- 躯干轻微转动配合

### 4. `boxing` - 拳击动作
- 交替出拳动作
- 躯干前倾的拳击姿势

### 5. `dancing` - 舞蹈动作
- 双臂协调摆动
- 复杂的多关节协调运动

### 6. `stretching` - 拉伸动作
- 20秒循环的多阶段拉伸
- 包含向上拉伸、侧弯、前后拉伸等

## 自定义轨迹

### 修改现有轨迹
在 `deploy_mujoco3.py` 中的 `generate_upper_body_trajectory()` 函数里修改轨迹生成逻辑。

### 轨迹示例

#### 1. 简单的挥手动作
```python
# 右臂挥手
upper_body_targets[20] = 1.5  # shoulder pitch: 前伸
upper_body_targets[21] = -0.5 + 0.5 * math.sin(phase)  # shoulder roll: 左右摆动
upper_body_targets[23] = 1.2  # elbow pitch: 弯曲
```

#### 2. 太极推手动作
```python
# 缓慢的推拉动作
slow_phase = 2 * math.pi * 0.2 * time_sim  # 0.2 Hz
upper_body_targets[13] = 0.8 + 0.4 * math.sin(slow_phase)  # 左臂前后推拉
upper_body_targets[20] = 0.8 + 0.4 * math.sin(slow_phase + math.pi)  # 右臂相位相反
```

#### 3. 举重动作
```python
# 双臂举重
lift_phase = 2 * math.pi * 0.3 * time_sim
lift_height = 0.5 * (1 + math.sin(lift_phase))  # 0到1之间变化
upper_body_targets[13] = -0.5 + lift_height * 2.0  # 左肩
upper_body_targets[20] = -0.5 + lift_height * 2.0  # 右肩
upper_body_targets[16] = 1.5 - lift_height * 1.0   # 左肘
upper_body_targets[23] = 1.5 - lift_height * 1.0   # 右肘
```

## 技术细节

### PD控制器参数
- 下半身：使用原始策略的PD参数（较高增益）
- 上半身：使用较温和的PD参数（避免过度振荡）

### 观察空间
- 保持与原始策略一致的47维观察空间
- 仅包含下半身的关节信息
- 包含IMU、重力方向、指令等信息

### 控制频率
- 物理仿真：500Hz (dt=0.002)
- 控制更新：50Hz (decimation=10)
- 与训练时的频率保持一致

## 故障排除

### 常见问题
1. **机器人倒下**：检查PD增益是否合适，特别是下半身的增益
2. **上半身抖动**：降低上半身的PD增益
3. **动作不协调**：调整轨迹的幅度和频率参数
4. **策略加载失败**：检查策略文件路径是否正确

### 调试输出
脚本会每10个控制周期（0.2秒）输出一次调试信息，包括：
- 当前时间
- 下半身目标关节角度
- 上半身目标关节角度

## 进一步开发

### 可能的改进方向
1. **更复杂的轨迹**：添加基于状态的轨迹规划
2. **交互控制**：添加键盘或手柄输入来实时改变轨迹
3. **力控制**：在某些关节上使用力控制而非位置控制
4. **自适应轨迹**：根据下半身的运动状态调整上半身轨迹
5. **多模态切换**：不同的轨迹模式之间的平滑切换
