# H1_2 混合控制系统 - 新轨迹版本

## 概述

本系统为 H1_2 人形机器人实现了混合控制方案：
- **下半身（12 DOF）**：由强化学习 policy 控制，实现稳定行走
- **上半身（15 DOF）**：由预定义轨迹控制，支持多种动作模式

## 新轨迹功能

### 1. 双臂动作轨迹
- **`2arm_circles`** - 双臂同步圆周摆动
  - 双臂同时进行前后摆动
  - 可调节摆动幅度和频率
  
- **`waving_2arm`** - 双臂挥手
  - 双臂同步左右挥手动作
  - 适合打招呼场景

### 2. 单臂动作轨迹  
- **`1arm_circles`** - 单臂圆周摆动
  - 仅左臂进行摆动，右臂保持静止
  - 节能的单侧运动模式
  
- **`waving_1arm`** - 单臂挥手
  - 仅左臂挥手，右臂保持固定
  - 适合单侧交互

### 3. 复杂动作轨迹
- **`taichi`** - 太极推手动作
- **`boxing`** - 拳击动作 
- **`dancing`** - 舞蹈动作
- **`stretching`** - 拉伸动作
- **`random`** - 随机运动

### 4. 静止姿态
- **`pose_arms_forward`** - 双臂前伸
- **`pose_t_shape`** - T字形张开
- **`pose_arms_up`** - 双臂上举
- **`pose_left_down_right_forward`** - 左下右前
- **`pose_left_down_right_side`** - 左下右侧
- **`pose_torso_side_twist`** - 躯干扭转

## 使用方法

### 基本运行命令
```bash
# 使用默认轨迹（双臂圆周）
./run_hybrid_deploy.sh

# 指定特定轨迹
./run_hybrid_deploy.sh 2arm_circles
./run_hybrid_deploy.sh 1arm_circles
./run_hybrid_deploy.sh waving_2arm
./run_hybrid_deploy.sh taichi
```

### 直接Python命令
```bash
python deploy_mujoco3.py h1_2_hybrid.yaml --trajectory 2arm_circles
python deploy_mujoco3.py h1_2_hybrid.yaml --trajectory 1arm_circles
python deploy_mujoco3.py h1_2_hybrid.yaml --trajectory waving_2arm
```

### 演示所有轨迹
```bash
./demo_new_trajectories.sh
```

## 轨迹参数

新的轨迹函数采用简化的参数设计：

### 双臂/单臂圆周运动参数
```python
# 在轨迹函数中硬编码，可根据需要调整
traj_amp = 1.2      # 摆动幅度 (弧度)
traj_freq = 0.8     # 摆动频率 (Hz)  
arm_offset = 0.0    # 中心位置 (弧度)
```

### 挥手动作参数
```python
wave_amp = 0.5      # 挥手幅度 (弧度)
wave_freq = 2.0     # 挥手频率 (Hz)
wave_offset = 0.3   # 基础偏移 (弧度)
```

## 轨迹特点对比

| 轨迹类型 | 双臂/单臂 | 运动类型 | 适用场景 |
|---------|----------|----------|----------|
| 2arm_circles | 双臂 | 圆周摆动 | 基础演示、热身 |
| 1arm_circles | 单臂 | 圆周摆动 | 节能模式、单侧交互 |
| waving_2arm | 双臂 | 挥手 | 双手问候、表达 |
| waving_1arm | 单臂 | 挥手 | 单手问候 |
| taichi | 双臂 | 慢速推拉 | 展示、康复训练 |
| boxing | 双臂 | 快速出拳 | 运动演示 |
| dancing | 双臂 | 协调摆动 | 娱乐表演 |
| stretching | 双臂 | 拉伸 | 维护关节活动度 |
| random | 双臂 | 随机变化 | 测试、随机演示 |

## 配置文件

轨迹参数在 `configs/h1_2_hybrid.yaml` 中配置：

```yaml
# 上身轨迹参数 (已废弃，现在硬编码在轨迹函数中)
trajectory_amplitude: 0.5
trajectory_frequency: 0.5
torso_bend_amplitude: 0.2
torso_bend_frequency: 0.3
arm_forward_offset: -2.0
```

## 开发和自定义

### 添加新轨迹
1. 在 `deploy_mujoco3.py` 中添加新的轨迹函数：
```python
def trajectory_my_custom(time_sim, config):
    upper_body_targets = np.zeros(15)
    # 实现您的轨迹逻辑
    return upper_body_targets
```

2. 在 `trajectory_functions` 字典中注册：
```python
trajectory_functions = {
    # ...existing trajectories...
    "my_custom": trajectory_my_custom,
}
```

3. 更新命令行选择列表和文档

### 轨迹函数说明
- **输入**: `time_sim` (仿真时间), `config` (配置字典)
- **输出**: `upper_body_targets` (15个上身关节目标角度)
- **关节顺序**: [躯干] + [左臂7个] + [右臂7个]

## 故障排除

### 常见问题
1. **轨迹不平滑**: 检查函数中的数学公式和相位计算
2. **机器人不稳定**: 调整轨迹幅度，避免过大的快速运动
3. **关节超限**: 确保关节角度在安全范围内

### 调试信息
运行时会显示详细的调试信息：
```
Time: 1.20s
Lower body targets: [-0.01 -0.16  0.01  0.36 -0.20  0.01]
Upper body targets: [ 0.82  0.30  0.00  1.50]
```

## 更新日志

**v2.0 - 新轨迹版本**
- 新增双臂和单臂独立控制轨迹
- 简化轨迹参数配置
- 优化轨迹函数实现
- 更新文档和演示脚本

**v1.0 - 基础版本** 
- 实现基础混合控制系统
- 支持基本轨迹和静止姿态
