# H1_2 混合控制系统

## 📋 目录

- [概述](#概述)
- [系统架构](#系统架构)
  - [1. 训练配置](#1-训练配置)
  - [2. 轨迹类型](#2-轨迹类型)
- [🎬 演示视频](#演示视频)
  - [静态姿态演示](#静态姿态演示)
  - [动态轨迹演示](#动态轨迹演示)
- [命令行参数](#3-命令行参数)
- [运行方法](#4-运行方法)
- [技术规格](#技术规格)
- [配置文件](#配置文件)
- [开发指南](#自定义新环境)

## 概述

本系统为 H1_2 人形机器人实现了混合控制架构，将机器人 27 自由度分为两个独立控制模块：

- **下半身（12 DOF）**：采用强化学习策略网络控制，实现稳定的双足行走和平衡
- **上半身（15 DOF）**：采用预定义轨迹控制，支持多种复杂动作和姿态

## 系统架构

### 1. 训练配置

#### 12 DOF 强化学习方法
- **控制关节**：左右腿各6个关节（髋关节3个 + 膝关节1个 + 踝关节2个）
- **观测空间**：47维向量，包含：
  - 机器人基础状态（姿态、角速度等）
  - 关节位置和速度
  - 历史动作信息
  - 相位信息
- **动作空间**：12维连续动作，对应下半身关节角度偏移
- **奖励设计**：基于前进速度、姿态稳定性、能耗等多目标优化
- **训练算法**：基于 Isaac Gym 的 PPO 算法

#### 15 DOF 轨迹控制方法
- **控制关节**：躯干1个 + 左臂7个 + 右臂7个
- **控制方式**：PD控制器 + 预定义轨迹生成
- **更新频率**：50Hz（与强化学习策略同步）

### 2. 轨迹类型

#### 静态轨迹（Static Poses）
静态轨迹用于展示特定姿态，上身保持固定位置：

- **`pose_arms_forward`** - 双臂前伸姿态
  - 适用场景：展示、初始化姿态
  
- **`pose_t_shape`** - T字形张开姿态
  - 适用场景：平衡训练、展示
  
- **`pose_arms_up`** - 双臂上举姿态
  - 适用场景：庆祝动作、伸展

- **`pose_left_down_right_forward`** - 左下右前姿态
  - 适用场景：不对称动作展示

- **`pose_left_down_right_side`** - 左下右侧姿态
  - 适用场景：复杂姿态演示

- **`pose_torso_side_twist`** - 躯干侧扭姿态
  - 适用场景：躯干灵活性展示

#### 动态轨迹（Dynamic Trajectories）

**双臂动作轨迹**
- **`2arms_circles`** - 双臂同步圆周摆动
  - 参数：幅度1.2弧度，频率0.8Hz
  - 特点：双臂同时前后摆动，保持同步
  
- **`2arms_waving`** - 双臂挥手动作
  - 参数：幅度0.5弧度，频率2.0Hz
  - 特点：双臂左右挥手，适合问候场景

**单臂动作轨迹**  
- **`1arm_circles`** - 单臂圆周摆动
  - 特点：仅左臂摆动，右臂保持静止
  - 优势：节能，适合单侧交互
  
- **`1arm_waving`** - 单臂挥手动作
  - 特点：仅左臂挥手，右臂固定
  - 适用：单手问候、指示动作

**复杂动作轨迹**
- **`taichi`** - 太极推手动作
  - 特点：缓慢流畅的推拉动作
  - 适用：展示、康复训练
  
- **`boxing`** - 拳击动作
  - 特点：快速交替出拳动作
  - 适用：运动演示、力量展示
  
- **`random`** - 随机运动
  - 特点：随机变化的运动模式
  - 适用：测试、随机演示

## 🎬 演示视频

> **🎯 快速预览**: 本节展示所有10种轨迹的实际运行效果，每个轨迹在不同运动命令下的表现，帮助您直观了解系统功能。

本节展示所有支持的轨迹类型在 H1_2 机器人上的运行效果。每个轨迹都展示了在不同运动命令（前进、转向、横移）下的表现，充分体现了下半身强化学习控制与上半身轨迹控制的协调配合。

> **💡 提示**: 视频文件位于 `demo/` 目录。如需批量查看，可运行 `./demo_new_trajectories.sh` 脚本。

### 静态姿态演示

#### 双臂前伸姿态 (pose_arms_forward)
<table>
<tr>
<td width="33%" align="center">
<strong>静止站立</strong><br>
<video width="100%" controls>
  <source src="demo/pose_arms_forward_stand.mp4" type="video/mp4">
  您的浏览器不支持视频标签。
</video>
<em>命令: [0.0, 0.0, 0.0] - 原地保持平衡</em>
</td>
<td width="33%" align="center">
<strong>前进行走</strong><br>
<video width="100%" controls>
  <source src="demo/pose_arms_forward_walk.mp4" type="video/mp4">
  您的浏览器不支持视频标签。
</video>
<em>命令: [1.0, 0.0, 0.0] - 前进1m/s</em>
</td>
<td width="33%" align="center">
<strong>转向行走</strong><br>
<video width="100%" controls>
  <source src="demo/pose_arms_forward_turn.mp4" type="video/mp4">
  您的浏览器不支持视频标签。
</video>
<em>命令: [0.5, 0.0, 0.5] - 前进+转向</em>
</td>
</tr>
</table>

#### T字形张开姿态 (pose_t_shape)
<table>
<tr>
<td width="33%" align="center">
<strong>静止站立</strong><br>
<video width="100%" controls>
  <source src="demo/pose_t_shape_stand.mp4" type="video/mp4">
  您的浏览器不支持视频标签。
</video>
<em>命令: [0.0, 0.0, 0.0] - 原地保持平衡</em>
</td>
<td width="33%" align="center">
<strong>前进行走</strong><br>
<video width="100%" controls>
  <source src="demo/pose_t_shape_walk.mp4" type="video/mp4">
  您的浏览器不支持视频标签。
</video>
<em>命令: [1.0, 0.0, 0.0] - 前进1m/s</em>
</td>
<td width="33%" align="center">
<strong>侧向行走</strong><br>
<video width="100%" controls>
  <source src="demo/pose_t_shape_side.mp4" type="video/mp4">
  您的浏览器不支持视频标签。
</video>
<em>命令: [0.0, 0.8, 0.0] - 侧向0.8m/s</em>
</td>
</tr>
</table>

#### 双臂上举姿态 (pose_arms_up)
<table>
<tr>
<td width="33%" align="center">
<strong>静止站立</strong><br>
<video width="100%" controls>
  <source src="demo/pose_arms_up_stand.mp4" type="video/mp4">
  您的浏览器不支持视频标签。
</video>
<em>命令: [0.0, 0.0, 0.0] - 原地保持平衡</em>
</td>
<td width="33%" align="center">
<strong>前进行走</strong><br>
<video width="100%" controls>
  <source src="demo/pose_arms_up_walk.mp4" type="video/mp4">
  您的浏览器不支持视频标签。
</video>
<em>命令: [1.0, 0.0, 0.0] - 前进1m/s</em>
</td>
<td width="33%" align="center">
<strong>快速转向</strong><br>
<video width="100%" controls>
  <source src="demo/pose_arms_up_turn.mp4" type="video/mp4">
  您的浏览器不支持视频标签。
</video>
<em>命令: [0.0, 0.0, 1.0] - 原地转向1rad/s</em>
</td>
</tr>
</table>

#### 左下右前姿态 (pose_left_down_right_forward)
<table>
<tr>
<td width="33%" align="center">
<strong>静止站立</strong><br>
<video width="100%" controls>
  <source src="demo/pose_left_down_right_forward_stand.mp4" type="video/mp4">
  您的浏览器不支持视频标签。
</video>
<em>命令: [0.0, 0.0, 0.0] - 原地保持平衡</em>
</td>
<td width="33%" align="center">
<strong>前进行走</strong><br>
<video width="100%" controls>
  <source src="demo/pose_left_down_right_forward_walk.mp4" type="video/mp4">
  您的浏览器不支持视频标签。
</video>
<em>命令: [0.8, 0.0, 0.0] - 前进0.8m/s</em>
</td>
<td width="33%" align="center">
<strong>复合运动</strong><br>
<video width="100%" controls>
  <source src="demo/pose_left_down_right_forward_complex.mp4" type="video/mp4">
  您的浏览器不支持视频标签。
</video>
<em>命令: [0.6, 0.3, 0.3] - 前进+侧移+转向</em>
</td>
</tr>
</table>

#### 左下右侧姿态 (pose_left_down_right_side)
<table>
<tr>
<td width="33%" align="center">
<strong>静止站立</strong><br>
<video width="100%" controls>
  <source src="demo/pose_left_down_right_side_stand.mp4" type="video/mp4">
  您的浏览器不支持视频标签。
</video>
<em>命令: [0.0, 0.0, 0.0] - 原地保持平衡</em>
</td>
<td width="33%" align="center">
<strong>侧向行走</strong><br>
<video width="100%" controls>
  <source src="demo/pose_left_down_right_side_side.mp4" type="video/mp4">
  您的浏览器不支持视频标签。
</video>
<em>命令: [0.0, 0.6, 0.0] - 侧向0.6m/s</em>
</td>
<td width="33%" align="center">
<strong>前进行走</strong><br>
<video width="100%" controls>
  <source src="demo/pose_left_down_right_side_walk.mp4" type="video/mp4">
  您的浏览器不支持视频标签。
</video>
<em>命令: [0.8, 0.0, 0.0] - 前进0.8m/s</em>
</td>
</tr>
</table>

#### 躯干侧扭姿态 (pose_torso_side_twist)
<table>
<tr>
<td width="33%" align="center">
<strong>静止站立</strong><br>
<video width="100%" controls>
  <source src="demo/pose_torso_side_twist_stand.mp4" type="video/mp4">
  您的浏览器不支持视频标签。
</video>
<em>命令: [0.0, 0.0, 0.0] - 原地保持平衡</em>
</td>
<td width="33%" align="center">
<strong>前进行走</strong><br>
<video width="100%" controls>
  <source src="demo/pose_torso_side_twist_walk.mp4" type="video/mp4">
  您的浏览器不支持视频标签。
</video>
<em>命令: [0.7, 0.0, 0.0] - 前进0.7m/s</em>
</td>
<td width="33%" align="center">
<strong>旋转行走</strong><br>
<video width="100%" controls>
  <source src="demo/pose_torso_side_twist_turn.mp4" type="video/mp4">
  您的浏览器不支持视频标签。
</video>
<em>命令: [0.3, 0.0, 0.8] - 慢进+快转</em>
</td>
</tr>
</table>

### 动态轨迹演示

#### 双臂圆周摆动 (2arms_circles)
<table>
<tr>
<td width="33%" align="center">
<strong>静止摆动</strong><br>
<video width="100%" controls>
  <source src="demo/2arms_circles_stand.mp4" type="video/mp4">
  您的浏览器不支持视频标签。
</video>
<em>命令: [0.0, 0.0, 0.0] - 原地双臂摆动</em>
</td>
<td width="33%" align="center">
<strong>行走摆动</strong><br>
<video width="100%" controls>
  <source src="demo/2arms_circles_walk.mp4" type="video/mp4">
  您的浏览器不支持视频标签。
</video>
<em>命令: [1.0, 0.0, 0.0] - 前进+双臂摆动</em>
</td>
<td width="33%" align="center">
<strong>转向摆动</strong><br>
<video width="100%" controls>
  <source src="demo/2arms_circles_turn.mp4" type="video/mp4">
  您的浏览器不支持视频标签。
</video>
<em>命令: [0.5, 0.0, 0.6] - 转向+双臂摆动</em>
</td>
</tr>
</table>

#### 双臂挥手动作 (2arms_waving)
<table>
<tr>
<td width="33%" align="center">
<strong>静止挥手</strong><br>
<video width="100%" controls>
  <source src="demo/2arms_waving_stand.mp4" type="video/mp4">
  您的浏览器不支持视频标签。
</video>
<em>命令: [0.0, 0.0, 0.0] - 原地双臂挥手</em>
</td>
<td width="33%" align="center">
<strong>行走挥手</strong><br>
<video width="100%" controls>
  <source src="demo/2arms_waving_walk.mp4" type="video/mp4">
  您的浏览器不支持视频标签。
</video>
<em>命令: [0.8, 0.0, 0.0] - 前进+双臂挥手</em>
</td>
<td width="33%" align="center">
<strong>侧移挥手</strong><br>
<video width="100%" controls>
  <source src="demo/2arms_waving_turn.mp4" type="video/mp4">
  您的浏览器不支持视频标签。
</video>
<em>命令: [0.0, 0.7, 0.0] - 侧移+双臂挥手</em>
</td>
</tr>
</table>

#### 单臂圆周摆动 (1arm_circles)
<table>
<tr>
<td width="33%" align="center">
<strong>静止摆动</strong><br>
<video width="100%" controls>
  <source src="demo/1arm_circles_stand.mp4" type="video/mp4">
  您的浏览器不支持视频标签。
</video>
<em>命令: [0.0, 0.0, 0.0] - 原地单臂摆动</em>
</td>
<td width="33%" align="center">
<strong>行走摆动</strong><br>
<video width="100%" controls>
  <source src="demo/1arm_circles_walk.mp4" type="video/mp4">
  您的浏览器不支持视频标签。
</video>
<em>命令: [1.0, 0.0, 0.0] - 前进+单臂摆动</em>
</td>
<td width="33%" align="center">
<strong>复合摆动</strong><br>
<video width="100%" controls>
  <source src="demo/1arm_circles_turn.mp4" type="video/mp4">
  您的浏览器不支持视频标签。
</video>
<em>命令: [0.6, 0.4, 0.2] - 复合运动+摆动</em>
</td>
</tr>
</table>

#### 单臂挥手动作 (1arm_waving)
<table>
<tr>
<td width="33%" align="center">
<strong>静止挥手</strong><br>
<video width="100%" controls>
  <source src="demo/1arm_waving_stand.mp4" type="video/mp4">
  您的浏览器不支持视频标签。
</video>
<em>命令: [0.0, 0.0, 0.0] - 原地单臂挥手</em>
</td>
<td width="33%" align="center">
<strong>行走挥手</strong><br>
<video width="100%" controls>
  <source src="demo/1arm_waving_walk.mp4" type="video/mp4">
  您的浏览器不支持视频标签。
</video>
<em>命令: [0.9, 0.0, 0.0] - 前进+单臂挥手</em>
</td>
<td width="33%" align="center">
<strong>转向挥手</strong><br>
<video width="100%" controls>
  <source src="demo/1arm_waving_turn.mp4" type="video/mp4">
  您的浏览器不支持视频标签。
</video>
<em>命令: [0.0, 0.0, 0.8] - 转向+单臂挥手</em>
</td>
</tr>
</table>

### 复杂运动演示

#### 太极推手动作 (taichi)
<table>
<tr>
<td width="33%" align="center">
<strong>静止太极</strong><br>
<video width="100%" controls>
  <source src="demo/taichi_stand.mp4" type="video/mp4">
  您的浏览器不支持视频标签。
</video>
<em>命令: [0.0, 0.0, 0.0] - 原地太极推手</em>
</td>
<td width="33%" align="center">
<strong>慢步太极</strong><br>
<video width="100%" controls>
  <source src="demo/taichi_walk.mp4" type="video/mp4">
  您的浏览器不支持视频标签。
</video>
<em>命令: [0.3, 0.0, 0.0] - 慢步+太极</em>
</td>
<td width="33%" align="center">
<strong>转身太极</strong><br>
<video width="100%" controls>
  <source src="demo/taichi_turn.mp4" type="video/mp4">
  您的浏览器不支持视频标签。
</video>
<em>命令: [0.2, 0.0, 0.4] - 慢行+转身太极</em>
</td>
</tr>
</table>

#### 拳击动作 (boxing)
<table>
<tr>
<td width="33%" align="center">
<strong>静止拳击</strong><br>
<video width="100%" controls>
  <source src="demo/boxing_stand.mp4" type="video/mp4">
  您的浏览器不支持视频标签。
</video>
<em>命令: [0.0, 0.0, 0.0] - 原地拳击</em>
</td>
<td width="33%" align="center">
<strong>前进拳击</strong><br>
<video width="100%" controls>
  <source src="demo/boxing_walk.mp4" type="video/mp4">
  您的浏览器不支持视频标签。
</video>
<em>命令: [0.6, 0.0, 0.0] - 前进+拳击</em>
</td>
<td width="33%" align="center">
<strong>机动拳击</strong><br>
<video width="100%" controls>
  <source src="demo/boxing_turn.mp4" type="video/mp4">
  您的浏览器不支持视频标签。
</video>
<em>命令: [0.4, 0.3, 0.5] - 机动+拳击</em>
</td>
</tr>
</table>

#### 随机运动 (random)
<table>
<tr>
<td width="33%" align="center">
<strong>静止随机</strong><br>
<video width="100%" controls>
  <source src="demo/random_stand.mp4" type="video/mp4">
  您的浏览器不支持视频标签。
</video>
<em>命令: [0.0, 0.0, 0.0] - 原地随机动作</em>
</td>
<td width="33%" align="center">
<strong>行走随机</strong><br>
<video width="100%" controls>
  <source src="demo/random_walk.mp4" type="video/mp4">
  您的浏览器不支持视频标签。
</video>
<em>命令: [0.8, 0.0, 0.0] - 前进+随机动作</em>
</td>
<td width="33%" align="center">
<strong>全向随机</strong><br>
<video width="100%" controls>
  <source src="demo/random_turn.mp4" type="video/mp4">
  您的浏览器不支持视频标签。
</video>
<em>命令: [0.5, 0.5, 0.3] - 全向+随机动作</em>
</td>
</tr>
</table>

### 运动命令说明

#### 命令格式
```python
cmd = [linear_x, linear_y, angular_z]
```

- **linear_x**: 前进/后退速度 (m/s)，正值前进，负值后退
- **linear_y**: 左右侧移速度 (m/s)，正值向左，负值向右  
- **angular_z**: 旋转角速度 (rad/s)，正值逆时针，负值顺时针

#### 典型命令示例
- `[0.0, 0.0, 0.0]` - 原地站立保持平衡
- `[1.0, 0.0, 0.0]` - 以1m/s前进
- `[0.0, 0.8, 0.0]` - 以0.8m/s向左侧移
- `[0.0, 0.0, 1.0]` - 以1rad/s逆时针转向
- `[0.6, 0.3, 0.2]` - 复合运动：前进+侧移+转向

### 视频技术规格

- **分辨率**: 1920×1080 (Full HD)
- **帧率**: 30 FPS
- **格式**: MP4 (H.264 编码)
- **时长**: 每个演示约 15-20 秒
- **视角**: 侧视角，便于观察上下身协调
- **内容**: 每个视频展示2-3个完整的动作循环

### 3. 命令行参数

#### 基本参数
```bash
python deploy_mujoco3.py <config_file> [options]
```

**必需参数：**
- `config_file`：配置文件名（位于 configs/ 目录）

**可选参数：**
- `--trajectory, -t`：指定上身轨迹类型
  - 默认值：`2arms_circles`
  - 可选值：见轨迹类型章节

#### 轨迹参数选择

| 参数值 | 类型 | 描述 |
|--------|------|------|
| `2arms_circles` | 动态 | 双臂圆周摆动 |
| `2arms_waving` | 动态 | 双臂挥手 |
| `1arm_circles` | 动态 | 单臂圆周摆动 |
| `1arm_waving` | 动态 | 单臂挥手 |
| `taichi` | 动态 | 太极推手 |
| `boxing` | 动态 | 拳击动作 |
| `random` | 动态 | 随机运动 |
| `pose_arms_forward` | 静态 | 双臂前伸 |
| `pose_t_shape` | 静态 | T字形张开 |
| `pose_arms_up` | 静态 | 双臂上举 |
| `pose_left_down_right_forward` | 静态 | 左下右前 |
| `pose_left_down_right_side` | 静态 | 左下右侧 |
| `pose_torso_side_twist` | 静态 | 躯干扭转 |

### 4. 运行方法

#### 快速启动
```bash
# 进入部署目录
cd deploy/deploy_mujoco

# 使用默认轨迹运行
./run_hybrid_deploy.sh

# 指定特定轨迹运行
./run_hybrid_deploy.sh 2arms_circles
./run_hybrid_deploy.sh boxing
./run_hybrid_deploy.sh pose_t_shape
```

#### 详细命令
```bash
# 双臂动作
python deploy_mujoco3.py h1_2_hybrid.yaml --trajectory 2arms_circles
python deploy_mujoco3.py h1_2_hybrid.yaml --trajectory 2arms_waving

# 单臂动作
python deploy_mujoco3.py h1_2_hybrid.yaml --trajectory 1arm_circles
python deploy_mujoco3.py h1_2_hybrid.yaml --trajectory 1arm_waving

# 复杂动作
python deploy_mujoco3.py h1_2_hybrid.yaml --trajectory taichi
python deploy_mujoco3.py h1_2_hybrid.yaml --trajectory boxing

# 静态姿态
python deploy_mujoco3.py h1_2_hybrid.yaml --trajectory pose_arms_forward
python deploy_mujoco3.py h1_2_hybrid.yaml --trajectory pose_t_shape
```

#### 批量演示
```bash
# 演示所有轨迹（每个10秒）
./demo_new_trajectories.sh

# 演示特定类型（自定义脚本）
./demo_dynamic_trajectories.sh    # 仅动态轨迹
./demo_static_poses.sh            # 仅静态姿态
```

## 技术规格

### 控制参数

#### PD控制器增益
```yaml
# 下半身（12 DOF）
lower_body_kps: [200, 200, 200, 300, 40, 40, 200, 200, 200, 300, 40, 40]
lower_body_kds: [2.5, 2.5, 2.5, 4, 2, 2, 2.5, 2.5, 2.5, 4, 2, 2]

# 上半身（15 DOF）
upper_body_kps: [50, 80, 80, 50, 50, 20, 20, 20, 80, 80, 50, 50, 20, 20, 20]
upper_body_kds: [0, 2.0, 2.0, 1.5, 1.5, 1.0, 1.0, 1.0, 2.0, 2.0, 1.5, 1.5, 1.0, 1.0, 1.0]
```

#### 系统频率
- **仿真频率**：500Hz（dt=0.002s）
- **控制频率**：50Hz（decimation=10）
- **策略更新**：50Hz（与控制同步）

#### 关节映射
```
下半身关节（12 DOF）：
├── 左腿（6 DOF）：hip_yaw, hip_roll, hip_pitch, knee, ankle_pitch, ankle_roll
└── 右腿（6 DOF）：hip_yaw, hip_roll, hip_pitch, knee, ankle_pitch, ankle_roll

上半身关节（15 DOF）：
├── 躯干（1 DOF）：torso_joint
├── 左臂（7 DOF）：shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_yaw, wrist_roll, wrist_pitch
└── 右臂（7 DOF）：shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_yaw, wrist_roll, wrist_pitch
```

## 配置文件

### 主配置文件 (h1_2_hybrid.yaml)

#### 基本设置
```yaml
# 策略和模型路径
policy_path: "{LEGGED_GYM_ROOT_DIR}/logs/h1_2/exported/policies/policy_lstm_1.pt"
xml_path: "{LEGGED_GYM_ROOT_DIR}/resources/robots/h1_2/scene.xml"

# 仿真参数
simulation_duration: 60.0      # 仿真总时长（秒）
simulation_dt: 0.002          # 仿真时间步长
control_decimation: 10        # 控制频率分频系数
```

#### 策略参数
```yaml
# 观测和动作缩放
ang_vel_scale: 0.25           # 角速度观测缩放
dof_pos_scale: 1.0           # 关节位置观测缩放
dof_vel_scale: 0.05          # 关节速度观测缩放
action_scale: 0.25           # 动作输出缩放
cmd_scale: [2.0, 2.0, 0.25]  # 命令速度缩放

# 网络结构
num_actions: 12              # 策略输出动作数
num_obs: 47                  # 观测向量维度
```

## 自定义新环境😆

### 添加新轨迹

1. **定义轨迹函数**
```python
def trajectory_custom_motion(time_sim, config):
    """自定义轨迹函数"""
    upper_body_targets = np.zeros(15)
    
    # 实现轨迹逻辑
    # upper_body_targets[0] = ...    # 躯干
    # upper_body_targets[1:8] = ...  # 左臂
    # upper_body_targets[8:15] = ... # 右臂
    
    return upper_body_targets
```

2. **注册轨迹函数**
```python
trajectory_functions = {
    # ...existing trajectories...
    "custom_motion": trajectory_custom_motion,
}
```

3. **更新命令行参数**
```python
parser.add_argument(
    "--trajectory",
    choices=[
        # ...existing choices...
        "custom_motion",
    ]
)
```


### 调试和监控

#### 实时信息显示
```
Time: 1.20s
Lower body targets: [-0.01 -0.16  0.01  0.36 -0.20  0.01]
Upper body targets: [ 0.82  0.30  0.00  1.50]
```


