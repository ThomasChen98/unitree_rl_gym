# H1_2 新轨迹快速参考

## 命令格式
```bash
./run_hybrid_deploy.sh [trajectory_name]
# 或
python deploy_mujoco3.py h1_2_hybrid.yaml --trajectory [trajectory_name]
```

## 轨迹列表

### 🤲 双臂动作
| 命令 | 描述 | 特点 |
|------|------|------|
| `2arm_circles` | 双臂圆周摆动 | 同步前后摆动，平滑连续 |
| `waving_2arm` | 双臂挥手 | 左右摆动，问候动作 |

### 🤚 单臂动作  
| 命令 | 描述 | 特点 |
|------|------|------|
| `1arm_circles` | 单臂圆周摆动 | 仅左臂摆动，节能模式 |
| `waving_1arm` | 单臂挥手 | 仅左臂挥手，右臂静止 |

### 🎭 复杂动作
| 命令 | 描述 | 频率 | 特点 |
|------|------|------|------|
| `taichi` | 太极推手 | 慢 (0.15Hz) | 左右臂交替推拉 |
| `boxing` | 拳击动作 | 中 (1.5Hz) | 交替出拳防守 |
| `dancing` | 舞蹈动作 | 中 (0.8Hz) | 协调摆动，多关节 |
| `stretching` | 拉伸动作 | 极慢 (0.1Hz) | 20秒循环，4个阶段 |
| `random` | 随机运动 | 变化 | 所有关节随机小幅变化 |

### 🧘 静止姿态
| 命令 | 描述 |
|------|------|
| `pose_arms_forward` | 双臂前伸 |
| `pose_t_shape` | T字形张开 |
| `pose_arms_up` | 双臂上举 |
| `pose_left_down_right_forward` | 左下右前 |
| `pose_left_down_right_side` | 左下右侧 |
| `pose_torso_side_twist` | 躯干前倾 |

## 快速测试命令

### 演示所有轨迹
```bash
./demo_new_trajectories.sh
```

### 测试基础轨迹
```bash
# 双臂摆动 (默认)
./run_hybrid_deploy.sh

# 单臂摆动
./run_hybrid_deploy.sh 1arm_circles

# 双臂挥手
./run_hybrid_deploy.sh waving_2arm
```

### 测试静止姿态
```bash
./run_hybrid_deploy.sh pose_t_shape
./run_hybrid_deploy.sh pose_arms_up
```

## 轨迹参数 (硬编码)

### 圆周摆动
```python
traj_amp = 1.2    # 摆动幅度
traj_freq = 0.8   # 摆动频率  
arm_offset = 0.0  # 中心位置
```

### 挥手动作
```python
wave_amp = 0.5      # 挥手幅度
wave_freq = 2.0     # 挥手频率
wave_offset = 0.3   # 基础偏移
```

## 关节映射 (上身15DOF)
```
索引 0:    躯干关节
索引 1-7:  左臂关节 (肩x3, 肘x1, 腕x3)
索引 8-14: 右臂关节 (肩x3, 肘x1, 腕x3)
```

## 文件结构
```
deploy_mujoco/
├── deploy_mujoco3.py           # 主部署脚本
├── configs/h1_2_hybrid.yaml   # 配置文件
├── run_hybrid_deploy.sh        # 运行脚本
├── demo_new_trajectories.sh    # 演示脚本
├── README_NEW_TRAJECTORIES.md  # 详细文档
└── QUICK_REFERENCE_NEW.md      # 本文件
```

---
**提示**: 首次运行建议使用 `2arm_circles` 测试基础功能
