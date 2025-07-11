# H1_2混合控制快速参考

## 基本使用

```bash
# 基本运行
python deploy_mujoco3.py h1_2_hybrid.yaml

# 指定轨迹类型
python deploy_mujoco3.py h1_2_hybrid.yaml --trajectory circles
```

## 可用轨迹类型

### 动态轨迹
| 轨迹名称 | 描述 | 特点 |
|---------|------|------|
| `circles` | 双臂圆周运动 | 连续圆形轨迹 |
| `waving` | 挥手动作 | 友好招手手势 |
| `taichi` | 太极动作 | 缓慢流畅运动 |
| `boxing` | 拳击动作 | 快速打击动作 |
| `dancing` | 舞蹈动作 | 节奏性手臂摆动 |
| `stretching` | 拉伸动作 | 关节伸展运动 |
| `random` | 随机动作 | 关节随机小幅变化 |

### 静止上身动作
| 动作名称 | 描述 | 姿态特点 |
|---------|------|---------|
| `pose_arms_forward` | 双臂前伸 | 精确前伸控制 |
| `pose_left_down_right_forward` | 左下右前 | 优化不对称 |
| `pose_t_shape` | T形姿态 | 完美十字形 |
| `pose_left_down_right_side` | 左下右侧 | 稳定单侧展开 |
| `pose_torso_side_twist` | 躯干侧扭 | 明显扭转效果 |
| `pose_left_up_right_down` | 左上右下 | 自然上举动作 |

## 快速示例

```bash
# 动态轨迹
python deploy_mujoco3.py h1_2_hybrid.yaml -t circles
python deploy_mujoco3.py h1_2_hybrid.yaml -t taichi
python deploy_mujoco3.py h1_2_hybrid.yaml -t dancing

# 静止动作
python deploy_mujoco3.py h1_2_hybrid.yaml -t pose_t_shape
python deploy_mujoco3.py h1_2_hybrid.yaml -t pose_arms_forward
python deploy_mujoco3.py h1_2_hybrid.yaml -t pose_left_up_right_down

# 演示所有轨迹
./demo_complete.sh
```

## 配置调节

### 轨迹参数
在 `h1_2_hybrid.yaml` 中调节：
```yaml
upper_body:
  trajectory:
    amplitude: 0.8        # 动作幅度
    frequency: 0.5        # 动作频率
    phase_offset: 0.0     # 相位偏移
    damping: 0.1          # 阻尼系数
```

### 控制权重
```yaml
control:
  lower_body_weight: 1.0  # 下半身policy权重
  upper_body_weight: 1.0  # 上半身轨迹权重
  blend_factor: 0.1       # 混合因子
```

## 常用命令

```bash
# 查看帮助
python deploy_mujoco3.py --help

# 列出所有可用轨迹
python deploy_mujoco3.py h1_2_hybrid.yaml -t invalid_name  # 会显示所有选项

# 运行特定时长
timeout 30s python deploy_mujoco3.py h1_2_hybrid.yaml -t circles

# 后台运行
nohup python deploy_mujoco3.py h1_2_hybrid.yaml -t taichi > output.log 2>&1 &
```

## 故障排除

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| 模型加载失败 | 路径错误 | 检查配置文件路径 |
| 关节超限 | 角度过大 | 降低amplitude参数 |
| 动作不自然 | 频率太高 | 降低frequency参数 |
| 平衡问题 | 重心偏移 | 调整blend_factor |

## 高级用法

### 自定义轨迹
1. 在 `deploy_mujoco3.py` 中添加新函数
2. 在 `trajectory_functions` 字典中注册
3. 在命令行choices中添加选项

### 批量测试
```bash
# 测试所有动态轨迹
for traj in circles waving taichi boxing dancing stretching; do
    echo "Testing $traj"
    timeout 5s python deploy_mujoco3.py h1_2_hybrid.yaml -t $traj
done
```

### 参数优化
```bash
# 不同幅度测试
for amp in 0.3 0.5 0.8 1.0; do
    sed -i "s/amplitude: .*/amplitude: $amp/" configs/h1_2_hybrid.yaml
    python deploy_mujoco3.py h1_2_hybrid.yaml -t circles
done
```
