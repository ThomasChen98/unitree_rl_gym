# H1_2机器人轨迹完整参考

本文档详细描述了H1_2机器人混合控制系统中可用的所有轨迹类型。

## 轨迹分类

### 1. 动态轨迹

#### circles - 双臂圆周运动
- **描述**: 双臂同步进行圆周运动
- **特点**: 连续流畅的圆形轨迹
- **参数**: 可配置幅度和频率

#### waving - 挥手动作
- **描述**: 右臂挥手打招呼
- **特点**: 友好的招手手势
- **频率**: 2Hz

#### taichi - 太极动作
- **描述**: 缓慢的太极推手动作
- **特点**: 左右臂交替推拉
- **频率**: 0.15Hz（很慢）

#### boxing - 拳击动作
- **描述**: 模拟拳击的出拳动作
- **特点**: 左右拳交替出击
- **频率**: 1.5Hz

#### dancing - 舞蹈动作
- **描述**: 协调的手臂舞蹈动作
- **特点**: 左右臂不同相位的协调摆动
- **频率**: 0.8Hz

#### stretching - 拉伸动作
- **描述**: 多阶段拉伸运动
- **特点**: 20秒循环，包含上举、侧弯、前后拉伸、放松
- **频率**: 0.1Hz（很慢）

#### random - 随机动作 **[新增]**
- **描述**: 每个关节进行随机小幅度变化
- **特点**: 
  - 每个关节独立的随机运动
  - 基于时间的伪随机种子
  - 安全的角度限制
- **参数**:
  - 基础幅度: 0.3弧度
  - 频率变化: 0.5弧度/秒
  - 肩膀关节幅度更大，肘关节保持正值

### 2. 静止动作

#### pose_arms_forward - 双臂前伸
- **描述**: 双臂水平向前伸展
- **角度**: 左右臂前伸 -2.0弧度

#### pose_left_down_right_forward - 左下右前
- **描述**: 左臂下垂，右臂前伸
- **角度**: 右臂前伸 -2.0弧度

#### pose_t_shape - T形姿态
- **描述**: 双臂水平张开成十字形
- **角度**: 左臂外展 1.57弧度，右臂外展 -1.57弧度（90度）

#### pose_left_down_right_side - 左下右侧
- **描述**: 左臂下垂，右臂侧向张开
- **角度**: 右臂外展 -1.57弧度

#### pose_torso_side_twist - 躯干扭转
- **描述**: 躯干向前大幅扭转
- **角度**: 躯干前倾 1.57弧度（90度）

#### pose_left_up_right_down - 左上右下
- **描述**: 左臂上举，右臂下垂
- **角度**: 左臂后伸上举 -3.0弧度

## 使用示例

### 基本调用
```bash
# 动态轨迹
python deploy_mujoco3.py h1_2_hybrid.yaml -t circles
python deploy_mujoco3.py h1_2_hybrid.yaml -t random
python deploy_mujoco3.py h1_2_hybrid.yaml -t taichi

# 静止动作
python deploy_mujoco3.py h1_2_hybrid.yaml -t pose_t_shape
python deploy_mujoco3.py h1_2_hybrid.yaml -t pose_left_up_right_down
```

### 批量测试
```bash
# 测试所有动态轨迹
for traj in circles waving taichi boxing dancing stretching random; do
    echo "Testing $traj"
    timeout 10s python deploy_mujoco3.py h1_2_hybrid.yaml -t $traj
done

# 测试所有静止动作
for pose in pose_arms_forward pose_left_down_right_forward pose_t_shape pose_left_down_right_side pose_torso_side_twist pose_left_up_right_down; do
    echo "Testing $pose"
    timeout 8s python deploy_mujoco3.py h1_2_hybrid.yaml -t $pose
done
```

## 随机动作详细说明

### 算法特点
1. **时间相关随机性**: 使用 `int(time_sim * 1000) % 10000` 作为随机种子
2. **关节独立性**: 每个关节有独立的随机因子
3. **频率变化**: 每个关节有不同的振荡频率
4. **安全限制**: 
   - 肘关节保持正值避免过度伸展
   - 限制最大角度变化范围

### 参数调节
```python
base_amplitude = 0.3    # 基础随机幅度
freq_variation = 0.5    # 频率变化范围
```

可以通过修改这些参数来调整随机动作的强度和变化速度。

## 完整轨迹列表

**动态轨迹 (7个)**: circles, waving, taichi, boxing, dancing, stretching, random
**静止动作 (6个)**: pose_arms_forward, pose_left_down_right_forward, pose_t_shape, pose_left_down_right_side, pose_torso_side_twist, pose_left_up_right_down

**总计**: 13种不同的上身轨迹类型

## 性能建议

- **实时性**: 随机动作计算量较小，适合实时运行
- **可重复性**: 相同时间点的随机值是确定的（伪随机）
- **安全性**: 所有轨迹都经过角度限制检查
- **平衡性**: 下半身policy会自动适应上身动作变化
