# H1_2 混合控制系统演示视频清单

本文件列出了需要录制和放置的所有演示视频文件。每个轨迹需要3个视频，对应不同的运动命令设定。

## 视频文件路径
所有视频文件应放置在：`demo/` 目录下

## 需要的视频文件清单 (总计30个视频)

### 静态姿态 (18个视频 = 6个轨迹 × 3个命令)

#### 双臂前伸姿态 (pose_arms_forward)
1. `pose_arms_forward_stand.mp4` - 静止站立 [0.0, 0.0, 0.0]
2. `pose_arms_forward_walk.mp4` - 前进行走 [1.0, 0.0, 0.0]
3. `pose_arms_forward_turn.mp4` - 转向行走 [0.5, 0.0, 0.5]

#### T字形张开姿态 (pose_t_shape)
4. `pose_t_shape_stand.mp4` - 静止站立 [0.0, 0.0, 0.0]
5. `pose_t_shape_walk.mp4` - 前进行走 [1.0, 0.0, 0.0]
6. `pose_t_shape_side.mp4` - 侧向行走 [0.0, 0.8, 0.0]

#### 双臂上举姿态 (pose_arms_up)
7. `pose_arms_up_stand.mp4` - 静止站立 [0.0, 0.0, 0.0]
8. `pose_arms_up_walk.mp4` - 前进行走 [1.0, 0.0, 0.0]
9. `pose_arms_up_turn.mp4` - 快速转向 [0.0, 0.0, 1.0]

#### 左下右前姿态 (pose_left_down_right_forward)
10. `pose_left_down_right_forward_stand.mp4` - 静止站立 [0.0, 0.0, 0.0]
11. `pose_left_down_right_forward_walk.mp4` - 前进行走 [0.8, 0.0, 0.0]
12. `pose_left_down_right_forward_complex.mp4` - 复合运动 [0.6, 0.3, 0.3]

#### 左下右侧姿态 (pose_left_down_right_side)
13. `pose_left_down_right_side_stand.mp4` - 静止站立 [0.0, 0.0, 0.0]
14. `pose_left_down_right_side_side.mp4` - 侧向行走 [0.0, 0.6, 0.0]
15. `pose_left_down_right_side_walk.mp4` - 前进行走 [0.8, 0.0, 0.0]

#### 躯干侧扭姿态 (pose_torso_side_twist)
16. `pose_torso_side_twist_stand.mp4` - 静止站立 [0.0, 0.0, 0.0]
17. `pose_torso_side_twist_walk.mp4` - 前进行走 [0.7, 0.0, 0.0]
18. `pose_torso_side_twist_rotate.mp4` - 旋转行走 [0.3, 0.0, 0.8]

### 动态轨迹 (12个视频 = 4个轨迹 × 3个命令)

#### 双臂圆周摆动 (2arms_circles)
19. `2arms_circles_stand.mp4` - 静止摆动 [0.0, 0.0, 0.0]
20. `2arms_circles_walk.mp4` - 行走摆动 [1.0, 0.0, 0.0]
21. `2arms_circles_turn.mp4` - 转向摆动 [0.5, 0.0, 0.6]

#### 双臂挥手动作 (2arms_waving)
22. `2arms_waving_stand.mp4` - 静止挥手 [0.0, 0.0, 0.0]
23. `2arms_waving_walk.mp4` - 行走挥手 [0.8, 0.0, 0.0]
24. `2arms_waving_side.mp4` - 侧移挥手 [0.0, 0.7, 0.0]

#### 单臂圆周摆动 (1arm_circles)
25. `1arm_circles_stand.mp4` - 静止摆动 [0.0, 0.0, 0.0]
26. `1arm_circles_walk.mp4` - 行走摆动 [1.0, 0.0, 0.0]
27. `1arm_circles_complex.mp4` - 复合摆动 [0.6, 0.4, 0.2]

#### 单臂挥手动作 (1arm_waving)
28. `1arm_waving_stand.mp4` - 静止挥手 [0.0, 0.0, 0.0]
29. `1arm_waving_walk.mp4` - 行走挥手 [0.9, 0.0, 0.0]
30. `1arm_waving_turn.mp4` - 转向挥手 [0.0, 0.0, 0.8]

### 复杂运动 (9个视频 = 3个轨迹 × 3个命令) - 待更新

#### 太极推手动作 (taichi) - 待完善
31. `taichi_stand.mp4` - 静止太极 [0.0, 0.0, 0.0]
32. `taichi_slow.mp4` - 慢步太极 [0.3, 0.0, 0.0]
33. `taichi_turn.mp4` - 转身太极 [0.2, 0.0, 0.4]

#### 拳击动作 (boxing) - 待完善
34. `boxing_stand.mp4` - 静止拳击 [0.0, 0.0, 0.0]
35. `boxing_advance.mp4` - 前进拳击 [0.6, 0.0, 0.0]
36. `boxing_mobile.mp4` - 机动拳击 [0.4, 0.3, 0.5]

#### 随机运动 (random) - 待完善
37. `random_stand.mp4` - 静止随机 [0.0, 0.0, 0.0]
38. `random_walk.mp4` - 行走随机 [0.8, 0.0, 0.0]
39. `random_omni.mp4` - 全向随机 [0.5, 0.5, 0.3]

## 录制建议

### 技术要求
- **分辨率**: 1920×1080 (Full HD)
- **帧率**: 30 FPS
- **格式**: MP4 (H.264 编码)
- **时长**: 15-20秒每个演示
- **比特率**: 2-5 Mbps (平衡质量和文件大小)

### 拍摄建议
- **视角**: 侧视角为主，能同时看到上下身动作协调
- **背景**: 简洁的仿真环境背景
- **光照**: 充足且均匀的光照
- **焦点**: 机器人整体，确保上身轨迹和下身行走清晰可见

### 内容要求
- **每个视频**: 展示特定轨迹+特定运动命令的组合
- **演示时长**: 包含2-3个完整的动作循环
- **展示重点**: 上身轨迹执行 + 下身行走控制的协调配合

### 录制流程

#### 第一步：准备环境
```bash
cd /path/to/unitree_rl_gym/deploy/deploy_mujoco
# 确保有正确的模型文件和配置
```

#### 第二步：分类录制 

**静态姿态录制（已实现的6个轨迹）:**
```bash
# 示例：双臂前伸姿态的三个命令设定
python deploy_mujoco3.py g1.yaml -t pose_arms_forward --cmd 0.0,0.0,0.0  # 录制为 pose_arms_forward_stand.mp4
python deploy_mujoco3.py g1.yaml -t pose_arms_forward --cmd 1.0,0.0,0.0  # 录制为 pose_arms_forward_walk.mp4
python deploy_mujoco3.py g1.yaml -t pose_arms_forward --cmd 0.5,0.0,0.5  # 录制为 pose_arms_forward_turn.mp4
```

**动态轨迹录制（已实现的4个轨迹）:**
```bash
# 示例：双臂圆周摆动的三个命令设定
python deploy_mujoco3.py g1.yaml -t 2arms_circles --cmd 0.0,0.0,0.0  # 录制为 2arms_circles_stand.mp4
python deploy_mujoco3.py g1.yaml -t 2arms_circles --cmd 1.0,0.0,0.0  # 录制为 2arms_circles_walk.mp4
python deploy_mujoco3.py g1.yaml -t 2arms_circles --cmd 0.5,0.0,0.6  # 录制为 2arms_circles_turn.mp4
```

**复杂运动录制（taichi, boxing, random - 需要先实现）:**
```bash
# 注意：这些轨迹可能需要先在代码中实现或调试
python deploy_mujoco3.py g1.yaml -t taichi --cmd 0.0,0.0,0.0  # 录制为 taichi_stand.mp4
python deploy_mujoco3.py g1.yaml -t boxing --cmd 0.0,0.0,0.0  # 录制为 boxing_stand.mp4  
python deploy_mujoco3.py g1.yaml -t random --cmd 0.0,0.0,0.0  # 录制为 random_stand.mp4
```

#### 第三步：文件管理
```bash
# 检查所有视频文件是否已录制
./check_videos.sh

# 确保文件命名格式正确
ls demo/*.mp4 | wc -l  # 应该显示30个文件（完成后）
```

## 命令参数对照表

| 命令设定 | 含义 | 适用场景 |
|---------|------|---------|
| [0.0, 0.0, 0.0] | 原地站立 | 展示纯轨迹动作 |
| [1.0, 0.0, 0.0] | 前进1m/s | 展示轨迹+前进协调 |
| [0.8, 0.0, 0.0] | 前进0.8m/s | 稍慢前进，适合复杂轨迹 |
| [0.0, 0.8, 0.0] | 左侧移0.8m/s | 展示轨迹+侧移协调 |
| [0.0, 0.0, 1.0] | 逆时针转1rad/s | 展示轨迹+快速转向 |
| [0.5, 0.0, 0.5] | 前进+转向 | 展示轨迹+复合运动 |

## 注意事项

1. **优先级**: 先录制已实现的轨迹（静态姿态6个 + 动态轨迹4个）
2. **命名规范**: 严格按照清单中的文件名，不要自创命名
3. **质量检查**: 录制后使用 `./check_videos.sh` 验证文件
4. **更新文档**: 如有轨迹变更，需同步更新README.md和此清单

可以使用以下命令进行录制演示：

```bash
# 静态姿态
./run_hybrid_deploy.sh pose_arms_forward
./run_hybrid_deploy.sh pose_t_shape
./run_hybrid_deploy.sh pose_arms_up
./run_hybrid_deploy.sh pose_left_down_right_forward
./run_hybrid_deploy.sh pose_left_down_right_side
./run_hybrid_deploy.sh pose_torso_side_twist

# 动态轨迹
./run_hybrid_deploy.sh 2arms_circles
./run_hybrid_deploy.sh 2arms_waving
./run_hybrid_deploy.sh 1arm_circles
./run_hybrid_deploy.sh 1arm_waving

# 复杂运动
./run_hybrid_deploy.sh taichi
./run_hybrid_deploy.sh boxing
./run_hybrid_deploy.sh random
```

## 文件检查清单

录制完成后，请确认：
- [ ] 所有13个MP4文件都已创建
- [ ] 文件命名与上述清单完全一致
- [ ] 文件大小合理（建议每个文件5-20MB）
- [ ] 视频播放正常，画质清晰
- [ ] 音频可选（建议静音或添加背景音乐）
