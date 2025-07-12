# H1_2 Hybrid Control testsets

**Language**: [🇺🇸 English (current)](#) | [🇨🇳 中文版本](README_zh.md)

A hybrid control testsets for H1_2 humanoid robots, combining reinforcement learning locomotion with trajectory-based upper body control.

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Demo Videos](#demo-videos)
- [Quick Start](#quick-start)
- [Development](#development)

## Overview

H1_2 Hybrid Control System implements a dual-control architecture for humanoid robotics, partitioning 27 DOF into specialized control domains:

- **Lower Body (12 DOF)**: PPO locomotion control for stable bipedal gait and balance
- **Upper Body (15 DOF)**: Trajectory-based control for complex manipulation and expressive motions

## System Architecture

### PPO-based Lower Body Control (12 DOF)
- **Joints**: 6 DOF per leg (3 hip + 1 knee + 2 ankle)
- **Observation**: 47D state vector (base orientation, joint states, action history, gait phase)
- **Action**: 12D continuous joint angle offsets
- **Training**: PPO algorithm on Isaac Gym with multi-objective rewards
- **Update Rate**: 50Hz

### Trajectory-based Upper Body Control (15 DOF)
- **Joints**: 1 torso + 7 per arm (shoulder, elbow, wrist joints)
- **Control**: PD controller with predefined motion primitives
- **Update Rate**: 50Hz (synchronized with RL policy)

### MuJoCo Joint Order

The system uses the following standardized order for 27 joint control:

```python
MUJOCO_JOINT_ORDER = [
    # Lower body joints (12 DOF) - RL control
    'left_hip_yaw_joint',    'left_hip_pitch_joint',   'left_hip_roll_joint',
    'left_knee_joint',       'left_ankle_pitch_joint', 'left_ankle_roll_joint',
    'right_hip_yaw_joint',   'right_hip_pitch_joint',  'right_hip_roll_joint', 
    'right_knee_joint',      'right_ankle_pitch_joint','right_ankle_roll_joint',
    
    # Upper body joints (15 DOF) - Trajectory control
    'torso_joint',                                     # Torso (1 DOF)
    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',  # L shoulder (3 DOF)
    'left_elbow_pitch_joint',    'left_elbow_roll_joint',                                # L elbow (2 DOF)
    'left_wrist_pitch_joint',    'left_wrist_yaw_joint',                                 # L wrist (2 DOF)
    'right_shoulder_pitch_joint','right_shoulder_roll_joint','right_shoulder_yaw_joint', # R shoulder (3 DOF)
    'right_elbow_pitch_joint',   'right_elbow_roll_joint',                               # R elbow (2 DOF)
    'right_wrist_pitch_joint',   'right_wrist_yaw_joint'                                 # R wrist (2 DOF)
]
```

### Supported Trajectories

#### Static Poses
- `pose_arms_forward` - Arms extended forward
- `pose_t_shape` - T-pose with arms spread
- `pose_arms_up` - Victory pose with arms raised
- `pose_left_down_right_forward` - Asymmetric pointing gesture
- `pose_left_down_right_side` - Asymmetric side pose
- `pose_torso_side_twist` - Torso rotation demonstration

#### Dynamic Motions
- `2arms_circles` - Synchronized bilateral arm swinging
- `2arms_waving` - Bilateral greeting gesture
- `1arm_circles` - Unilateral arm swinging
- `1arm_waving` - Unilateral greeting gesture
- `taichi` - Tai Chi flowing movements
- `boxing` - Alternating punching motions
- `random` - Stochastic upper body movements

## Demo Videos

> Each trajectory demonstrates coordination between RL-based locomotion and predefined upper body motions under various movement commands.

All video demonstrations showcase different motion commands: `[forward, lateral, angular]` velocities in m/s and rad/s.
The `stand` command is `[0.0, 0.0, 0.0]`, `walk forward` command is `[0.5, 0.0, 0.0]`, and `walk+turn` command is `[0.5, 0.3, 0.0]`.

> **💡 Tip**: Video files are located in the `demo/` directory.

### Static Poses

#### pose_arms_forward
<table>
<tr>
<td width="33%" align="center">
<strong>stand</strong><br>
<img src="demo/pose_arms_forward_stand.gif" width="100%" alt="pose_arms_forward_stand">
</td>
<td width="33%" align="center">
<strong>walk forward</strong><br>
<img src="demo/pose_arms_forward_walk.gif" width="100%" alt="pose_arms_forward_walk">
</td>
<td width="33%" align="center">
<strong>walk+turn</strong><br>
<img src="demo/pose_arms_forward_turn.gif" width="100%" alt="pose_arms_forward_turn">
</td>
</tr>
</table>

#### pose_t_shape
<table>
<tr>
<td width="33%" align="center">
<strong>stand</strong><br>
<img src="demo/pose_t_shape_stand.gif" width="100%" alt="pose_t_shape_stand">
</td>
<td width="33%" align="center">
<strong>walk forward</strong><br>
<img src="demo/pose_t_shape_walk.gif" width="100%" alt="pose_t_shape_walk">
</td>
<td width="33%" align="center">
<strong>walk+turn</strong><br>
<img src="demo/pose_t_shape_turn.gif" width="100%" alt="pose_t_shape_turn">
</td>
</tr>
</table>

#### pose_arms_up
<table>
<tr>
<td width="33%" align="center">
<strong>stand</strong><br>
<img src="demo/pose_arms_up_stand.gif" width="100%" alt="pose_arms_up_stand">
</td>
</td>
<td width="33%" align="center">
<strong>walk forward</strong><br>
<img src="demo/pose_arms_up_walk.gif" width="100%" alt="pose_arms_up_walk">
</td>
<td width="33%" align="center">
<strong>walk+turn</strong><br>
<img src="demo/pose_arms_up_turn.gif" width="100%" alt="pose_arms_up_turn">
</td>
</tr>
</table>

#### pose_left_down_right_forward
<table>
<tr>
<td width="33%" align="center">
<strong>stand</strong><br>
<img src="demo/pose_left_down_right_forward_stand.gif" width="100%" alt="pose_left_down_right_forward_stand">
</td>
<td width="33%" align="center">
<strong>walk forward</strong><br>
<img src="demo/pose_left_down_right_forward_walk.gif" width="100%" alt="pose_left_down_right_forward_walk">
</td>
<td width="33%" align="center">
<strong>walk+turn</strong><br>
<img src="demo/pose_left_down_right_forward_turn.gif" width="100%" alt="pose_left_down_right_forward_turn">
</td>
</tr>
</table>

#### pose_left_down_right_side
<table>
<tr>
<td width="33%" align="center">
<strong>stand</strong><br>
<img src="demo/pose_left_down_right_side_stand.gif" width="100%" alt="pose_left_down_right_side_stand">
</td>
<td width="33%" align="center">
<strong>walk+turn</strong><br>
<img src="demo/pose_left_down_right_side_walk.gif" width="100%" alt="pose_left_down_right_side_walk">
</td>
<td width="33%" align="center">
<strong>walk forward</strong><br>
<img src="demo/pose_left_down_right_side_turn.gif" width="100%" alt="pose_left_down_right_side_turn">
</td>
</tr>
</table>

#### pose_torso_side_twist
<table>
<tr>
<td width="33%" align="center">
<strong>stand</strong><br>
<img src="demo/pose_torso_side_twist_stand.gif" width="100%" alt="pose_torso_side_twist_stand">
</td>
<td width="33%" align="center">
<strong>walk forward</strong><br>
<img src="demo/pose_torso_side_twist_walk.gif" width="100%" alt="pose_torso_side_twist_walk">
</td>
<td width="33%" align="center">
<strong>walk+turn</strong><br>
<img src="demo/pose_torso_side_twist_turn.gif" width="100%" alt="pose_torso_side_twist_turn">
</td>
</tr>
</table>

### Dynamic Motions

#### 2arms_circles
<table>
<tr>
<td width="33%" align="center">
<strong>stand</strong><br>
<img src="demo/2arms_circles_stand.gif" width="100%" alt="2arms_circles_stand">
</td>
<td width="33%" align="center">
<strong>walk forward</strong><br>
<img src="demo/2arms_circles_walk.gif" width="100%" alt="2arms_circles_walk">
</td>
<td width="33%" align="center">
<strong>walk+turn</strong><br>
<img src="demo/2arms_circles_turn.gif" width="100%" alt="2arms_circles_turn">
</td>
</tr>
</table>

#### 2arms_waving
<table>
<tr>
<td width="33%" align="center">
<strong>stand</strong><br>
<img src="demo/2arms_waving_stand.gif" width="100%" alt="2arms_waving_stand">
</td>
<td width="33%" align="center">
<strong>walk forward</strong><br>
<img src="demo/2arms_waving_walk.gif" width="100%" alt="2arms_waving_walk">
</td>
<td width="33%" align="center">
<strong>walk+turn</strong><br>
<img src="demo/2arms_waving_turn.gif" width="100%" alt="2arms_waving_turn">
</td>
</tr>
</table>

#### 1arm_circles_complex
<table>
<tr>
<td width="33%" align="center">
<strong>stand</strong><br>
<img src="demo/1arm_circles_stand.gif" width="100%" alt="1arm_circles_stand">
</td>
<td width="33%" align="center">
<strong>walk forward</strong><br>
<img src="demo/1arm_circles_walk.gif" width="100%" alt="1arm_circles_walk">
</td>
<td width="33%" align="center">
<strong>walk+turn</strong><br>
<img src="demo/1arm_circles_turn.gif" width="100%" alt="1arm_circles_turn">
</td>
</tr>
</table>

#### 1arm_waving
<table>
<tr>
<td width="33%" align="center">
<strong>stand</strong><br>
<img src="demo/1arm_waving_stand.gif" width="100%" alt="1arm_waving_stand">
</td>
<td width="33%" align="center">
<strong>walk forward</strong><br>
<img src="demo/1arm_waving_walk.gif" width="100%" alt="1arm_waving_walk">
</td>
<td width="33%" align="center">
<strong>walk+turn</strong><br>
<img src="demo/1arm_waving_turn.gif" width="100%" alt="1arm_waving_turn">
</td>
</tr>
</table>

### Other Motions

#### taichi
<table>
<tr>
<td width="33%" align="center">
<strong>stand</strong><br>
<img src="demo/taichi_stand.gif" width="100%" alt="taichi_stand">
</td>
<td width="33%" align="center">
<strong>walk forward</strong><br>
<img src="demo/taichi_walk.gif" width="100%" alt="taichi_walk">
</td>
<td width="33%" align="center">
<strong>walk+turn</strong><br>
<img src="demo/taichi_turn.gif" width="100%" alt="taichi_turn">
</td>
</tr>
</table>

#### boxing
<table>
<tr>
<td width="33%" align="center">
<strong>stand</strong><br>
<img src="demo/boxing_stand.gif" width="100%" alt="boxing_stand">
</td>
<td width="33%" align="center">
<strong>walk forward</strong><br>
<img src="demo/boxing_walk.gif" width="100%" alt="boxing_walk">
</td>
<td width="33%" align="center">
<strong>walk+turn</strong><br>
<img src="demo/boxing_turn.gif" width="100%" alt="boxing_turn">
</td>
</tr>
</table>

#### random
<table>
<tr>
<td width="33%" align="center">
<strong>stand</strong><br>
<img src="demo/random_stand.gif" width="100%" alt="random_stand">
</td>
<td width="33%" align="center">
<strong>walk forward</strong><br>
<img src="demo/random_walk.gif" width="100%" alt="random_walk">
</td>
<td width="33%" align="center">
<strong>walk+turn</strong><br>
<img src="demo/random_turn.gif" width="100%" alt="random_turn">
</td>
</tr>
</table>


## Quick Start

### Basic Usage
```bash
python deploy_mujoco_hybrid.py <config> --trajectory <type> --cmd <motion_command>
```

**Examples**:
```bash
# Static poses
python deploy_mujoco_hybrid.py h1_2_hybrid.yaml -t pose_t_shape
python deploy_mujoco_hybrid.py h1_2_hybrid.yaml -t pose_arms_up --cmd 0.5,0,0.3

# Dynamic trajectories  
python deploy_mujoco_hybrid.py h1_2_hybrid.yaml -t 2arms_circles --cmd 1.0,0,0
python deploy_mujoco_hybrid.py h1_2_boxing.yaml -t boxing --cmd 0.6,0.3,0.5
# boxing pose uses different yaml file
# Batch demonstrations
./demo_all_trajectories.sh              # All trajectories
```


## Configuration

### Main Config: `h1_2_hybrid.yaml`
```yaml
# Paths
policy_path: "{LEGGED_GYM_ROOT_DIR}/logs/h1_2/exported/policies/policy_lstm_1.pt"
xml_path: "{LEGGED_GYM_ROOT_DIR}/resources/robots/h1_2/scene.xml"

# Simulation  
simulation_duration: 60.0
simulation_dt: 0.002
control_decimation: 10

# Control parameters
lower_body_kps: [200, 200, 200, 300, 40, 40] * 2  # Left + Right leg
upper_body_kps: [50] +[80, 80, 50, 50, 20, 20, 20] *2
```

## Development

### Adding Custom Trajectories

1. **Define trajectory function**:
```python
def trajectory_custom(time_sim, config):
    targets = np.zeros(15)  # [torso, left_arm×7, right_arm×7]
    # Implement motion logic here
    return targets
```

2. **Register in trajectory functions dict**:
```python
trajectory_functions["custom"] = trajectory_custom
```

3. **Add to argument parser choices**.

