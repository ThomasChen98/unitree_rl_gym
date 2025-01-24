import time

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml


# Joint orders
POLICY_JOINT_ORDER = [
    'left_hip_yaw_joint', 'right_hip_yaw_joint', 'torso_joint',
    'left_hip_pitch_joint', 'right_hip_pitch_joint',
    'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint',
    'left_hip_roll_joint', 'right_hip_roll_joint',
    'left_shoulder_roll_joint', 'right_shoulder_roll_joint',
    'left_knee_joint', 'right_knee_joint',
    'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint',
    'left_ankle_pitch_joint', 'right_ankle_pitch_joint',
    'left_elbow_pitch_joint', 'right_elbow_pitch_joint',
    'left_ankle_roll_joint', 'right_ankle_roll_joint',
    'left_elbow_roll_joint', 'right_elbow_roll_joint',
    'left_wrist_pitch_joint', 'right_wrist_pitch_joint',
    'left_wrist_yaw_joint', 'right_wrist_yaw_joint'
]

MUJOCO_JOINT_ORDER = [
    'left_hip_yaw_joint', 'left_hip_pitch_joint', 'left_hip_roll_joint',
    'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
    'right_hip_yaw_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 
    'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
    'torso_joint', 
    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 
    'left_elbow_pitch_joint', 'left_elbow_roll_joint', 
    'left_wrist_pitch_joint', 'left_wrist_yaw_joint', 
    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
    'right_elbow_pitch_joint', 'right_elbow_roll_joint',
    'right_wrist_pitch_joint', 'right_wrist_yaw_joint'
]

# Create mapping indices
POLICY_TO_MUJOCO = [POLICY_JOINT_ORDER.index(joint) for joint in MUJOCO_JOINT_ORDER]
MUJOCO_TO_POLICY = [MUJOCO_JOINT_ORDER.index(joint) for joint in POLICY_JOINT_ORDER]


def get_gravity_orientation(quaternion):
    # qw = quaternion[0]
    # qx = quaternion[1]
    # qy = quaternion[2]
    # qz = quaternion[3]

    # gravity_orientation = np.zeros(3)

    # gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    # gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    # gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    # return gravity_orientation
    quat = quaternion
    quat /= np.linalg.norm(quat)
    vec = np.array([0.0, 0.0, -1.0])
    # Extract quaternion components
    qw, qx, qy, qz = quat
    
    # Compute the inverse quaternion
    quat_inv = np.array([qw, -qx, -qy, -qz])
    
    # Convert vector to quaternion (0, vec)
    vec_quat = np.array([0.0, *vec])
    
    # Perform quaternion multiplication: rotated_vec_quat = quat_inv * vec_quat * quat
    rotated_vec_quat = quat_mult(quat_mult(quat_inv, vec_quat), quat_conjugate(quat_inv))
    
    # Extract the rotated vector (x, y, z) part
    return rotated_vec_quat[1:]


def quat_mult(q1, q2):
    """
    Multiplies two quaternions.
    
    Parameters:
    - q1, q2: array-like, shape (4,)
      Quaternions [qw, qx, qy, qz].
    
    Returns:
    - result: array-like, shape (4,)
      Result of quaternion multiplication.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])


def quat_conjugate(quat):
    """
    Computes the conjugate of a quaternion.

    Parameters:
    - quat: array-like, shape (4,)
      Quaternion [qw, qx, qy, qz].
    
    Returns:
    - conjugate: array-like, shape (4,)
      Conjugate quaternion [qw, -qx, -qy, -qz].
    """
    qw, qx, qy, qz = quat
    return np.array([qw, -qx, -qy, -qz])

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


if __name__ == "__main__":
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # load policy
    policy = torch.jit.load(policy_path)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # Apply control signal here.

                # create observation
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]

                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                omega = omega * ang_vel_scale

                period = 0.8
                count = counter * simulation_dt
                phase = count % period / period
                sin_phase = np.sin(2 * np.pi * phase)
                cos_phase = np.cos(2 * np.pi * phase)

                obs[:3] = omega
                obs[3:6] = gravity_orientation
                obs[6:9] = cmd * cmd_scale
                obs[9 : 9 + num_actions] = qj
                obs[9 + num_actions : 9 + 2 * num_actions] = dqj
                obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
                obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase])
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                
                # -------------------------------------------
                # # Reorder qj, dqj, and action in obs to policy joint order
                # qj_reordered = qj[MUJOCO_TO_POLICY].copy()
                # obs[9 : 9 + num_actions] = qj_reordered

                # dqj_reordered = dqj[MUJOCO_TO_POLICY].copy()
                # obs[9 + num_actions : 9 + 2 * num_actions] = dqj_reordered

                # action_reordered_obs = action[MUJOCO_TO_POLICY].copy()
                # obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action_reordered_obs
                
                # obs_tensor = torch.from_numpy(obs[:-2]).unsqueeze(0)
                # -------------------------------------------
                
                # policy inference
                action = policy(obs_tensor).detach().numpy().squeeze()
                
                # -------------------------------------------
                # # Reorder action to match mujoco joint order
                # action_reordered = action[POLICY_TO_MUJOCO].copy()
                # action = action_reordered
                # -------------------------------------------
                
                # transform action to target_dof_pos
                target_dof_pos = action * action_scale + default_angles

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
