import time
import numpy as np
import mujoco.viewer
import mujoco
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml
import math


def get_gravity_orientation(quaternion):
    """Extract gravity vector from quaternion orientation"""
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


def generate_upper_body_trajectory(time_sim, config):
    """Generate predefined trajectories for upper body joints"""
    # Extract trajectory parameters
    traj_amp = config["trajectory_amplitude"]
    traj_freq = config["trajectory_frequency"]
    torso_amp = config["torso_bend_amplitude"]
    torso_freq = config["torso_bend_frequency"]
    arm_offset = config["arm_forward_offset"]
    
    # Time-based phase for smooth trajectories
    phase = 2 * math.pi * traj_freq * time_sim
    torso_phase = 2 * math.pi * torso_freq * time_sim
    
    # Initialize upper body target positions (15 DOF)
    upper_body_targets = np.zeros(15)
    
    # Torso joint (index 0): forward bending motion
    upper_body_targets[0] = torso_amp * math.sin(torso_phase)
    
    # Left arm joints (indices 1-7)
    # Shoulder pitch: circular motion + forward offset
    upper_body_targets[1] = arm_offset + traj_amp * math.sin(phase)
    # Shoulder roll: slight outward motion
    upper_body_targets[2] = 0.3 * math.sin(phase)
    # Shoulder yaw: no motion
    upper_body_targets[3] = 0.0
    # Elbow pitch: bent position with slight variation
    upper_body_targets[4] = 1.2 + 0.3 * math.cos(phase)
    # Elbow roll: no motion
    upper_body_targets[5] = 0.0
    # Wrist joints: minimal motion
    upper_body_targets[6] = 0.0
    upper_body_targets[7] = 0.0
    
    # Right arm joints (indices 8-14) - mirror left arm
    upper_body_targets[8] = arm_offset + traj_amp * math.sin(phase)
    # Shoulder roll: opposite direction for symmetric motion
    upper_body_targets[9] = -0.3 * math.sin(phase)
    upper_body_targets[10] = 0.0
    upper_body_targets[11] = 1.2 + 0.3 * math.cos(phase)
    upper_body_targets[12] = 0.0
    upper_body_targets[13] = 0.0
    upper_body_targets[14] = 0.0
    
    return upper_body_targets


def extract_observations(d, default_angles, config, action, time_sim):
    """Extract observations for the policy (same as original)"""
    # Joint positions and velocities (lower body only)
    qj = d.qpos[7:19]  # Lower body joints
    dqj = d.qvel[6:18]  # Lower body joint velocities
    quat = d.qpos[3:7]  # Base orientation
    omega = d.qvel[3:6]  # Angular velocity

    # Scale observations
    qj = (qj - default_angles) * config["dof_pos_scale"]
    dqj = dqj * config["dof_vel_scale"]
    gravity_orientation = get_gravity_orientation(quat)
    omega = omega * config["ang_vel_scale"]

    # Gait phase
    period = 0.8
    count = time_sim
    phase = count % period / period
    sin_phase = np.sin(2 * np.pi * phase)
    cos_phase = np.cos(2 * np.pi * phase)

    # Command
    cmd = np.array(config["cmd_init"], dtype=np.float32)
    cmd_scaled = cmd * np.array(config["cmd_scale"], dtype=np.float32)

    # Assemble observations
    obs = np.zeros(config["num_obs"], dtype=np.float32)
    obs[:3] = omega
    obs[3:6] = gravity_orientation
    obs[6:9] = cmd_scaled
    obs[9:9 + config["num_actions"]] = qj
    obs[9 + config["num_actions"]:9 + 2 * config["num_actions"]] = dqj
    obs[9 + 2 * config["num_actions"]:9 + 3 * config["num_actions"]] = action
    obs[9 + 3 * config["num_actions"]:9 + 3 * config["num_actions"] + 2] = \
        np.array([sin_phase, cos_phase])

    return obs


if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file", type=str, help="config file name in the config folder"
    )
    args = parser.parse_args()
    config_file = args.config_file

    # Load configuration
    config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/" \
                  f"{config_file}"
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace(
            "{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR
        )
        xml_path = config["xml_path"].replace(
            "{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR
        )

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        # Lower body PD gains and defaults
        lower_kps = np.array(config["lower_body_kps"], dtype=np.float32)
        lower_kds = np.array(config["lower_body_kds"], dtype=np.float32)
        lower_default_angles = np.array(
            config["lower_body_default_angles"], dtype=np.float32)

        # Upper body PD gains and defaults
        upper_kps = np.array(config["upper_body_kps"], dtype=np.float32)
        upper_kds = np.array(config["upper_body_kds"], dtype=np.float32)
        upper_default_angles = np.array(
            config["upper_body_default_angles"], dtype=np.float32)

        # Combine PD gains for all joints (27 DOF total)
        all_kps = np.concatenate([lower_kps, upper_kps])
        all_kds = np.concatenate([lower_kds, upper_kds])

    print(f"Policy path: {policy_path}")
    print(f"XML path: {xml_path}")
    print(f"Lower body DOF: {len(lower_kps)}")
    print(f"Upper body DOF: {len(upper_kps)}")
    print(f"Total DOF: {len(all_kps)}")

    # Initialize control variables
    lower_action = np.zeros(config["num_actions"], dtype=np.float32)
    lower_target_dof_pos = lower_default_angles.copy()
    upper_target_dof_pos = upper_default_angles.copy()
    obs = np.zeros(config["num_obs"], dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    print(f"Model has {m.nq} position DOF and {m.nv} velocity DOF")
    print(f"Model has {m.nu} actuators")

    # Load policy
    policy = torch.jit.load(policy_path)
    print("Policy loaded successfully")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration.
        start = time.time()
        
        while viewer.is_running() and \
                time.time() - start < simulation_duration:
            step_start = time.time()
            current_time = time.time() - start

            # Combine target positions for all joints
            all_target_dof_pos = np.concatenate([
                lower_target_dof_pos, upper_target_dof_pos])
            
            # Current joint positions (27 DOF: 12 lower + 15 upper)
            current_joint_pos = d.qpos[7:34]  # Skip floating base (7 DOF)
            current_joint_vel = d.qvel[6:33]  # Skip floating base (6 DOF)

            # Compute control torques using PD control
            tau = pd_control(
                all_target_dof_pos, current_joint_pos, all_kps,
                np.zeros_like(all_kds), current_joint_vel, all_kds
            )
            
            # Apply control torques
            d.ctrl[:] = tau

            # Step physics
            mujoco.mj_step(m, d)
            counter += 1

            # Update control at decimated frequency
            if counter % control_decimation == 0:
                # Generate upper body trajectory
                upper_target_dof_pos = generate_upper_body_trajectory(
                    current_time, config)
                upper_target_dof_pos += upper_default_angles

                # Create observations for policy (lower body only)
                obs = extract_observations(
                    d, lower_default_angles, config, lower_action,
                    current_time)

                # Get action from policy
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                lower_action = policy(obs_tensor).detach().numpy().squeeze()
                lower_action = lower_action[:config["num_actions"]]

                # Transform action to target positions for lower body
                lower_target_dof_pos = (
                    lower_action * config["action_scale"] +
                    lower_default_angles)

                # Debug output
                debug_freq = control_decimation * 10
                if counter % debug_freq == 0:  # Print every 10 control steps
                    print(f"Time: {current_time:.2f}s")
                    print(f"Lower body targets: {lower_target_dof_pos[:6]}")
                    print(f"Upper body targets: {upper_target_dof_pos[:4]}")
                    print("---")

            # Sync viewer
            viewer.sync()

            # Time keeping
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    print("Simulation completed!")
