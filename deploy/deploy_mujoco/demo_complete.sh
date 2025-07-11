#!/bin/bash

# Script to demonstrate all available trajectories including static poses
# Each trajectory will run for specified duration

CONFIG_FILE="h1_2_hybrid.yaml"
DYNAMIC_DURATION=10  # seconds per dynamic trajectory
STATIC_DURATION=8    # seconds per static pose

echo "=========================================="
echo "H1_2 Hybrid Control - Complete Demo"
echo "=========================================="
echo "This script will demonstrate all available trajectories and poses"
echo ""

# Check if config file exists
if [ ! -f "configs/$CONFIG_FILE" ]; then
    echo "Error: Configuration file configs/$CONFIG_FILE not found!"
    exit 1
fi

echo "=== DYNAMIC TRAJECTORIES ==="
echo ""

# Dynamic trajectories
dynamic_trajectories=("circles" "waving" "taichi" "boxing" "dancing" "stretching" "random")

# Dynamic trajectory descriptions
declare -A dynamic_descriptions
dynamic_descriptions[circles]="Arm Circles Motion"
dynamic_descriptions[waving]="Waving Hello Gesture"
dynamic_descriptions[taichi]="Tai Chi Movements"
dynamic_descriptions[boxing]="Boxing Punches"
dynamic_descriptions[dancing]="Dancing Movements"
dynamic_descriptions[stretching]="Stretching Exercises"
dynamic_descriptions[random]="Random Joint Movements"

for traj in "${dynamic_trajectories[@]}"; do
    echo "Running: ${dynamic_descriptions[$traj]} ($traj)"
    echo "Duration: $DYNAMIC_DURATION seconds"
    echo "Command: python deploy_mujoco3.py $CONFIG_FILE --trajectory $traj"
    echo ""
    
    # Run the trajectory
    timeout ${DYNAMIC_DURATION}s python deploy_mujoco3.py "$CONFIG_FILE" --trajectory "$traj" || {
        echo "Trajectory $traj completed or interrupted"
    }
    
    echo "---"
    sleep 1
done

echo ""
echo "=== STATIC POSES ==="
echo ""

# Static poses
static_poses=("pose_arms_forward" "pose_left_down_right_forward" "pose_t_shape" 
              "pose_left_down_right_side" "pose_torso_side_twist" "pose_left_up_right_down")

# Static pose descriptions
declare -A static_descriptions
static_descriptions[pose_arms_forward]="Both Arms Forward"
static_descriptions[pose_left_down_right_forward]="Left Down, Right Forward"
static_descriptions[pose_t_shape]="T-Shape Cross Pose"
static_descriptions[pose_left_down_right_side]="Left Down, Right Side"
static_descriptions[pose_torso_side_twist]="Torso Side Twist"
static_descriptions[pose_left_up_right_down]="Left Up, Right Down"

# All static poses
all_static_poses=("${static_poses[@]}")

for pose in "${all_static_poses[@]}"; do
    echo "Running: ${static_descriptions[$pose]} ($pose)"
    echo "Duration: $STATIC_DURATION seconds"
    echo "Command: python deploy_mujoco3.py $CONFIG_FILE --trajectory $pose"
    echo ""
    
    # Run the pose
    timeout ${STATIC_DURATION}s python deploy_mujoco3.py "$CONFIG_FILE" --trajectory "$pose" || {
        echo "Pose $pose completed or interrupted"
    }
    
    echo "---"
    sleep 1
done

echo ""
echo "=========================================="
echo "Complete demonstration finished!"
echo "=========================================="
echo ""
echo "Summary:"
echo "- Dynamic trajectories: ${#dynamic_trajectories[@]}"
echo "- Static poses: ${#all_static_poses[@]}"
echo "- Total demonstrations: $((${#dynamic_trajectories[@]} + ${#all_static_poses[@]}))"
echo ""
echo "To run individual trajectories:"
echo "python deploy_mujoco3.py $CONFIG_FILE --trajectory <trajectory_name>"
echo ""
echo "Available trajectories:"
for traj in "${dynamic_trajectories[@]}"; do
    echo "  - $traj (dynamic)"
done
for pose in "${all_static_poses[@]}"; do
    echo "  - $pose (static)"
done
