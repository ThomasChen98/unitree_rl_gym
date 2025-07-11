#!/bin/bash

# Script to run the hybrid control deployment with trajectory selection
# Usage: ./run_hybrid_deploy.sh [trajectory_type]
# Available trajectories: 
#   双臂动作: 2arm_circles, waving_2arm
#   单臂动作: 1arm_circles, waving_1arm
#   复杂动作: taichi, boxing, dancing, stretching, random
#   静止姿态: pose_arms_forward, pose_t_shape, pose_arms_up, etc.

# Default trajectory if not specified
TRAJECTORY=${1:-2arm_circles}

echo "Starting H1_2 hybrid control deployment..."
echo "Lower body: Policy-controlled (12 DOF)"
echo "Upper body: Trajectory-controlled (15 DOF)"
echo "Selected trajectory: $TRAJECTORY"
echo ""

# Check if config file exists
CONFIG_FILE="h1_2_hybrid.yaml"
if [ ! -f "configs/$CONFIG_FILE" ]; then
    echo "Error: Configuration file configs/$CONFIG_FILE not found!"
    exit 1
fi

# Validate trajectory type
case $TRAJECTORY in
    2arm_circles|waving_2arm|1arm_circles|waving_1arm|taichi|boxing|dancing|stretching|random|pose_arms_forward|pose_left_down_right_forward|pose_t_shape|pose_left_down_right_side|pose_torso_side_twist|pose_arms_up)
        echo "Valid trajectory selected: $TRAJECTORY"
        ;;
    *)
        echo "Warning: Unknown trajectory '$TRAJECTORY', using default '2arm_circles'"
        echo "Available trajectories:"
        echo "  双臂动作: 2arm_circles, waving_2arm"
        echo "  单臂动作: 1arm_circles, waving_1arm"
        echo "  复杂动作: taichi, boxing, dancing, stretching, random"
        echo "  静止姿态: pose_arms_forward, pose_t_shape, pose_arms_up, etc."
        TRAJECTORY="2arm_circles"
        ;;
esac

echo ""
echo "Running deployment..."
echo "=========================="

# Run the deployment
cd "$(dirname "$0")"
python deploy_mujoco3.py $CONFIG_FILE --trajectory $TRAJECTORY
