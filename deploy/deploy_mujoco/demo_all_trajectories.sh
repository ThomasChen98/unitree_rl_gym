#!/bin/bash

# Script to demonstrate all available trajectories for H1_2 hybrid control
# Each trajectory will run for 10 seconds with different command parameters

CONFIG_FILE="h1_2_hybrid.yaml"
DEMO_DURATION=10  # seconds per trajectory

echo "=========================================="
echo "H1_2 Hybrid Control - All Trajectories Demo"
echo "=========================================="
echo "This script will demonstrate all available trajectories"
echo "Each trajectory will run for $DEMO_DURATION seconds"
echo ""

# Check if config file exists
if [ ! -f "configs/$CONFIG_FILE" ]; then
    echo "Error: Configuration file configs/$CONFIG_FILE not found!"
    exit 1
fi

# All available trajectories grouped by category
# Dynamic trajectories - dual arm
dual_arm_trajectories=("2arms_circles" "2arms_waving")

# Dynamic trajectories - single arm  
single_arm_trajectories=("1arm_circles" "1arm_waving")

# Complex motion trajectories
complex_trajectories=("taichi" "boxing" "random")

# Static upper body poses
static_poses=("pose_arms_forward" "pose_left_down_right_forward" "pose_t_shape" 
              "pose_left_down_right_side" "pose_torso_side_twist" "pose_arms_up")

# Combine all trajectories
all_trajectories=(
    "${dual_arm_trajectories[@]}" 
    "${single_arm_trajectories[@]}" 
    "${complex_trajectories[@]}" 
    "${static_poses[@]}"
)

# Trajectory descriptions
declare -A descriptions
descriptions["2arms_circles"]="Dual arm circular motion - Both arms move in circles with torso bending"
descriptions["2arms_waving"]="Dual arm waving - Both arms wave simultaneously"
descriptions["1arm_circles"]="Single arm circular motion - One arm moves in circles"
descriptions["1arm_waving"]="Single arm waving - Single arm waving motion"
descriptions["taichi"]="Tai Chi motion - Slow push-pull movements"
descriptions["boxing"]="Boxing motion - Alternating punching movements"
descriptions["random"]="Random motion - Random upper body movements"
descriptions["pose_arms_forward"]="Static pose - Arms extended forward"
descriptions["pose_left_down_right_forward"]="Static pose - Left arm down, right arm forward"
descriptions["pose_t_shape"]="Static pose - T-shape with arms extended sideways"
descriptions["pose_left_down_right_side"]="Static pose - Left arm down, right arm to side"
descriptions["pose_torso_side_twist"]="Static pose - Torso twisted to side"
descriptions["pose_arms_up"]="Static pose - Both arms raised up"

# Command parameters for demo (forward, angular, lateral)
cmd_params="0.0,0.0,0.0"

cd "$(dirname "$0")"

echo ""
echo "=========================================="
echo "Demonstrating all trajectory categories:"
echo "1. Dual arm dynamic motions (${#dual_arm_trajectories[@]} trajectories)"
echo "2. Single arm dynamic motions (${#single_arm_trajectories[@]} trajectories)"
echo "3. Complex motions (${#complex_trajectories[@]} trajectories)"
echo "4. Static poses (${#static_poses[@]} trajectories)"
echo "Total: ${#all_trajectories[@]} trajectories"
echo "=========================================="
echo ""

for trajectory in "${all_trajectories[@]}"; do
    echo ""
    echo "=========================================="
    echo "Demonstrating: $trajectory"
    echo "Description: ${descriptions[$trajectory]}"
    echo "Duration: $DEMO_DURATION seconds"
    echo "Command: $cmd_params (forward, angular, lateral)"
    echo "=========================================="
    echo ""
    echo "Press Enter to continue, or Ctrl+C to exit..."
    read -r
    
    echo "Starting $trajectory demonstration..."
    
    # Run the trajectory with hybrid controller
    python deploy_mujoco_hybrid.py  "$CONFIG_FILE" \
                                   -t "$trajectory" \
                                   --cmd $cmd_params \
                        
    
    echo ""
    echo "$trajectory demonstration completed"
done

echo ""
echo "=========================================="
echo "All trajectory demonstrations completed!"
echo ""
echo "Available motion types:"
echo "  Dynamic (dual arm): ${dual_arm_trajectories[*]}"
echo "  Dynamic (single arm): ${single_arm_trajectories[*]}"
echo "  Complex motions: ${complex_trajectories[*]}"
echo "  Static poses: ${static_poses[*]}"
echo ""
echo "Usage examples:"
echo "  python deploy_mujoco_hybrid.py --config configs/h1_2_hybrid.yaml -t 2arms_circles --cmd 0.5,0,0.0"
echo "  python deploy_mujoco_hybrid.py --config configs/h1_2_boxing.yaml -t boxing --cmd 0.0,0.3,0.0"
echo "  python deploy_mujoco_hybrid.py --config configs/h1_2_hybrid.yaml -t pose_t_shape --cmd 0.0,0.0,0.2"
echo "=========================================="
