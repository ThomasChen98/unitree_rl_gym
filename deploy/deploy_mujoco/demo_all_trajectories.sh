#!/bin/bash

# Script to demonstrate all available trajectories
# Each trajectory will run for 10 seconds

CONFIG_FILE="h1_2_hybrid.yaml"
DEMO_DURATION=10  # seconds per trajectory

echo "=========================================="
echo "H1_2 Hybrid Control - Trajectory Demo"
echo "=========================================="
echo "This script will demonstrate all available trajectories"
echo "Each trajectory will run for $DEMO_DURATION seconds"
echo ""

# Check if config file exists
if [ ! -f "configs/$CONFIG_FILE" ]; then
    echo "Error: Configuration file configs/$CONFIG_FILE not found!"
    exit 1
fi

# Array of all available trajectories
trajectories=("circles" "waving" "taichi" "boxing" "dancing" "stretching")

# Trajectory descriptions
declare -A descriptions
descriptions["circles"]="双臂圆周运动 - 手臂做圆周运动，躯干前弯"
descriptions["waving"]="挥手打招呼 - 右臂挥手动作"
descriptions["taichi"]="太极推手 - 缓慢的推拉动作"
descriptions["boxing"]="拳击动作 - 交替出拳"
descriptions["dancing"]="舞蹈动作 - 双臂协调摆动"
descriptions["stretching"]="拉伸动作 - 多阶段拉伸运动"

cd "$(dirname "$0")"

for trajectory in "${trajectories[@]}"; do
    echo ""
    echo "==========================================Ó"
    echo "演示轨迹: $trajectory"
    echo "描述: ${descriptions[$trajectory]}"
    echo "持续时间: $DEMO_DURATION 秒"
    echo "=========================================="
    echo ""
    echo "按 Enter 继续，或 Ctrl+C 退出..."
    read -r
    
    echo "启动 $trajectory 轨迹演示..."
    
    # Create temporary config with short duration
    temp_config="h1_2_hybrid.yaml"
    cp "configs/$CONFIG_FILE" "$temp_config"
    
    # Modify duration in temp config
    sed -i "s/simulation_duration: .*/simulation_duration: $DEMO_DURATION.0/" "$temp_config"
    
    # Run the trajectory
    python deploy_mujoco3.py "$temp_config" --trajectory "$trajectory"
    
    # Clean up
    rm -f "$temp_config"
    
    echo ""
    echo "$trajectory 轨迹演示完成"
done

echo ""
echo "=========================================="
echo "所有轨迹演示完成！"
echo "你可以使用以下命令运行特定轨迹："
echo "  python deploy_mujoco3.py h1_2_hybrid.yaml --trajectory [轨迹名称]"
echo "或者使用便捷脚本："
echo "  ./run_hybrid_deploy.sh [轨迹名称]"
echo "=========================================="
