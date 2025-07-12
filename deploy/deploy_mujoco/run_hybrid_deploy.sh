#!/bin/bash

# H1_2 混合控制系统部署脚本
# 下半身（12 DOF）：强化学习策略控制
# 上半身（15 DOF）：预定义轨迹控制
#
# 使用方法: ./run_hybrid_deploy.sh [trajectory_type]
#
# 可用轨迹类型:
#   静态姿态: pose_arms_forward, pose_t_shape, pose_arms_up, 
#             pose_left_down_right_forward, pose_left_down_right_side, pose_torso_side_twist
#   双臂动作: 2arms_circles, 2arms_waving  
#   单臂动作: 1arm_circles, 1arm_waving
#   复杂动作: taichi, boxing, random

# 默认轨迹
TRAJECTORY=${1:-2arms_circles}

echo "====================================="
echo "H1_2 混合控制系统启动"
echo "====================================="
echo "系统配置："
echo "  下半身: 强化学习策略控制 (12 DOF)"
echo "  上半身: 轨迹控制 (15 DOF)"
echo "  选择轨迹: $TRAJECTORY"
echo "  配置文件: h1_2_hybrid.yaml"
echo ""

# Check if config file exists
CONFIG_FILE="h1_2_hybrid.yaml"
if [ ! -f "configs/$CONFIG_FILE" ]; then
    echo "Error: Configuration file configs/$CONFIG_FILE not found!"
    exit 1
fi

# 验证轨迹类型
case $TRAJECTORY in
    # 静态姿态
    pose_arms_forward|pose_t_shape|pose_arms_up|pose_left_down_right_forward|pose_left_down_right_side|pose_torso_side_twist)
        echo "✓ 静态姿态轨迹: $TRAJECTORY"
        ;;
    # 双臂动作  
    2arms_circles|2arms_waving)
        echo "✓ 双臂动作轨迹: $TRAJECTORY"
        ;;
    # 单臂动作
    1arm_circles|1arm_waving)
        echo "✓ 单臂动作轨迹: $TRAJECTORY"
        ;;
    # 复杂动作
    taichi|boxing|random)
        echo "✓ 复杂动作轨迹: $TRAJECTORY"
        ;;
    *)
        echo "⚠ 警告: 未知轨迹 '$TRAJECTORY'，使用默认轨迹 '2arms_circles'"
        echo ""
        echo "可用轨迹类型："
        echo "  静态姿态: pose_arms_forward, pose_t_shape, pose_arms_up"
        echo "           pose_left_down_right_forward, pose_left_down_right_side, pose_torso_side_twist"
        echo "  双臂动作: 2arms_circles, 2arms_waving"  
        echo "  单臂动作: 1arm_circles, 1arm_waving"
        echo "  复杂动作: taichi, boxing, random"
        TRAJECTORY="2arms_circles"
        ;;
esac

echo ""
echo "启动部署..."
echo "====================================="

# 运行部署
cd "$(dirname "$0")"
python deploy_mujoco3.py $CONFIG_FILE --trajectory $TRAJECTORY

echo ""
echo "部署完成"
