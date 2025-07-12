#!/bin/bash

# H1_2 混合控制系统 - 静态姿态演示脚本
# 仅演示静态姿态，每个姿态保持10秒

echo "======================================="
echo "H1_2 静态姿态演示"
echo "======================================="
echo "系统配置："
echo "  下半身: 强化学习策略控制 (12 DOF)"
echo "  上半身: 静态姿态控制 (15 DOF)"
echo "  展示时长: 每个姿态10秒"
echo ""

CONFIG_FILE="h1_2_hybrid.yaml"
cd "$(dirname "$0")"

echo "开始演示静态姿态..."
echo "-------------------------------------"

echo "1. 双臂前伸姿态 (pose_arms_forward)"
timeout 10s python deploy_mujoco3.py $CONFIG_FILE --trajectory pose_arms_forward || echo "✓ 演示完成"
sleep 2

echo "2. T字形张开姿态 (pose_t_shape)"
timeout 10s python deploy_mujoco3.py $CONFIG_FILE --trajectory pose_t_shape || echo "✓ 演示完成"
sleep 2

echo "3. 双臂上举姿态 (pose_arms_up)"
timeout 10s python deploy_mujoco3.py $CONFIG_FILE --trajectory pose_arms_up || echo "✓ 演示完成"
sleep 2

echo "4. 左下右前姿态 (pose_left_down_right_forward)"
timeout 10s python deploy_mujoco3.py $CONFIG_FILE --trajectory pose_left_down_right_forward || echo "✓ 演示完成"
sleep 2

echo "5. 左下右侧姿态 (pose_left_down_right_side)"
timeout 10s python deploy_mujoco3.py $CONFIG_FILE --trajectory pose_left_down_right_side || echo "✓ 演示完成"
sleep 2

echo "6. 躯干扭转姿态 (pose_torso_side_twist)"
timeout 10s python deploy_mujoco3.py $CONFIG_FILE --trajectory pose_torso_side_twist || echo "✓ 演示完成"
sleep 2

echo ""
echo "======================================="
echo "静态姿态演示完成！"
echo "======================================="

echo ""
echo "演示总结："
echo "  展示了 6 种静态姿态"
echo "  每种姿态持续 10 秒"
echo ""
echo "如需单独展示特定姿态："
echo "  ./run_hybrid_deploy.sh <pose_name>"
echo "  例如: ./run_hybrid_deploy.sh pose_t_shape"
