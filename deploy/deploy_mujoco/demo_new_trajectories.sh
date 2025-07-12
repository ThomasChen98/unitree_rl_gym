#!/bin/bash

# H1_2 混合控制系统 - 轨迹演示脚本
# 展示所有可用轨迹类型，每个轨迹运行10秒

echo "======================================="
echo "H1_2 混合控制系统轨迹演示"
echo "======================================="
echo "系统配置："
echo "  下半身: 强化学习策略控制 (12 DOF)"
echo "  上半身: 轨迹控制 (15 DOF)"
echo "  演示时长: 每个轨迹10秒"
echo ""

# 配置文件
CONFIG_FILE="h1_2_hybrid.yaml"

# 确保在正确目录
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

echo ""
echo "开始演示双臂动作轨迹..."
echo "-------------------------------------"

echo "4. 双臂圆周摆动 (2arms_circles)"
timeout 10s python deploy_mujoco3.py $CONFIG_FILE --trajectory 2arms_circles || echo "✓ 演示完成"
sleep 2

echo "5. 双臂挥手动作 (2arms_waving)"
timeout 10s python deploy_mujoco3.py $CONFIG_FILE --trajectory 2arms_waving || echo "✓ 演示完成"
sleep 2

echo ""
echo "开始演示单臂动作轨迹..."
echo "-------------------------------------"

echo "6. 单臂圆周摆动 (1arm_circles)"
timeout 10s python deploy_mujoco3.py $CONFIG_FILE --trajectory 1arm_circles || echo "✓ 演示完成"
sleep 2

echo "7. 单臂挥手动作 (1arm_waving)"
timeout 10s python deploy_mujoco3.py $CONFIG_FILE --trajectory 1arm_waving || echo "✓ 演示完成"
sleep 2

echo ""
echo "开始演示复杂动作轨迹..."
echo "-------------------------------------"

echo "8. 太极推手动作 (taichi)"
timeout 10s python deploy_mujoco3.py $CONFIG_FILE --trajectory taichi || echo "✓ 演示完成"
sleep 2

echo "9. 拳击动作 (boxing)"
timeout 10s python deploy_mujoco3.py $CONFIG_FILE --trajectory boxing || echo "✓ 演示完成"
sleep 2

echo "10. 舞蹈动作 (dancing)"
timeout 10s python deploy_mujoco3.py $CONFIG_FILE --trajectory dancing || echo "✓ 演示完成"
sleep 2

echo "11. 拉伸动作 (stretching)"
timeout 10s python deploy_mujoco3.py $CONFIG_FILE --trajectory stretching || echo "✓ 演示完成"
sleep 2

echo "10. 随机动作 (random)"
timeout 10s python deploy_mujoco3.py $CONFIG_FILE --trajectory random || echo "✓ 演示完成"
sleep 2

echo ""
echo "======================================="
echo "所有轨迹演示完成！"
echo "======================================="

echo ""
echo "演示总结："
echo "  静态姿态: 3个轨迹"
echo "  双臂动作: 2个轨迹"  
echo "  单臂动作: 2个轨迹"
echo "  复杂动作: 3个轨迹"
echo "  总计: 10个轨迹"
echo ""
echo "如需单独运行特定轨迹："
echo "  ./run_hybrid_deploy.sh <trajectory_name>"
echo "  例如: ./run_hybrid_deploy.sh 2arms_circles"
echo "  ./run_hybrid_deploy.sh [trajectory_name]"
echo ""
echo "轨迹选项："
echo "  双臂动作: 2arm_circles, waving_2arm"
echo "  单臂动作: 1arm_circles, waving_1arm"  
echo "  复杂动作: taichi, boxing, dancing, stretching, random"
echo "  静止姿态: pose_arms_forward, pose_t_shape, pose_arms_up"
echo "========================================="
