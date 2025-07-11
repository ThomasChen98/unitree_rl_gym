#!/bin/bash

# 演示所有新轨迹的脚本
# 每个轨迹运行10秒钟

echo "H1_2 混合控制轨迹演示 - 新版本"
echo "================================"
echo "每个轨迹运行10秒钟"
echo ""

# 设置配置文件为混合控制模式
CONFIG_FILE="h1_2_hybrid.yaml"

# 确保脚本在正确的目录中运行
cd "$(dirname "$0")"

echo "开始演示双臂轨迹..."
echo "--------------------"

echo "1. 双臂圆周摆动 (2arm_circles)"
timeout 10s python deploy_mujoco3.py $CONFIG_FILE --trajectory 2arm_circles || echo "演示完成"
sleep 2

echo "2. 双臂挥手 (waving_2arm)"
timeout 10s python deploy_mujoco3.py $CONFIG_FILE --trajectory waving_2arm || echo "演示完成"
sleep 2

echo ""
echo "开始演示单臂轨迹..."
echo "--------------------"

echo "3. 单臂圆周摆动 (1arm_circles)"
timeout 10s python deploy_mujoco3.py $CONFIG_FILE --trajectory 1arm_circles || echo "演示完成"
sleep 2

echo "4. 单臂挥手 (waving_1arm)"
timeout 10s python deploy_mujoco3.py $CONFIG_FILE --trajectory waving_1arm || echo "演示完成"
sleep 2

echo ""
echo "开始演示复杂轨迹..."
echo "--------------------"

echo "5. 太极推手 (taichi)"
timeout 10s python deploy_mujoco3.py $CONFIG_FILE --trajectory taichi || echo "演示完成"
sleep 2

echo "6. 拳击动作 (boxing)"
timeout 10s python deploy_mujoco3.py $CONFIG_FILE --trajectory boxing || echo "演示完成"
sleep 2

echo "7. 舞蹈动作 (dancing)"
timeout 10s python deploy_mujoco3.py $CONFIG_FILE --trajectory dancing || echo "演示完成"
sleep 2

echo "8. 拉伸动作 (stretching)"
timeout 10s python deploy_mujoco3.py $CONFIG_FILE --trajectory stretching || echo "演示完成"
sleep 2

echo "9. 随机动作 (random)"
timeout 10s python deploy_mujoco3.py $CONFIG_FILE --trajectory random || echo "演示完成"
sleep 2

echo ""
echo "开始演示静止姿态..."
echo "--------------------"

echo "10. 双臂前伸 (pose_arms_forward)"
timeout 5s python deploy_mujoco3.py $CONFIG_FILE --trajectory pose_arms_forward || echo "演示完成"
sleep 1

echo "11. T字形姿态 (pose_t_shape)"
timeout 5s python deploy_mujoco3.py $CONFIG_FILE --trajectory pose_t_shape || echo "演示完成"
sleep 1

echo "12. 双臂上举 (pose_arms_up)"
timeout 5s python deploy_mujoco3.py $CONFIG_FILE --trajectory pose_arms_up || echo "演示完成"

echo ""
echo "========================================="
echo "演示完成！"
echo "您可以使用以下命令单独运行任何轨迹："
echo "  ./run_hybrid_deploy.sh [trajectory_name]"
echo ""
echo "轨迹选项："
echo "  双臂动作: 2arm_circles, waving_2arm"
echo "  单臂动作: 1arm_circles, waving_1arm"  
echo "  复杂动作: taichi, boxing, dancing, stretching, random"
echo "  静止姿态: pose_arms_forward, pose_t_shape, pose_arms_up"
echo "========================================="
