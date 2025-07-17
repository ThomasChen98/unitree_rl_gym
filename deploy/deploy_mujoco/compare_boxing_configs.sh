#!/bin/bash

# 对比测试脚本：原始 vs 优化的拳击配置
# 用于观察胳膊晃动问题的改善

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_SCRIPT="$SCRIPT_DIR/deploy_mujoco3.py"

echo "==== H1_2 拳击动作稳定性对比测试 ===="
echo ""

echo "即将进行两个测试："
echo "1. 使用原始配置（可能有胳膊晃动）"
echo "2. 使用优化配置（应该更稳定）"
echo ""

read -p "按 Enter 开始第一个测试（原始配置）..."

echo ""
echo "=== 测试 1: 原始配置 ==="
echo "预期现象: 胳膊出拳后可能会晃动，不够稳定"
echo "观察要点: 注意胳膊在出拳保持阶段的稳定性"
echo ""

cd "$SCRIPT_DIR"
timeout 15s python3 "$DEPLOY_SCRIPT" --config configs/h1_2_hybrid.yaml --trajectory boxing

echo ""
echo "原始配置测试完成"
echo ""

read -p "按 Enter 开始第二个测试（优化配置）..."

echo ""
echo "=== 测试 2: 优化配置 ==="
echo "预期改善: 胳膊出拳后更稳定，晃动明显减少"
echo "优化内容: 平滑过渡 + 增强PD参数 + 调整角度范围"
echo ""

timeout 15s python3 "$DEPLOY_SCRIPT" --config configs/h1_2_boxing.yaml --trajectory boxing

echo ""
echo "优化配置测试完成"
echo ""

echo "==== 对比测试结束 ===="
echo ""
echo "改善要点："
echo "✓ 平滑过渡减少突然的角度跳跃"
echo "✓ 更高的PD增益提供更快的响应"
echo "✓ 增强的阻尼快速消除振荡"
echo "✓ 调整的角度范围更安全可控"
echo ""
echo "如果效果满意，建议使用优化配置进行拳击动作演示"
