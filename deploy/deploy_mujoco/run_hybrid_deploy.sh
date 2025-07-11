#!/bin/bash

# Script to run the hybrid control deployment with trajectory selection
# Usage: ./run_hybrid_deploy.sh [trajectory_type]
# Available trajectories: circles, waving, taichi, boxing, dancing, stretching

# Default trajectory if not specified
TRAJECTORY=${1:-circles}

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
    circles|waving|taichi|boxing|dancing|stretching)
        echo "Valid trajectory selected: $TRAJECTORY"
        ;;
    *)
        echo "Warning: Unknown trajectory '$TRAJECTORY', using default 'circles'"
        echo "Available trajectories: circles, waving, taichi, boxing, dancing, stretching"
        TRAJECTORY="circles"
        ;;
esac

echo ""
echo "Running deployment..."
echo "=========================="

# Run the deployment
cd "$(dirname "$0")"
python deploy_mujoco3.py $CONFIG_FILE --trajectory $TRAJECTORY
