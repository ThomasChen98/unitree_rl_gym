#!/bin/bash

# Script to run the hybrid control deployment
# Usage: ./run_hybrid_deploy.sh

echo "Starting H1_2 hybrid control deployment..."
echo "Lower body: Policy-controlled (12 DOF)"
echo "Upper body: Trajectory-controlled (15 DOF)"
echo ""

# Check if config file exists
CONFIG_FILE="h1_2_hybrid.yaml"
if [ ! -f "configs/$CONFIG_FILE" ]; then
    echo "Error: Configuration file configs/$CONFIG_FILE not found!"
    exit 1
fi

# Run the deployment
cd "$(dirname "$0")"
python deploy_mujoco3.py $CONFIG_FILE
