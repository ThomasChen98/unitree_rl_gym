#!/usr/bin/env python3
"""
Test script to validate the hybrid control configuration
"""

import sys
import os
import yaml
import numpy as np

# Add the parent directory to the path to import legged_gym
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from legged_gym import LEGGED_GYM_ROOT_DIR


def test_config():
    """Test the hybrid configuration file"""
    config_file = "h1_2_hybrid.yaml"
    config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}"

    print(f"Testing configuration: {config_path}")

    try:
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        print("✓ Configuration file loaded successfully")
    except FileNotFoundError:
        print("✗ Configuration file not found")
        return False
    except yaml.YAMLError as e:
        print(f"✗ YAML parsing error: {e}")
        return False

    # Test policy path
    policy_path = config["policy_path"].replace(
        "{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR
    )
    if os.path.exists(policy_path):
        print(f"✓ Policy file found: {policy_path}")
    else:
        print(f"⚠ Policy file not found: {policy_path}")
        print("  This is expected if you haven't trained a policy yet")

    # Test XML path
    xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    if os.path.exists(xml_path):
        print(f"✓ XML file found: {xml_path}")
    else:
        print(f"✗ XML file not found: {xml_path}")
        return False

    # Test configuration parameters
    required_params = [
        "lower_body_kps",
        "lower_body_kds",
        "lower_body_default_angles",
        "upper_body_kps",
        "upper_body_kds",
        "upper_body_default_angles",
        "num_actions",
        "num_obs",
        "simulation_dt",
        "control_decimation",
    ]

    for param in required_params:
        if param in config:
            print(f"✓ Parameter '{param}' found")
        else:
            print(f"✗ Required parameter '{param}' missing")
            return False

    # Test array dimensions
    lower_kps = np.array(config["lower_body_kps"])
    lower_kds = np.array(config["lower_body_kds"])
    lower_defaults = np.array(config["lower_body_default_angles"])
    upper_kps = np.array(config["upper_body_kps"])
    upper_kds = np.array(config["upper_body_kds"])
    upper_defaults = np.array(config["upper_body_default_angles"])

    print(
        f"✓ Lower body dimensions: KP={len(lower_kps)}, KD={len(lower_kds)}, defaults={len(lower_defaults)}"
    )
    print(
        f"✓ Upper body dimensions: KP={len(upper_kps)}, KD={len(upper_kds)}, defaults={len(upper_defaults)}"
    )

    if len(lower_kps) != 12 or len(lower_kds) != 12 or len(lower_defaults) != 12:
        print("✗ Lower body should have 12 DOF")
        return False

    if len(upper_kps) != 15 or len(upper_kds) != 15 or len(upper_defaults) != 15:
        print("✗ Upper body should have 15 DOF")
        return False

    total_dof = len(lower_kps) + len(upper_kps)
    print(f"✓ Total DOF: {total_dof} (12 lower + 15 upper)")

    print("\n✓ All configuration tests passed!")
    return True


def test_imports():
    """Test if required packages can be imported"""
    print("\nTesting imports...")

    try:
        import torch

        print("✓ PyTorch imported successfully")
    except ImportError:
        print("✗ PyTorch not found - required for policy loading")
        return False

    try:
        import mujoco

        print("✓ MuJoCo imported successfully")
    except ImportError:
        print("✗ MuJoCo not found - required for simulation")
        return False

    try:
        import mujoco.viewer

        print("✓ MuJoCo viewer imported successfully")
    except ImportError:
        print("✗ MuJoCo viewer not found - required for visualization")
        return False

    print("✓ All required packages imported successfully!")
    return True


if __name__ == "__main__":
    print("=" * 50)
    print("H1_2 Hybrid Control Configuration Test")
    print("=" * 50)

    config_ok = test_config()
    imports_ok = test_imports()

    print("\n" + "=" * 50)
    if config_ok and imports_ok:
        print("✓ ALL TESTS PASSED - Ready to run hybrid control!")
        print("\nTo run the simulation:")
        print("  cd /home/yuxin/unitree_rl_gym/deploy/deploy_mujoco")
        print("  python deploy_mujoco3.py h1_2_hybrid.yaml")
    else:
        print("✗ Some tests failed - please fix the issues above")
    print("=" * 50)
