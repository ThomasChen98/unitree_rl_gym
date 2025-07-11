import copy

# 姿态设置（仅调整关键link的quat）
pose_defs = {
    "torso_link_default": {},  # 原始不变
    "torso_link_arms_down": {
        # 双臂自然下垂（elbow向后摆，肩保持基本姿态）
        "left_elbow_pitch_link": {"quat": "0.707 0 0 0.707"},
        "right_elbow_pitch_link": {"quat": "0.707 0 0 -0.707"},
    },
    "torso_link_elbow_bent": {
        # 肩略抬起，肘上弯
        "left_shoulder_pitch_link": {"quat": "0.9239 0 0 0.3827"},  # 45°
        "left_elbow_pitch_link": {"quat": "0.707 0 0 0.707"},
        "right_shoulder_pitch_link": {"quat": "0.9239 0 0 -0.3827"},
        "right_elbow_pitch_link": {"quat": "0.707 0 0 -0.707"},
    },
    "torso_link_open_wide": {
        # 手臂展开到两侧，roll控制
        "left_shoulder_roll_link": {"quat": "0.707 0.707 0 0"},
        "right_shoulder_roll_link": {"quat": "0.707 -0.707 0 0"},
    },
    "torso_link_salute": {
        # 右臂抬高并弯曲至头部
        "right_shoulder_pitch_link": {"quat": "0.9239 0 0 -0.3827"},
        "right_elbow_pitch_link": {"quat": "0.707 0 0 -0.707"},
        "right_wrist_pitch_link": {"quat": "0.707 0 0 -0.707"},
    },
    "torso_link_forward_hold": {
        # 双臂前平举
        "left_shoulder_pitch_link": {"quat": "0.707 0 0 0.707"},
        "right_shoulder_pitch_link": {"quat": "0.707 0 0 -0.707"},
        "left_elbow_pitch_link": {"quat": "0.9239 0 0 0.3827"},
        "right_elbow_pitch_link": {"quat": "0.9239 0 0 -0.3827"},
    }
}

# 生成多个 torso_link 变体
torso_variants = []

for pose_name, mod_dict in pose_defs.items():
    torso_copy = copy.deepcopy(original_torso)
    torso_copy.attrib["name"] = pose_name

    # 遍历所有子 body，若命中则修改 quat
    for body in torso_copy.iter("body"):
        bname = body.attrib.get("name")
        if bname in mod_dict:
            new_quat = mod_dict[bname]["quat"]
            body.attrib["quat"] = new_quat

    torso_variants.append(torso_copy)

# 将多个 variant 写入一个新的 XML 片段
output_root = ET.Element("upper_body_poses")
for variant in torso_variants:
    output_root.append(variant)

output_path = "/home/yuxin/unitree_rl_gym/resources/robots/h1_2/h1_2_12dof_alt_pose.xml"
ET.ElementTree(output_root).write(output_path, encoding="utf-8", xml_declaration=True)

output_path
