import math

import pytest

from cta_learn.core import embodiment_from_dict, load_embodiment

SO101_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def test_so101_loads_with_expected_order_and_cameras():
    emb = load_embodiment("so101")
    assert emb.name == "so101"
    assert emb.joint_names == SO101_ORDER
    assert emb.num_joints == 6
    assert emb.camera_names == ["front", "wrist"]
    assert emb.control_hz == 30


def test_so101_limits_are_radians_from_usd_degrees():
    emb = load_embodiment("so101")
    elbow = emb.joints[2]
    assert elbow.lower == pytest.approx(math.radians(-100.0))
    assert elbow.upper == pytest.approx(math.radians(90.0))
    assert emb.joints[5].kind == "gripper"
    assert emb.joints[5].lerobot_norm == "0_100"


def _arm(name):
    return {
        "name": name,
        "ee_body": "gripper",
        "joints": [
            {"name": "j1", "kind": "revolute", "limits_deg": [-90, 90], "lerobot_norm": "m100_100"},
            {"name": "grip", "kind": "gripper", "limits_deg": [0, 90], "lerobot_norm": "0_100"},
        ],
    }


def test_bimanual_joint_names_are_prefixed_per_arm():
    emb = embodiment_from_dict({"name": "bi", "control_hz": 30, "arms": [_arm("left"), _arm("right")]})
    assert emb.joint_names == ["left.j1", "left.grip", "right.j1", "right.grip"]
    assert emb.num_joints == 4


def test_invalid_limits_are_rejected():
    raw = {"name": "bad", "control_hz": 30, "arms": [_arm("main")]}
    raw["arms"][0]["joints"][0]["limits_deg"] = [10, -10]
    with pytest.raises(ValueError, match="lower"):
        embodiment_from_dict(raw)


def test_duplicate_camera_names_are_rejected():
    raw = {
        "name": "dup",
        "control_hz": 30,
        "arms": [_arm("main")],
        "cameras": [
            {"name": "front", "width": 640, "height": 480, "fps": 30, "mount": "world"},
            {"name": "front", "width": 640, "height": 480, "fps": 30, "mount": "world"},
        ],
    }
    with pytest.raises(ValueError, match="duplicate camera"):
        embodiment_from_dict(raw)
