"""Gesture library: the generic pick_place_box gives the P3 plans for the profiles, and plans the angle block."""
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import yaml

from cta_learn.datagen.gestures import GestureMissing, pick_place_box, plan_task
from cta_learn.datagen.stack_expert import StackExpert, StackParams, plan_from_layout, rot2
from cta_learn.retarget.layout import task_success

CFG = Path(__file__).resolve().parents[1] / "configs" / "tasks"


def load(name):
    return yaml.safe_load((CFG / name).read_text(encoding="utf-8"))


def test_profile_plans_equal_p3():
    new = plan_task(load("so101_cta_profile_upright.yaml"))
    old = plan_from_layout(load("so101_profile_stack.yaml"))
    for a, b in zip(new, old):
        da, db = asdict(a), asdict(b)
        for k in da:
            if isinstance(da[k], float):
                assert da[k] == pytest.approx(db[k], abs=1e-9), k
            else:
                assert da[k] == db[k], k


def test_block_plan():
    (p,) = plan_task(load("so101_cta_angle_block.yaml"))
    assert p.support is None and p.slide == 0.0 and p.symmetry == math.pi
    assert p.grip_yaw == math.pi / 2          # closing across the narrower side (27.8 mm, local y)
    assert p.grasp_z == pytest.approx(0.012) and p.height == pytest.approx(0.025)


def test_missing_gestures():
    lay = load("so101_cta_angle_block.yaml")
    lay["goals"]["angle_block"]["posture"] = "upright"
    with pytest.raises(GestureMissing, match="turning a part over"):
        plan_task(lay)
    lay = load("so101_cta_angle_block.yaml")
    lay["objects"]["angle_block"]["size_m"] = [0.06, 0.05, 0.025]
    with pytest.raises(GestureMissing, match="gripper"):
        plan_task(lay)


def run_ideal(expert, parts, max_steps=4000):
    """Ideal tracking (as tests/test_stack_expert.py): a grasped part follows the TCP and turns with the gripper."""
    tcp = np.array([0.195, 0.021, 0.071])
    expert.reset(tcp, {k: (v[0], v[1]) for k, v in parts.items()}, StackParams(), arm_q=np.zeros(5))
    held = rel = None
    for _ in range(max_steps):
        obs = {k: (v[0].copy(), v[1], np.array([0.0, 0.0, 1.0])) for k, v in parts.items()}
        cmd = expert.step(tcp, obs, 1 / 30)
        if cmd.joint_goal is not None:
            continue
        tcp = cmd.tcp_goal.copy()
        obj = expert.plan.obj
        if cmd.phase == "close" and held is None:
            held = obj
            rel = (rot2(-cmd.yaw, (parts[obj][0] - tcp)[:2]), parts[obj][0][2] - tcp[2], parts[obj][1] - cmd.yaw)
        if held and cmd.phase in ("release", "clear", "slide", "retreat", "approach"):
            held = None
        if held:
            parts[held][0] = np.r_[tcp[:2] + rot2(cmd.yaw, rel[0]), tcp[2] + rel[1]]
            parts[held][1] = cmd.yaw + rel[2]
        if expert.done or expert.failed:
            break
    return parts, tcp


def test_block_ideal_run_succeeds():
    lay = load("so101_cta_angle_block.yaml")
    o = lay["objects"]["angle_block"]
    rng = np.random.default_rng(0)
    for _ in range(10):
        s = [o["start_xyyaw"][k] + rng.uniform(-o["range"][k], o["range"][k]) for k in range(3)]
        expert = StackExpert(plan_task(lay))
        parts, tcp = run_ideal(expert, {"angle_block": [np.array([s[0], s[1], o["size_m"][2] / 2]), s[2]]})
        assert expert.done and not expert.failed, expert.failure_reason
        p, yaw = parts["angle_block"]
        pos = np.array([[p[0], p[1], o["size_m"][2] / 2]])
        quat = np.array([[0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]])
        ok, checks = task_success({"angle_block": pos}, {"angle_block": quat}, tcp[None],
                                  {"angle_block": np.array([lay["goals"]["angle_block"]["xyyaw"][:2]])}, lay, np)
        assert bool(ok[0]), {k: bool(v[0]) for k, v in checks.items()}
