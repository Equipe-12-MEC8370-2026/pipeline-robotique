import math

import numpy as np
import pytest

from cta_learn.datagen.pick_place_expert import (
    PHASES,
    READY_POSE_DEG,
    ExpertParams,
    PickPlaceExpert,
    wrap_to_quarter_turn,
    yaw_from_quat_xyzw,
)
from cta_learn.core.geometry import quat_from_euler_deg

DT = 1 / 30


def _run_ideal(expert, cube, target, tcp0, max_steps=2000):
    """Ideal tracking: the TCP lands on the command; the cube follows the TCP once grasped."""
    tcp = np.array(tcp0, dtype=float)
    cube = np.array(cube, dtype=float)
    q = np.zeros(5)
    expert.reset(tcp, cube, 0.3, arm_q=q)
    phases, cmds = [], []
    for _ in range(max_steps):
        cmd = expert.step(tcp, cube, target, DT, arm_q=q)
        if cmd.joint_goal is not None:
            q = cmd.joint_goal  # ideal joint tracking
            phases.append(cmd.phase)
            continue
        if expert.phase in ("lift", "transport", "lower"):
            cube = np.array([*cmd.tcp_goal[:2], max(cmd.tcp_goal[2], 0.015)])
        tcp = cmd.tcp_goal
        phases.append(cmd.phase)
        cmds.append(cmd)
        if expert.done or expert.failed:
            break
    return phases, cmds, cube


def test_phases_run_in_order_and_finish():
    phases, _, _ = _run_ideal(PickPlaceExpert(), [0.22, -0.06, 0.015], np.array([0.22, 0.12, 0.0]), [0.25, 0.0, 0.2])
    seen = list(dict.fromkeys(phases))
    assert seen == list(PHASES)


def test_cube_ends_on_target_with_ideal_tracking():
    target = np.array([0.20, 0.10, 0.0])
    _, _, cube = _run_ideal(PickPlaceExpert(), [0.25, -0.10, 0.015], target, [0.25, 0.0, 0.2])
    assert np.linalg.norm(cube[:2] - target[:2]) < 0.005


def test_command_speed_is_bounded():
    p = ExpertParams(speed=0.12)
    _, cmds, _ = _run_ideal(PickPlaceExpert(params=p), [0.22, -0.06, 0.015], np.array([0.22, 0.12, 0.0]), [0.25, 0.0, 0.2])
    steps = np.diff(np.stack([c.tcp_goal for c in cmds if c.joint_goal is None]), axis=0)
    assert np.linalg.norm(steps, axis=1).max() <= 0.12 * DT + 1e-9


def test_gripper_open_before_grasp_closed_while_carrying():
    _, cmds, _ = _run_ideal(PickPlaceExpert(), [0.22, -0.06, 0.015], np.array([0.22, 0.12, 0.0]), [0.25, 0.0, 0.2])
    by_phase = {}
    for c in cmds:
        by_phase.setdefault(c.phase, set()).add(round(c.gripper, 6))
    assert by_phase["descend"] == {round(math.radians(35), 6)}
    assert by_phase["transport"] == {round(math.radians(-10), 6)}


def test_stuck_tcp_triggers_failure():
    ex = PickPlaceExpert()
    ex.reset([0.25, 0, 0.2], [0.22, -0.06, 0.015], 0.0, arm_q=np.radians(READY_POSE_DEG))
    for _ in range(400):
        ex.step([0.25, 0, 0.2], [0.22, -0.06, 0.015], [0.22, 0.12, 0], DT, arm_q=np.radians(READY_POSE_DEG))
    assert ex.failed and "approach" in ex.failure_reason


@pytest.mark.parametrize("yaw_deg,expected", [(0, 0), (50, -40), (-50, 40), (90, 0), (44, 44), (135, 45 - 90)])
def test_yaw_wrapped_to_quarter_turn(yaw_deg, expected):
    assert math.degrees(wrap_to_quarter_turn(math.radians(yaw_deg))) == pytest.approx(expected)


def test_yaw_from_quat_matches_euler_helper():
    assert yaw_from_quat_xyzw(quat_from_euler_deg(0, 0, 37)) == pytest.approx(math.radians(37))


def test_sampled_params_stay_in_ranges():
    rng = np.random.default_rng(0)
    for _ in range(100):
        p = ExpertParams.sample(rng)
        assert 0.04 <= p.approach_height <= 0.06 and 0.10 <= p.speed <= 0.18
        assert p.lift_height <= 0.08 and 0.075 <= p.retreat_height <= 0.09
        # the success term needs the TCP > 5 cm from the placed cube's centre (z = 0.015)
        assert p.retreat_height - 0.015 > 0.05 + 0.005
        assert 0.010 <= p.grasp_height <= 0.014


def test_ready_phase_moves_joints_at_bounded_speed_then_hands_over():
    ex = PickPlaceExpert()
    q = np.radians([0.0, -95.0, 85.0, 50.0, 0.0])  # folded start pose
    ex.reset([0.17, 0.02, 0.06], [0.22, -0.06, 0.015], 0.0, arm_q=q)
    prev, n = q.copy(), 0
    while ex.phase == "ready" and n < 1000:
        cmd = ex.step([0.17, 0.02, 0.06], [0.22, -0.06, 0.015], [0.22, 0.12, 0], DT, arm_q=prev)
        assert np.max(np.abs(cmd.joint_goal - prev)) <= 0.5 * DT + 1e-9
        prev, n = cmd.joint_goal, n + 1
    assert ex.phase == "approach" and not ex.failed
    np.testing.assert_allclose(np.degrees(prev), READY_POSE_DEG, atol=3.0)


def _run_with_misses(expert, n_miss, cube=(0.22, -0.06, 0.015), target=(0.22, 0.12, 0.0), max_steps=4000):
    """Ideal tracking, but the first ``n_miss`` grasps do not pick the cube up (it stays on the table)."""
    tcp, cube, target = np.array([0.25, 0.0, 0.2]), np.array(cube, float), np.array(target, float)
    q = np.zeros(5)
    expert.reset(tcp, cube, 0.3, arm_q=q)
    close_goals, phases = [], []
    for _ in range(max_steps):
        cmd = expert.step(tcp, cube, target, DT, arm_q=q, cube_yaw=0.1)
        phases.append(cmd.phase)
        if cmd.joint_goal is not None:
            q = cmd.joint_goal
            continue
        if cmd.phase == "close":
            close_goals.append(cmd.tcp_goal[:2].copy())
        if expert.phase in ("lift", "transport", "lower") and expert.regrasps >= n_miss:
            cube = np.array([*cmd.tcp_goal[:2], max(cmd.tcp_goal[2], 0.015)])
        tcp = cmd.tcp_goal
        if expert.done or expert.failed:
            break
    return phases, close_goals, cube


def test_missed_grasp_is_detected_and_retried():
    ex = PickPlaceExpert(params=ExpertParams(first_grasp_offset_xy=(0.02, 0.0)))
    phases, close_goals, cube = _run_with_misses(ex, n_miss=1)
    assert ex.regrasps == 1 and ex.done and not ex.failed
    seen = [ph for i, ph in enumerate(phases) if i == 0 or ph != phases[i - 1]]
    assert seen[:6] == ["ready", "approach", "descend", "close", "lift", "approach"]
    assert np.linalg.norm(cube[:2] - [0.22, 0.12]) < 0.005
    # the injected offset applies to the first grasp only
    assert np.allclose(close_goals[0], [0.24, -0.06]) and np.allclose(close_goals[-1], [0.22, -0.06])
    assert ex._yaw == pytest.approx(0.1)  # yaw re-measured for the retry


def test_too_many_missed_grasps_fail():
    ex = PickPlaceExpert(params=ExpertParams(max_regrasps=2))
    _run_with_misses(ex, n_miss=99)
    assert ex.failed and ex.failure_reason == "grasp missed too many times" and ex.regrasps == 2
