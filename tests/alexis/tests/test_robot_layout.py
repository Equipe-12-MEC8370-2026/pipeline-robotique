"""Generic SO-101 adaptation (retarget.layout): reproduces the published profile layout; generic success."""
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from cta_learn.core.taskspec import TaskSpec
from cta_learn.retarget.layout import posture_ok, robot_layout, task_success
from cta_learn.retarget.profile_stack import stack_success

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs"


def _profile():
    spec = TaskSpec.load(CFG / "taskspecs" / "cta_profile_upright_113656.json")
    adapt = yaml.safe_load((CFG / "tasks" / "so101_adapt_cta_profile_upright.yaml").read_text(encoding="utf-8"))
    return robot_layout(spec, adapt), yaml.safe_load((CFG / "tasks" / "so101_profile_stack.yaml").read_text(encoding="utf-8"))


def test_profile_layout_matches_published_p3():
    new, old = _profile()
    assert new["order"] == old["order"] and new["instruction"] == old["instruction"]
    for p in ("profile_short", "profile_long"):
        for k in ("size_m", "mass_kg", "start_posture", "start_xyyaw", "range", "range_ood", "demo_size_m"):
            assert new["objects"][p][k] == old["objects"][p][k], (p, k)
    gs, gl = new["goals"]["profile_short"], new["goals"]["profile_long"]
    og, ol = old["goals"]["profile_short"], old["goals"]["profile_long"]
    assert gs["xyyaw"] == og["xyyaw"] and gs["tol_xy_m"] == og["tol_xy_m"] and gs["posture"] == "lying"
    assert gl["offset_xy_m"] == ol["offset_xy_m"] and gl["yaw_rel_deg"] == ol["yaw_rel_deg"]
    assert gl["tol_xy_m"] == ol["tol_xy_m"] and gl["tol_posture_deg"] == ol["tol_upright_deg"] and gl["ref"] == "profile_short"


def test_block_layout_defaults():
    spec = TaskSpec.load(CFG / "taskspecs" / "cta_angle_block_114110.json")
    lay = robot_layout(spec, {})
    o, g = lay["objects"]["angle_block"], lay["goals"]["angle_block"]
    assert min(o["box_m"]) == pytest.approx(0.025) and o["start_posture"] == "lying" and g["posture"] == "lying"
    assert g["relation"] == "on_table" and g["xyyaw"] == [0.21, 0.10, 0.0]
    assert any("unknown" in d for d in lay["deviations"])


def _q(axis, deg):
    a = np.radians(deg) / 2
    v = np.array(axis, float) * np.sin(a)
    return np.array([[*v, np.cos(a)]])


def _state(dx_long=0.0, tilt_long=0.0, short_zone_dx=0.0, tcp=(0.0, 0.3, 0.2), roll_short=0.0):
    lay, _ = _profile()
    gx, gy = lay["goals"]["profile_short"]["xyyaw"][:2]
    ox, oy = lay["goals"]["profile_long"]["offset_xy_m"]
    ps = np.array([[gx + short_zone_dx, gy, 0.0125]])
    pl = np.array([[ps[0, 0] + ox + dx_long, gy + oy, 0.025 + 0.0375]])
    pos = {"profile_short": ps, "profile_long": pl}
    quat = {"profile_short": _q((1, 0, 0), roll_short), "profile_long": _q((1, 0, 0), tilt_long)}
    return lay, pos, quat, np.array([tcp], float), np.array([[gx, gy]])


@pytest.mark.parametrize("kw,expected", [
    ({}, True), ({"roll_short": 90.0}, True), ({"dx_long": 0.03}, False), ({"tilt_long": 30.0}, False),
    ({"short_zone_dx": 0.03}, False), ({"tcp": (0.21, 0.10, 0.08)}, False)])
def test_generic_success_agrees_with_p3_on_clear_cases(kw, expected):
    lay, pos, quat, tcp, zone = _state(**kw)
    ok, _ = task_success(pos, quat, tcp, {"profile_short": zone}, lay, np)
    old_lay = yaml.safe_load((CFG / "tasks" / "so101_profile_stack.yaml").read_text(encoding="utf-8"))
    ok_old, _ = stack_success(pos["profile_short"], quat["profile_short"], pos["profile_long"], quat["profile_long"],
                              tcp, zone, old_lay, np)
    assert bool(ok[0]) == bool(ok_old[0]) == expected


def test_posture_rules():
    q0 = _q((0, 0, 1), 0)
    assert posture_ok(q0, [0.05, 0.025, 0.025], 0.025, 20, np)[0]                       # lying bar
    assert posture_ok(_q((1, 0, 0), 90), [0.05, 0.025, 0.025], 0.025, 20, np)[0]        # rolled onto another face
    assert not posture_ok(_q((0, 1, 0), 90), [0.05, 0.025, 0.025], 0.025, 20, np)[0]    # stood up
    assert posture_ok(q0, [0.0286, 0.0278, 0.025], 0.025, 15, np)[0]                    # block flat
    assert not posture_ok(_q((1, 0, 0), 90), [0.0286, 0.0278, 0.025], 0.025, 15, np)[0]  # block on its side
    assert math.isclose(min([0.0286, 0.0278, 0.025]), 0.025)


def test_start_posture_follows_goal_without_override():
    """A new task (no adaptation config): the short profile starts upright in the demo, ends lying -> starts lying."""
    spec = TaskSpec.load(CFG / "taskspecs" / "cta_profile_upright_113656.json")
    lay = robot_layout(spec, {})
    assert lay["objects"]["profile_short"]["start_posture"] == "lying"
    assert lay["objects"]["profile_long"]["start_posture"] == "upright"
    assert any("as at arrival" in d for d in lay["deviations"])
