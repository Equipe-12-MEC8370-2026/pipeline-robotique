"""P3.1: TaskSpec -> SO-101 layout (D10, D11) and the stack success predicate (numpy path)."""
import math

import numpy as np
import pytest

from cta_learn import CONFIGS_DIR
from cta_learn.core.taskspec import TaskSpec
from cta_learn.retarget.profile_stack import rotate_xyzw, so101_layout, stack_success

SPEC = CONFIGS_DIR / "taskspecs" / "cta_profile_upright_113656.json"


def _quat_z(yaw):   # XYZW
    return np.array([[0, 0, math.sin(yaw / 2), math.cos(yaw / 2)]])


def _quat_y(a):
    return np.array([[0, math.sin(a / 2), 0, math.cos(a / 2)]])


def test_layout_applies_d10_d11_and_keeps_relations():
    lay = so101_layout(TaskSpec.load(SPEC))
    so, lo = lay["objects"]["profile_short"], lay["objects"]["profile_long"]
    assert so["start_posture"] == "lying" and so["size_m"] == [0.05, 0.025, 0.025]
    assert lo["start_posture"] == "upright" and lo["size_m"] == [0.025, 0.025, 0.075]
    assert any(d.startswith("D11") for d in lay["deviations"]) and any(d.startswith("D10") for d in lay["deviations"])
    g = lay["goals"]["profile_long"]
    assert g["relation"] == "on_top_of" and g["ref"] == "profile_short"
    assert all(abs(v) < 0.005 for v in g["offset_xy_m"])          # demo offset (mm) scaled by 25/40
    assert lay["instruction"]["en"].startswith("Lay the short")
    assert lay["order"] == ["profile_short", "profile_long"]
    # starts do not overlap even at the range limits (footprints <= 50 mm)
    s, l_ = np.array(so["start_xyyaw"][:2]), np.array(lo["start_xyyaw"][:2])
    worst = np.linalg.norm(s - l_) - np.hypot(so["range"][0] + lo["range"][0], so["range"][1] + lo["range"][1])
    assert worst > 0.045
    # and both stay clear of the gripper's rest position at the start pose (measured in sim)
    tcp_rest = np.array([0.195, 0.021])
    for o in (so, lo):
        clear = np.linalg.norm(np.array(o["start_xyyaw"][:2]) - tcp_rest) - np.hypot(o["range"][0], o["range"][1])
        assert clear > 0.05


def test_rotate_xyzw():
    v = np.array([[1.0, 0, 0]])
    assert np.allclose(rotate_xyzw(_quat_z(math.pi / 2), v, np), [[0, 1, 0]], atol=1e-9)
    assert np.allclose(rotate_xyzw(_quat_y(math.pi / 2), np.array([[0, 0, 1.0]]), np), [[1, 0, 0]], atol=1e-9)


def _state(lay, **kw):
    sec, L = 0.025, lay["objects"]["profile_long"]["length_m"]
    st = dict(short_pos=np.array([[0.21, 0.10, sec / 2]]), short_quat=_quat_z(0.3),
              long_pos=np.array([[0.212, 0.101, sec + L / 2]]), long_quat=_quat_z(0.1),
              tcp_pos=np.array([[0.21, 0.10, 0.25]]), goal_xy=np.array([[0.21, 0.10]]))
    st.update(kw)
    return st


@pytest.mark.parametrize("change,failed", [
    ({}, None),
    ({"long_quat": _quat_y(math.radians(30))}, "long_upright"),
    ({"long_pos": np.array([[0.24, 0.10, 0.025 + 0.0375]])}, "long_on_short"),
    ({"long_pos": np.array([[0.30, 0.10, 0.0375]])}, "long_on_short"),
    ({"short_quat": _quat_y(math.radians(90))}, "short_lying"),
    ({"short_pos": np.array([[0.25, 0.10, 0.0125]])}, "short_in_zone"),
    ({"tcp_pos": np.array([[0.21, 0.10, 0.09]])}, "released"),
])
def test_stack_success(change, failed):
    lay = so101_layout(TaskSpec.load(SPEC))
    st = _state(lay, **change)
    if failed == "short_in_zone":  # keep the long part on top of the moved short part
        st["long_pos"] = np.array([[0.25, 0.10, 0.025 + 0.0375]])
    ok, checks = stack_success(xp=np, layout=lay, **st)
    if failed is None:
        assert ok[0] and all(v[0] for v in checks.values())
    else:
        assert not ok[0] and not checks[failed][0]
