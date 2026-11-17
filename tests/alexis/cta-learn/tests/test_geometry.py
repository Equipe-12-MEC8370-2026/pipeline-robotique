import math

import numpy as np
import pytest

from cta_learn.core.geometry import look_at_quat, matrix_to_quat, quat_from_euler_deg, quat_to_matrix


def test_identity_is_xyzw_with_w_last():
    assert quat_from_euler_deg(0, 0, 0) == pytest.approx((0.0, 0.0, 0.0, 1.0))


def test_yaw_90_rotates_x_to_y():
    m = quat_to_matrix(quat_from_euler_deg(0, 0, 90))
    np.testing.assert_allclose(m @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    # xyzw: rotation about z puts the sine in the 3rd slot, not the 4th
    q = quat_from_euler_deg(0, 0, 90)
    assert q[2] == pytest.approx(math.sqrt(0.5)) and q[3] == pytest.approx(math.sqrt(0.5))


def test_euler_matches_rz_ry_rx_composition():
    roll, pitch, yaw = 20.0, -35.0, 110.0
    def rx(a):
        a = math.radians(a); return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    def ry(a):
        a = math.radians(a); return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    def rz(a):
        a = math.radians(a); return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    np.testing.assert_allclose(quat_to_matrix(quat_from_euler_deg(roll, pitch, yaw)), rz(yaw) @ ry(pitch) @ rx(roll), atol=1e-12)


def test_matrix_quat_round_trip():
    rng = np.random.default_rng(1)
    for _ in range(50):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        q = q if q[3] >= 0 else -q
        np.testing.assert_allclose(matrix_to_quat(quat_to_matrix(q)), q, atol=1e-9)


def test_look_at_points_body_x_at_target():
    eye, target = np.array([0.5, -0.4, 0.6]), np.array([0.2, 0.1, 0.0])
    m = quat_to_matrix(look_at_quat(eye, target))
    fwd = (target - eye) / np.linalg.norm(target - eye)
    np.testing.assert_allclose(m[:, 0], fwd, atol=1e-12)
    assert m[2, 2] > 0  # camera up keeps a positive world-z component


def test_look_at_rejects_vertical_view():
    with pytest.raises(ValueError):
        look_at_quat([0, 0, 1], [0, 0, 0])
