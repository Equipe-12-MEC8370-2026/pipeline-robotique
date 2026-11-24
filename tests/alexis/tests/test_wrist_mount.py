"""P4 wrist camera mount: fitted transform and camera pose derived from it (pure numpy, no Isaac)."""

import numpy as np
import pytest

from cta_learn.sim import wrist_mount as W


def _quat_to_mat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def test_transform_is_rigid():
    T = W.mount_transform()
    R = T[:3, :3]
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_tab_holes_land_on_gripper_holes():
    fit = W.mount_fit()
    T = W.mount_transform()
    wall_axis = "xyz".index(fit["wall_axis"])
    for tab, grip in zip(fit["tab_holes_mount"], fit["gripper_holes_link"]):
        p = T[:3, :3] @ np.array(tab) + T[:3, 3]
        # same hole axis: compare across the wall (the hole centres sit at different depths along it)
        assert np.linalg.norm(np.delete(p - np.array(grip), wall_axis)) < 2e-4
    assert abs(fit["gripper_hole_spacing_m"] - 0.0081) < 4e-4
    assert fit["chosen"]["max_depth_m"] < 5e-4  # the mount does not enter the gripper


def test_camera_on_opening_axis_looking_out_of_the_plate():
    T = W.mount_transform()
    pos, R = W.wrist_camera_frame()
    back = np.linalg.inv(T) @ np.r_[pos, 1.0]
    assert np.allclose(back[:2], W.OPENING_CENTRE_MOUNT, atol=1e-9)
    assert np.isclose(back[2], W.LENS_CENTRE_Z)
    assert np.allclose(R[:, 2], T[:3, :3] @ [0, 0, 1])  # optical axis = plate normal, counterbored side
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-9) and np.isclose(np.linalg.det(R), 1.0)


def test_camera_looks_toward_the_fingertips():
    pos, R = W.wrist_camera_frame()
    tip = np.array([0.012, 0.0, -0.1044])  # TCP x, fixed fingertip z (pick_place.py)
    v = tip - pos
    assert np.degrees(np.arccos(v @ R[:, 2] / np.linalg.norm(v))) < 30  # well inside a 90 deg view


@pytest.mark.parametrize("seed", range(5))
def test_quaternion_matches_matrix(seed):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    q2 = W.mat_to_quat_xyzw(_quat_to_mat(q))
    assert np.allclose(_quat_to_mat(q2), _quat_to_mat(q), atol=1e-9)
    pos, quat = W.wrist_camera_offset()
    assert np.allclose(_quat_to_mat(quat), W.wrist_camera_frame()[1], atol=1e-9)
