"""P2.4d: box pose from a mask (synthetic camera), rest segments, hand contact."""
import numpy as np
import pytest

from cta_learn.perception.poses import box_corners, box_extents, contact_frames, fit_box_pose, rest_segments, silhouette
from cta_learn.video.calib import intrinsics


def _cam():
    """Camera ~0.75 m above the table, looking at the table frame origin area (like the TPV, half resolution)."""
    K = intrinsics(935.0, 960, 540)
    C = np.array([-0.45, -0.35, 0.75])
    target = np.array([0.25, 0.25, 0.0])
    fwd = target - C; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    Rct = np.vstack([right, down, fwd])  # table -> camera rotation (rows = camera axes)
    return K, Rct, -Rct @ C


@pytest.mark.parametrize("up,zb,yaw", [(False, 0.0, 30.0), (True, 0.0, 20.0), (True, 0.04, 65.0), (False, 0.0, 120.0)])
def test_fit_recovers_known_pose(up, zb, yaw):
    K, R, t = _cam()
    s, L = 0.04, 0.15 if up else 0.08
    true = (0.22, 0.31, yaw)
    mask = silhouette(K, R, t, box_corners(box_extents(s, L, up), *true, zb), (540, 960))
    p = fit_box_pose(mask, K, R, t, s, L, init_xy=(0.25, 0.27), postures=(False, True), z_bases=(0.0, 0.04))
    assert p.upright == up and p.z_base == zb
    assert abs(p.x - true[0]) < 0.004 and abs(p.y - true[1]) < 0.004
    period = 90 if up else 180
    d = abs((p.yaw_deg - yaw + period / 2) % period - period / 2)
    assert d < 2.0 and p.iou > 0.95


def test_rest_segments():
    fps = 15
    c = np.zeros((60, 2))
    c[:20] = [100, 100]                                     # still 0-19
    c[20:35] = np.c_[100 + 10 * np.arange(1, 16), np.full(15, 100)]  # moving fast
    c[35:] = [250, 100]                                     # still 35-59
    c[50:53] = np.nan                                       # hidden 50-52 -> breaks the rest
    seg = rest_segments(c, fps, speed_px=1.0, min_s=0.5)
    assert seg[0][0] == 0 and 15 <= seg[0][1] <= 20
    assert any(a >= 35 and b <= 49 for a, b in seg) and all(not (a <= 51 <= b) for a, b in seg)


def test_contact_frames():
    T = 3
    m = np.zeros((T, 50, 50), bool)
    m[:, 20:30, 20:30] = True
    kp = np.full((T, 2, 21, 2), np.nan)
    kp[0, 0, 0] = [25, 25]   # on the object
    kp[1, 1, 3] = [33, 25]   # 3 px outside, within the dilation
    kp[2, 0, 0] = [45, 45]   # far
    assert contact_frames(m, kp, dilate_px=6).tolist() == [True, True, False]


def test_same_pose_and_footprint():
    from cta_learn.perception.poses import inside_footprint, same_pose
    p = {"x": 0.15, "y": 0.16, "yaw_deg": 89.0, "upright": False, "z_base": 0.0}
    assert same_pose(p, dict(p, x=0.155, yaw_deg=91.0))
    assert same_pose(p, dict(p, yaw_deg=269.0 - 180.0 + 1.0))       # yaw is modulo 180 deg when lying
    assert not same_pose(p, dict(p, x=0.17))
    assert not same_pose(p, dict(p, upright=True))
    sup = {"x": 0.155, "y": 0.157, "yaw_deg": 90.0, "ext": [0.08, 0.04]}  # 8 cm along y, 4 cm along x
    assert inside_footprint(0.16, 0.155, sup)
    assert inside_footprint(0.155, 0.19, sup) and not inside_footprint(0.19, 0.157, sup)
    assert inside_footprint(0.18, 0.157, sup, margin=0.01)


def test_postures_of_a_box():
    from cta_learn.perception.poses import distinct_up_axes, posture_extents, yaw_period
    assert distinct_up_axes((0.04, 0.08, 0.04)) == [2, 0]            # profile: lying, upright
    assert distinct_up_axes((0.05, 0.07, 0.06)) == [2, 1, 0]         # three different sizes
    assert distinct_up_axes((0.05, 0.05, 0.05)) == [2]               # cube: one posture
    assert posture_extents((0.05, 0.07, 0.06), 1).tolist() == [0.07, 0.05, 0.06]
    assert yaw_period(posture_extents((0.04, 0.08, 0.04), 0)) == 90.0
    assert yaw_period(posture_extents((0.05, 0.07, 0.06), 0)) == 180.0


@pytest.mark.parametrize("up,yaw", [(0, 25.0), (1, 70.0), (2, 140.0)])
def test_fit_box_three_sizes(up, yaw):
    from cta_learn.perception.poses import fit_box, posture_extents
    K, R, t = _cam()
    size = (0.045, 0.07, 0.055)
    true = (0.22, 0.31, yaw)
    mask = silhouette(K, R, t, box_corners(posture_extents(size, up), *true, 0.0), (540, 960))
    p = fit_box(mask, K, R, t, size, init_xy=(0.25, 0.27))
    assert p.up_axis == up and p.yaw_period_deg == 180.0 and p.upright == (up == 0)
    assert abs(p.x - true[0]) < 0.004 and abs(p.y - true[1]) < 0.004
    d = abs((p.yaw_deg - yaw + 90) % 180 - 90)
    assert d < 2.0 and p.iou > 0.95
