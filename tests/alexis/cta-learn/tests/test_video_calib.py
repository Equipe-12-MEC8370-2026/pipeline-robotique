"""P2.3: table-plane calibration from vanishing points + one known length, on a synthetic camera."""
import cv2
import numpy as np
import pytest

from cta_learn.video.calib import (apply_h, focal_from_vps, img_to_plane_fn, intrinsics, line_through,
                                   plane_homography, plane_pose, refine_line, vanishing_point)

W, H = 1920, 1080


def _camera(f=1250.0):
    """Camera 1.2 m above a table, looking down ~50 deg, yawed 30 deg; table frame x back, y left, z up."""
    K = intrinsics(f, W, H)
    yaw, pitch = np.radians(30), np.radians(50)
    # camera axes in the table frame: forward (z_c), right (x_c), down (y_c)
    fwd = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), -np.sin(pitch)])
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R_tc = np.column_stack([right, down, fwd])  # camera -> table
    C = np.array([-0.6, -0.4, 1.2])             # camera centre in the table frame
    def project(P):
        Pc = (np.atleast_2d(P) - C) @ R_tc       # table -> camera coords
        p = Pc @ K.T
        return p[:, :2] / p[:, 2:3]
    return K, project


def _pipeline(project, f_true=None):
    # table 1.0 (x, depth) by 1.5 (y, width); origin = front-right corner
    corners = np.array([[0, 0, 0], [1.0, 0, 0], [1.0, 1.5, 0], [0, 1.5, 0]], float)
    c = project(corners)
    lx = [line_through(c[0], c[1]), line_through(c[3], c[2])]   # edges along x
    ly = [line_through(c[0], c[3]), line_through(c[1], c[2])]   # edges along y
    vx, vy = vanishing_point(lx), vanishing_point(ly)
    f = focal_from_vps(vx, vy, np.array([W / 2, H / 2]))
    K = intrinsics(f, W, H)
    R, t1 = plane_pose(K, vx, vy, c[0], project([[0.1, 0, 0]])[0], project([[0, 0.1, 0]])[0])
    # known length: tape from (0, 0.3) to (0.491, 0.3)
    tape = project([[0, 0.3, 0], [0.491, 0.3, 0]])
    unit = img_to_plane_fn(K, R, t1, 1.0)(tape)
    scale = 0.491 / np.linalg.norm(unit[1] - unit[0])
    return f, img_to_plane_fn(K, R, t1, scale), plane_homography(K, R, t1, scale)


def test_recovers_focal_and_metric_points():
    K_true, project = _camera(1250.0)
    f, to_plane, Hm = _pipeline(project)
    assert f == pytest.approx(1250.0, rel=1e-6)
    P = np.array([[0.25, 0.4, 0], [0.8, 1.2, 0], [0.05, 0.02, 0]])
    xy = to_plane(project(P))
    assert np.allclose(xy, P[:, :2], atol=1e-6)
    assert np.allclose(apply_h(Hm, P[:, :2]), project(P), atol=1e-6)  # metres -> px
    assert np.allclose(apply_h(np.linalg.inv(Hm), project(P)), P[:, :2], atol=1e-6)


def test_axis_signs_follow_the_given_points():
    _, project = _camera()
    _, to_plane, _ = _pipeline(project)
    xy = to_plane(project([[0.5, 0.0, 0], [0.0, 0.5, 0]]))
    assert xy[0, 0] > 0.4 and xy[1, 1] > 0.4


def test_refine_line_snaps_to_edge():
    img = np.full((400, 600), 60, np.uint8)
    pts = np.array([[0, 100], [600, 300], [600, 400], [0, 400]], np.int32)  # bright region below a slanted edge
    cv2.fillPoly(img, [pts], 200)
    img = cv2.GaussianBlur(img, (5, 5), 1.0)
    l, rms = refine_line(img, (50, 125 + 6), (550, 291 - 5), band=10)  # rough, ~6 px off
    true = line_through((0, 100), (600, 300))
    if np.sign(l[0]) != np.sign(true[0]):
        l = -l
    for x in (100.0, 500.0):
        y_fit = -(l[0] * x + l[2]) / l[1]
        assert abs(y_fit - (100 + x / 3)) < 1.0  # sub-pixel; fillPoly's pixel convention shifts ~0.5 px
    assert rms < 0.5


def test_pose_from_homography_and_heights():
    from cta_learn.video.calib import height_above, pose_from_homography, project, ray_to_height
    _, proj = _camera(1250.0)
    _, _, Hm = _pipeline(proj)
    K = intrinsics(1250.0, W, H)
    R, t = pose_from_homography(K, Hm)
    P = np.array([[0.3, 0.5, 0.0], [0.3, 0.5, 0.2], [0.7, 0.1, 0.05]])
    assert np.allclose(project(K, R, t, P), proj(P), atol=1e-4)            # 3D points, not only the plane
    assert np.allclose(ray_to_height(K, R, t, proj(P[2:])[0], 0.05), P[2], atol=1e-6)
    assert height_above(K, R, t, proj(P[:1])[0], 0.0, proj(P[1:2])[0]) == pytest.approx(0.2, abs=2e-4)
    assert (-R.T @ t)[2] == pytest.approx(1.2, abs=1e-6)                   # camera height above the table


def test_fit_orientation_recovers_focal():
    from cta_learn.video.calib import fit_orientation, line_angle_residuals
    _, proj = _camera(1250.0)
    seg = {"x": [((0, 0.2, 0), (0.9, 0.2, 0)), ((0, 1.0, 0), (0.9, 1.0, 0))],
           "y": [((0.1, 0, 0), (0.1, 1.4, 0)), ((0.8, 0, 0), (0.8, 1.4, 0))],
           "z": [((0.9, 0.3, 0), (0.9, 0.3, 0.5)), ((0.9, 1.2, 0), (0.9, 1.2, 0.5))]}
    lines = {}
    for a, ss in seg.items():
        lines[a] = []
        for p0, p1 in ss:
            q = proj(np.array([p0, p1], float))
            lines[a].append((line_through(q[0], q[1]), q.mean(0)))
    c = np.array([W / 2, H / 2])
    # start from a wrong focal and a slightly wrong orientation
    from cta_learn.video.calib import _rodrigues
    K = intrinsics(1250.0, W, H)
    R_true = np.column_stack([np.linalg.inv(K) @ vanishing_point([l for l, _ in lines[a]]) for a in "xyz"])
    R_true /= np.linalg.norm(R_true, axis=0)
    R_true = np.linalg.svd(R_true)[0] @ np.linalg.svd(R_true)[2]
    f, R, rdeg = fit_orientation(lines, c, 1500.0, R_true @ _rodrigues(np.array([0.02, -0.01, 0.015])))
    assert f == pytest.approx(1250.0, rel=2e-3) and np.abs(rdeg).max() < 0.01


def test_inconsistent_vanishing_points_raise():
    with pytest.raises(ValueError):
        focal_from_vps(np.array([2000, 540, 1.0]), np.array([3000, 540, 1.0]), np.array([960, 540]))
