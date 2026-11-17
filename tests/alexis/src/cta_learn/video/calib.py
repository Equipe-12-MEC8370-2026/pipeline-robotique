"""M1 — metric calibration of the fixed TPV camera against the table plane (no checkerboard).

Model: pinhole, square pixels, principal point at the image centre, no lens distortion (checked by the
straightness of the table edges). Steps:
  1. table edges refined from rough hand-placed segments (gradient peak search along the normal);
  2. two vanishing points of orthogonal table directions -> focal length and table orientation;
  3. metric scale from one known length on the table (tape measure: hook at the front edge -> case);
  4. image <-> table homography (metres). A second camera set-up is mapped onto the first through the
     plane-induced homography between the two views (exact for points on the table).
Table frame: origin = front-right table corner, x = along the right edge towards the back, y = along the
front edge towards the left, z = up (x × y). All functions are pure numpy/opencv, no I/O.
"""
from __future__ import annotations

import numpy as np


def hom(p) -> np.ndarray:
    p = np.asarray(p, float)
    return np.append(p, 1.0) if p.shape == (2,) else np.hstack([p, np.ones((len(p), 1))])


def refine_line(gray: np.ndarray, p0, p1, band: float = 8.0, n: int = 60, trim: float = 0.1) -> tuple[np.ndarray, float]:
    """Refine a rough segment p0->p1 onto the strongest straight edge nearby.

    Samples ``n`` points along the segment (ends trimmed), finds the max gradient along the normal within
    ±band px (sub-pixel parabola), fits a line by total least squares, drops 20 % worst points and refits.
    Returns (homogeneous line l with l·x = 0 and |l[:2]| = 1, rms residual in px).
    """
    g = gray.astype(np.float32)
    gy, gx = np.gradient(g)
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = (p1 - p0) / np.linalg.norm(p1 - p0)
    nrm = np.array([-d[1], d[0]])
    offs = np.arange(-band, band + 1)
    pts = []
    for t in np.linspace(trim, 1 - trim, n):
        c = p0 + t * (p1 - p0)
        xy = c[None, :] + offs[:, None] * nrm[None, :]
        xi, yi = np.clip(np.round(xy[:, 0]).astype(int), 0, g.shape[1] - 1), np.clip(np.round(xy[:, 1]).astype(int), 0, g.shape[0] - 1)
        mag = np.abs(gx[yi, xi] * nrm[0] + gy[yi, xi] * nrm[1])
        k = int(np.argmax(mag))
        if 0 < k < len(offs) - 1:
            y0, y1, y2 = mag[k - 1], mag[k], mag[k + 1]
            den = y0 - 2 * y1 + y2
            k = k + (0.5 * (y0 - y2) / den if den else 0.0)
        pts.append(c + (k - band) * nrm)
    pts = np.asarray(pts)
    for _ in range(2):
        m = pts.mean(0)
        _, _, vt = np.linalg.svd(pts - m)
        nn = vt[1]
        r = (pts - m) @ nn
        keep = np.abs(r) <= np.quantile(np.abs(r), 0.8)
        pts = pts[keep] if keep.sum() >= 5 else pts
    m = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - m)
    nn = vt[1]
    l = np.array([nn[0], nn[1], -nn @ m])
    rms = float(np.sqrt(np.mean(((pts - m) @ nn) ** 2)))
    return l, rms


def line_through(p0, p1) -> np.ndarray:
    l = np.cross(hom(p0), hom(p1))
    return l / np.linalg.norm(l[:2])


def vanishing_point(lines: list[np.ndarray]) -> np.ndarray:
    """Least-squares common point of >= 2 homogeneous lines (homogeneous, may be near infinity)."""
    L = np.asarray(lines, float)
    _, _, vt = np.linalg.svd(L)
    return vt[-1]


def focal_from_vps(v1: np.ndarray, v2: np.ndarray, c) -> float:
    """Focal (px) from two vanishing points of orthogonal directions, principal point c, square pixels."""
    a, b = v1[:2] / v1[2] - c, v2[:2] / v2[2] - c
    f2 = -float(a @ b)
    if f2 <= 0:
        raise ValueError("vanishing points inconsistent with orthogonal directions (f^2 <= 0)")
    return float(np.sqrt(f2))


def _rodrigues(w: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


def line_angle_residuals(f: float, R: np.ndarray, c, lines: dict[str, list[tuple[np.ndarray, np.ndarray]]]) -> np.ndarray:
    """sin(angle) between each observed line and the direction to its predicted vanishing point.

    ``lines`` = {"x"|"y"|"z": [(homogeneous line, image point on it), ...]}; R columns = table x, y, z axes
    in camera coordinates.
    """
    K = intrinsics(f, 2 * c[0], 2 * c[1])
    out = []
    for a, items in lines.items():
        v = K @ R[:, "xyz".index(a)]
        for l, m in items:
            d = v[:2] - np.asarray(m, float) * v[2]
            d /= np.linalg.norm(d)
            dobs = np.array([l[1], -l[0]]) / np.linalg.norm(l[:2])
            out.append(dobs[0] * d[1] - dobs[1] * d[0])
    return np.asarray(out)


def fit_orientation(lines: dict[str, list[tuple[np.ndarray, np.ndarray]]], c, f0: float, R0: np.ndarray,
                    iters: int = 50, fix_f: bool = False) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares focal + table orientation from lines of the 3 axis families (Levenberg–Marquardt).

    Returns (f, R, residuals_deg). R is kept a proper rotation (R0 @ exp([w]x)). ``fix_f`` keeps f = f0.
    """
    p = np.zeros(4)  # [log f/f0, w]
    free = np.array([not fix_f, True, True, True])

    def unpack(p):
        return f0 * np.exp(p[0]), R0 @ _rodrigues(p[1:])

    def res(p):
        f, R = unpack(p)
        return line_angle_residuals(f, R, c, lines)

    lam = 1e-3
    r = res(p)
    for _ in range(iters):
        J = np.zeros((len(r), 4))
        for i in np.flatnonzero(free):
            dp = np.zeros(4)
            dp[i] = 1e-6
            J[:, i] = (res(p + dp) - r) / 1e-6
        Jf = J[:, free]
        A, g = Jf.T @ Jf, Jf.T @ r
        step = np.zeros(4)
        step[free] = -np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), g)
        r_new = res(p + step)
        if r_new @ r_new < r @ r:
            p, r, lam = p + step, r_new, lam * 0.3
            if np.linalg.norm(step) < 1e-10:
                break
        else:
            lam *= 10
    f, R = unpack(p)
    return f, R, np.degrees(np.arcsin(np.clip(r, -1, 1)))


def intrinsics(f: float, w: int, h: int) -> np.ndarray:
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])


def plane_pose(K: np.ndarray, v_x: np.ndarray, v_y: np.ndarray, origin_px, dir_x_px, dir_y_px) -> tuple[np.ndarray, np.ndarray]:
    """Rotation of the table frame in the camera frame (columns x, y, z) + plane normal for unit distance.

    ``dir_*_px`` = an image point a little along +x / +y from the origin, to fix the sign of each axis.
    Returns (R, t1): table point (X, Y, 0) metres is at camera point t1 + R[:, :2] @ [X, Y] / scale, with
    t1 = the origin on the plane scaled to unit camera-plane distance; ``scale`` (metres per unit) comes from
    a known length (see img_to_plane_fn).
    """
    Ki = np.linalg.inv(K)
    rx, ry = Ki @ v_x, Ki @ v_y
    rx /= np.linalg.norm(rx)
    ry = ry / np.linalg.norm(ry)
    ry = ry - (ry @ rx) * rx  # re-orthogonalise (noise)
    ry /= np.linalg.norm(ry)
    n = np.cross(rx, ry)
    o = Ki @ hom(origin_px)
    if n @ o > 0:  # make n point from the table toward the camera ("up"): n·X = -1 on the plane
        n = -n
    t1 = -o / (n @ o)

    def on_plane(px):
        r = Ki @ hom(px)
        return -r / (n @ r)

    # signs: +x / +y must point from the origin toward the given image points
    if (on_plane(dir_x_px) - t1) @ rx < 0:
        rx = -rx
    if (on_plane(dir_y_px) - t1) @ ry < 0:
        ry = -ry
    rz = np.cross(rx, ry)
    if rz @ n < 0:
        raise ValueError("x, y, up is left-handed with these axis points: swap the +x / +y image points")
    R = np.column_stack([rx, ry, rz])
    t1 = -o / (rz @ o)
    return R, t1


def img_to_plane_fn(K: np.ndarray, R: np.ndarray, t1: np.ndarray, scale: float):
    """Image px (N,2) -> table (X, Y) metres (N,2): intersect pixel rays with the plane."""
    Ki = np.linalg.inv(K)
    n = R[:, 2]

    def f(px):
        rays = (Ki @ hom(np.atleast_2d(px)).T).T
        P = -rays / (rays @ n)[:, None]  # points on the plane n·X = -1 (n = table "up")
        return ((P - t1) @ R[:, :2]) * scale

    return f


def plane_homography(K: np.ndarray, R: np.ndarray, t1: np.ndarray, scale: float) -> np.ndarray:
    """3x3 H mapping table (X, Y, 1) metres -> image px (homogeneous). Inverse maps image -> table."""
    return K @ np.column_stack([R[:, 0] / scale, R[:, 1] / scale, t1])


def pose_from_homography(K: np.ndarray, H_table_to_img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Camera pose from the table->image homography: X_cam = R @ X_table + t (metres, z up)."""
    M = np.linalg.inv(K) @ H_table_to_img
    lam = 1.0 / np.linalg.norm(M[:, 0])
    if (M[:, 2] * lam)[2] < 0:  # table in front of the camera
        lam = -lam
    r1, r2, t = M[:, 0] * lam, M[:, 1] * lam, M[:, 2] * lam
    R = np.column_stack([r1, r2, np.cross(r1, r2)])
    u, _, vt = np.linalg.svd(R)  # nearest rotation
    return u @ vt, t


def project(K: np.ndarray, R: np.ndarray, t: np.ndarray, X) -> np.ndarray:
    Xc = np.atleast_2d(X) @ R.T + t
    p = Xc @ K.T
    return p[:, :2] / p[:, 2:3]


def ray_to_height(K: np.ndarray, R: np.ndarray, t: np.ndarray, px, z: float) -> np.ndarray:
    """Table-frame point (x, y, z) seen at pixel ``px`` on the horizontal plane at height ``z``."""
    C = -R.T @ t  # camera centre in the table frame
    d = R.T @ (np.linalg.inv(K) @ np.append(np.asarray(px, float), 1.0))
    s = (z - C[2]) / d[2]
    return C + s * d


def height_above(K: np.ndarray, R: np.ndarray, t: np.ndarray, base_px, z_base: float, top_px) -> float:
    """Height of the point seen at ``top_px`` vertically above the point at ``base_px`` (height z_base)."""
    B = ray_to_height(K, R, t, base_px, z_base)
    zs = np.linspace(z_base, z_base + 2.0, 20001)
    P = project(K, R, t, np.c_[np.full_like(zs, B[0]), np.full_like(zs, B[1]), zs])
    return float(zs[np.argmin(np.linalg.norm(P - np.asarray(top_px, float), axis=1))])


def apply_h(H: np.ndarray, pts) -> np.ndarray:
    p = hom(np.atleast_2d(pts)) @ H.T
    return p[:, :2] / p[:, 2:3]
