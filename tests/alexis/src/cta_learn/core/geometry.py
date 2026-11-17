"""Rotation helpers. Quaternions are ALWAYS (x, y, z, w), the Isaac Lab 3.0 convention.

Isaac Lab 3.0 switched from WXYZ to XYZW. Code written for 2.x (e.g. LeIsaac) uses WXYZ;
copying such literals silently rotates cameras and robots. In cta_learn, no quaternion
is written by hand: configs give readable Euler angles in degrees and these helpers
produce the quaternion.
"""

from __future__ import annotations

import math

import numpy as np


def quat_from_euler_deg(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Intrinsic-free XYZ Euler angles (degrees, applied roll about X, then pitch about Y,
    then yaw about Z in the fixed frame, i.e. R = Rz(yaw) @ Ry(pitch) @ Rx(roll)) -> (x, y, z, w).

    Same convention as ``isaaclab.utils.math.quat_from_euler_xyz``.
    """
    r, p, y = (math.radians(a) / 2.0 for a in (roll, pitch, yaw))
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    yv = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return (x, yv, z, w)


def quat_to_matrix(q_xyzw) -> np.ndarray:
    """(x, y, z, w) -> 3x3 rotation matrix."""
    x, y, z, w = np.asarray(q_xyzw, dtype=np.float64) / np.linalg.norm(q_xyzw)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def look_at_quat(eye, target, up=(0.0, 0.0, 1.0)) -> tuple[float, float, float, float]:
    """Quaternion (x, y, z, w) of a camera at ``eye`` looking at ``target``, in the *world*
    camera convention (forward = +X, up = +Z), i.e. Isaac Lab ``convention="world"``.
    """
    eye, target, up = (np.asarray(v, dtype=np.float64) for v in (eye, target, up))
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    left = np.cross(up, fwd)
    if np.linalg.norm(left) < 1e-9:
        raise ValueError("look_at: forward direction is parallel to 'up'")
    left /= np.linalg.norm(left)
    true_up = np.cross(fwd, left)
    m = np.stack([fwd, left, true_up], axis=1)  # columns = body X, Y, Z in world
    return matrix_to_quat(m)


def matrix_to_quat(m) -> tuple[float, float, float, float]:
    """3x3 rotation matrix -> (x, y, z, w), w >= 0."""
    m = np.asarray(m, dtype=np.float64)
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    q /= np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    return tuple(float(v) for v in q)
