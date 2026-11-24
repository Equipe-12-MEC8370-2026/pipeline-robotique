"""Wrist camera mount of the real SO-101 (P4) and the wrist camera pose derived from it. Pure numpy (no Isaac import).

The mount transform comes from ``scripts/sim/fit_wrist_mount.py`` (geometry only: the 2 M3 holes of the mount tab on
the only M3 pair with the same 8.1 mm spacing on the gripper body, tab face on the outer wall; see that script). It is
stored in ``assets/props/camera_mount_v2_fit.json``.

Mount frame (= STL frame, metres): plate between z = 0 (flat face, the camera board rests on it) and z = 0.003
(counterbored face, screw heads); lens opening centred at (0, 0.00375); tab at x ~ -0.020.

Camera (Waveshare IMX335 5MP USB Camera (B), 175 deg diagonal fisheye, M12 lens): optical centre on the opening axis,
``LENS_CENTRE_Z`` above the flat face. That height is an ESTIMATE (no measurement of the lens yet): it is randomised
during data generation (``WRIST_JITTER``) so the policy does not depend on it.
The simulated wrist camera stays a pinhole (90 deg horizontal); on the real robot the fisheye image is undistorted to
the same pinhole before the policy sees it (P4 plan, 2026-11-24).
Image "up" = toward the wrist (tab side). The real board orientation is not known: if the real image is rotated, a
fixed k x 90 deg rotation is applied on the real side, not here.
"""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np

from cta_learn import REPO_ROOT

FIT_JSON = REPO_ROOT / "assets" / "props" / "camera_mount_v2_fit.json"
MOUNT_USD = REPO_ROOT / "assets" / "props" / "camera_mount_v2.usd"
OPENING_CENTRE_MOUNT = (0.0, 0.00375)  # CameraMount_v2_cadquery.py: circle r 7.49 mm at (0, 3.75) mm
LENS_CENTRE_Z = 0.012  # ESTIMATE: optical centre 12 mm above the flat face (board 0, plate 3 mm, lens ~9 mm out)
# per-episode randomisation of the wrist camera around its nominal pose (data generation only)
WRIST_JITTER = {"pos_m": 0.003, "along_axis_m": 0.004, "rot_deg": 3.0}


@lru_cache(maxsize=1)
def mount_fit() -> dict:
    return json.loads(FIT_JSON.read_text(encoding="utf-8"))


def mount_transform() -> np.ndarray:
    """4x4 transform mount frame (m) -> gripper link frame."""
    return np.array(mount_fit()["T_mount_to_link"], dtype=float)


def mat_to_quat_xyzw(R: np.ndarray) -> tuple[float, float, float, float]:
    """Rotation matrix -> unit quaternion (x, y, z, w), w >= 0."""
    m = np.asarray(R, dtype=float)
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k]) * 2
        q = [0.0, 0.0, 0.0, 0.0]
        q[i] = 0.25 * s
        q[j] = (m[j, i] + m[i, j]) / s
        q[k] = (m[k, i] + m[i, k]) / s
        q[3] = (m[k, j] - m[j, k]) / s
    q = np.array(q)
    q /= np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    return tuple(float(v) for v in q)


def wrist_camera_frame(lens_z: float = LENS_CENTRE_Z) -> tuple[np.ndarray, np.ndarray]:
    """Nominal wrist camera in the gripper link frame: (position (3,), rotation (3, 3)) in the ROS convention
    (columns = camera +X right, +Y down, +Z forward)."""
    T = mount_transform()
    R = T[:3, :3]
    pos = T[:3, :3] @ np.array([*OPENING_CENTRE_MOUNT, lens_z]) + T[:3, 3]
    fwd = R @ np.array([0.0, 0.0, 1.0])
    up = R @ np.array([-1.0, 0.0, 0.0])  # toward the tab / wrist
    up -= np.dot(up, fwd) * fwd
    up /= np.linalg.norm(up)
    down = -up
    right = np.cross(down, fwd)
    return pos, np.c_[right, down, fwd]


def wrist_camera_offset(lens_z: float = LENS_CENTRE_Z) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """(pos, quat xyzw) for ``TiledCameraCfg.OffsetCfg(..., convention="ros")`` under the gripper link."""
    pos, R = wrist_camera_frame(lens_z)
    return tuple(float(v) for v in pos), mat_to_quat_xyzw(R)
