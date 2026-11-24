"""M2 — metric poses of box-like parts on the table from a single fixed view (TPV) + its calibration.

A part is approximated by its bounding box of sizes a >= b >= c (any three; a profile is (L, s, s)). Resting on
the table, one of its axes is vertical (``up_axis``: 0 = longest, 1 = middle, 2 = shortest; equal sizes give the
same posture, so a profile has two: lying or upright). The two other sizes are horizontal, longest along the
local x at ``yaw`` (period 180 deg, 90 deg when they are equal). The base is at height z_base (0 on the table, the
top of a support when stacked). The pose is found by maximising the IoU between the observed mask and the
silhouette of the box projected with the calibrated camera (convex hull of the 8 corners), coarse-to-fine grid
search. A box cannot tell the two ends of an asymmetric part apart: that needs its real shape (CAD).
Table frame of data/calib/tpv.json: origin front-right corner, x back, y left, z up. Pure numpy + OpenCV.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from cta_learn.video.calib import project


@dataclass
class BoxPose:
    x: float
    y: float
    yaw_deg: float
    upright: bool               # the longest size is vertical
    z_base: float
    iou: float
    up_axis: int = -1           # which of the sorted sizes (0 longest .. 2 shortest) is vertical
    extents_m: tuple = ()       # (along local x, along local y, vertical) at this posture
    yaw_period_deg: float = 180.0

    def to_dict(self) -> dict:
        return asdict(self)


def box_extents(section: float, length: float, upright: bool) -> np.ndarray:
    """Profile (section s x s, length L): extents lying or upright."""
    return np.array([section, section, length]) if upright else np.array([length, section, section])


def sorted_size(size_m) -> np.ndarray:
    """Box sizes, longest first."""
    d = np.sort(np.asarray(size_m, float))[::-1]
    if d.shape != (3,) or d.min() <= 0:
        raise ValueError("size_m must be 3 positive numbers")
    return d


def distinct_up_axes(size_m, tol: float = 1e-6) -> list[int]:
    """Resting postures that look different (equal sizes give the same posture), shortest-vertical first.
    Profile (L, s, s) -> [2 (lying), 0 (upright)]; a box with three different sizes -> [2, 1, 0]."""
    d = sorted_size(size_m)
    out: list[int] = []
    for i in (2, 1, 0):
        if not any(abs(d[i] - d[j]) < tol for j in out):
            out.append(i)
    return out


def posture_extents(size_m, up_axis: int) -> np.ndarray:
    """(ex, ey, ez): the vertical size is sorted_size[up_axis], the other two horizontal, longest along x."""
    d = sorted_size(size_m)
    hor = [d[i] for i in range(3) if i != up_axis]
    return np.array([max(hor), min(hor), d[up_axis]])


def yaw_period(ext, tol: float = 1e-6) -> float:
    """Yaw symmetry of a box seen from above: 90 deg for a square footprint, else 180 deg."""
    return 90.0 if abs(ext[0] - ext[1]) < tol else 180.0


def is_upright(size_m, up_axis: int, tol: float = 1e-6) -> bool:
    d = sorted_size(size_m)
    return bool(up_axis == 0 and d[0] - d[1] > tol)


def box_corners(ext: np.ndarray, x: float, y: float, yaw_deg: float, z_base: float) -> np.ndarray:
    """8 corners (table frame) of a box of extents (ex, ey, ez), centred at (x, y), yaw about z, base at z_base."""
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    hx, hy = ext[0] / 2, ext[1] / 2
    local = np.array([[sx * hx, sy * hy] for sx in (-1, 1) for sy in (-1, 1)])
    xy = local @ np.array([[c, s], [-s, c]]) + [x, y]
    return np.vstack([np.c_[xy, np.full(4, z_base)], np.c_[xy, np.full(4, z_base + ext[2])]])


def silhouette(K, R, t, corners: np.ndarray, shape: tuple[int, int], offset=(0, 0)) -> np.ndarray:
    """Filled convex hull of the projected corners, in an image of ``shape`` whose (0, 0) is ``offset`` px."""
    p = project(K, R, t, corners) - np.asarray(offset, float)
    hull = cv2.convexHull(np.round(p).astype(np.int32))
    m = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(m, hull, 1)
    return m.astype(bool)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def fit_box(mask: np.ndarray, K, R, t, size_m, init_xy, *, up_axes=None, z_bases=(0.0,),
            xy_window: float = 0.08, margin_px: int = 40) -> BoxPose:
    """Best (x, y, yaw, up_axis, z_base) for ``mask`` (bool, image of K's size), box of sizes ``size_m`` (any
    order). init_xy: rough table point. up_axes: postures tried (default: all distinct ones)."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError("empty mask")
    x0, y0 = max(xs.min() - margin_px, 0), max(ys.min() - margin_px, 0)
    x1, y1 = min(xs.max() + margin_px, mask.shape[1]), min(ys.max() + margin_px, mask.shape[0])
    crop = mask[y0:y1, x0:x1]
    shp, off = crop.shape, (x0, y0)
    axes = distinct_up_axes(size_m) if up_axes is None else list(up_axes)
    exts = {u: posture_extents(size_m, u) for u in axes}

    def score(px, py, yaw, up, zb):
        return iou(crop, silhouette(K, R, t, box_corners(exts[up], px, py, yaw, zb), shp, off))

    best = (-1.0, None)
    # coarse: 1 cm, 10 deg over one yaw period of the posture
    for up in axes:
        for zb in z_bases:
            yaws = np.arange(0, yaw_period(exts[up]), 10.0)
            for px in init_xy[0] + np.arange(-xy_window, xy_window + 1e-9, 0.01):
                for py in init_xy[1] + np.arange(-xy_window, xy_window + 1e-9, 0.01):
                    for yaw in yaws:
                        s = score(px, py, yaw, up, zb)
                        if s > best[0]:
                            best = (s, (px, py, yaw, up, zb))
    # refine twice: ±1.5 cm / ±10 deg, then ±0.4 cm / ±2.5 deg
    for step_xy, span_xy, step_yaw, span_yaw in ((0.0025, 0.015, 2.0, 10.0), (0.001, 0.004, 0.5, 2.5)):
        s0, (bx, by, byaw, up, zb) = best
        for px in bx + np.arange(-span_xy, span_xy + 1e-9, step_xy):
            for py in by + np.arange(-span_xy, span_xy + 1e-9, step_xy):
                for yaw in byaw + np.arange(-span_yaw, span_yaw + 1e-9, step_yaw):
                    s = score(px, py, yaw, up, zb)
                    if s > best[0]:
                        best = (s, (px, py, yaw, up, zb))
    s, (bx, by, byaw, up, zb) = best
    per = yaw_period(exts[up])
    return BoxPose(float(bx), float(by), float(byaw % per), is_upright(size_m, up), float(zb), float(s),
                   int(up), tuple(float(v) for v in exts[up]), per)


def fit_box_pose(mask: np.ndarray, K, R, t, section: float, length: float, init_xy, *,
                 postures=(False, True), z_bases=(0.0,), xy_window: float = 0.08, margin_px: int = 40) -> BoxPose:
    """Profile version of :func:`fit_box` (section s x s, length L; postures False = lying, True = upright)."""
    return fit_box(mask, K, R, t, (length, section, section), init_xy, up_axes=[0 if p else 2 for p in postures],
                   z_bases=z_bases, xy_window=xy_window, margin_px=margin_px)


def rest_segments(centroids: np.ndarray, fps: float, speed_px: float = 1.0, min_s: float = 0.5,
                  smooth: int = 5) -> list[tuple[int, int]]:
    """Frame ranges [a, b] where the (smoothed) mask centroid moves less than ``speed_px`` per frame for at
    least ``min_s`` seconds. centroids (T, 2), NaN when hidden (never at rest)."""
    c = np.asarray(centroids, float)
    T = len(c)
    if T < 2:
        return []
    k = np.ones(smooth) / smooth
    h = smooth // 2
    cs = np.column_stack([np.convolve(np.pad(np.nan_to_num(c[:, i], nan=1e9), (h, smooth - 1 - h), mode="edge"), k, mode="valid")
                          for i in range(2)])
    d = np.linalg.norm(np.diff(cs, axis=0), axis=1)
    sp = np.r_[d[0], d]  # frame 0 takes the speed of the first step
    sp[np.isnan(c).any(1)] = np.inf
    still = sp < speed_px
    out, a = [], None
    for i, s in enumerate(np.r_[still, False]):
        if s and a is None:
            a = i
        elif not s and a is not None:
            if (i - a) / fps >= min_s:
                out.append((a, i - 1))
            a = None
    return out


def contact_frames(mask_seq: np.ndarray, hand_kp_px: np.ndarray, dilate_px: int = 6) -> np.ndarray:
    """(T,) bool: some hand keypoint lies on the object's mask dilated by ``dilate_px``.
    mask_seq (T, H, W) bool; hand_kp_px (T, n_hands, 21, 2) in the SAME pixel scale, NaN when absent."""
    ker = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8)
    T, H, W = mask_seq.shape
    out = np.zeros(T, bool)
    for i in range(T):
        kp = hand_kp_px[i].reshape(-1, 2)
        kp = kp[~np.isnan(kp[:, 0])]
        if kp.size == 0 or not mask_seq[i].any():
            continue
        m = cv2.dilate(mask_seq[i].astype(np.uint8), ker).astype(bool)
        xi, yi = np.clip(kp[:, 0].astype(int), 0, W - 1), np.clip(kp[:, 1].astype(int), 0, H - 1)
        out[i] = bool(m[yi, xi].any())
    return out


def same_pose(p: dict, q: dict, tol_m: float = 0.01, tol_deg: float = 5.0) -> bool:
    """Same rest pose (posture, base, position, yaw modulo the box symmetry). Pose dicts of object_trajectories;
    ``up_axis`` / ``yaw_period_deg`` when present, else the profile convention (upright -> 90 deg)."""
    if p["upright"] != q["upright"] or p.get("up_axis") != q.get("up_axis") or abs(p["z_base"] - q["z_base"]) > 1e-6:
        return False
    per = p.get("yaw_period_deg", 90.0 if p["upright"] else 180.0)
    dyaw = abs((p["yaw_deg"] - q["yaw_deg"] + per / 2) % per - per / 2)
    return bool(np.hypot(p["x"] - q["x"], p["y"] - q["y"]) < tol_m and dyaw < tol_deg)


def inside_footprint(x: float, y: float, o: dict, margin: float = 0.0) -> bool:
    """Is (x, y) over the rectangle footprint of support ``o`` (centre, yaw, extents), with a margin?"""
    c, s = np.cos(np.radians(o["yaw_deg"])), np.sin(np.radians(o["yaw_deg"]))
    dx, dy = x - o["x"], y - o["y"]
    u, v = c * dx + s * dy, -s * dx + c * dy
    return abs(u) <= o["ext"][0] / 2 + margin and abs(v) <= o["ext"][1] / 2 + margin
