"""M3 — temporal segmentation of a demonstration from the hands (and, when available, the object moves).

Start/end gesture: the demonstrator shows both open hands before and after the task. Detected when BOTH views
(TPV and FPV, synced clips at the same rate) see two open hands for >= ``min_s`` seconds. Hand openness = mean
over the 4 long fingers of |tip - wrist| / |MCP - wrist| (about 1 for a fist, about 2 for a flat hand).
Pure numpy.
"""
from __future__ import annotations

import numpy as np

FINGERS = ((8, 5), (12, 9), (16, 13), (20, 17))  # (tip, MCP) of index, middle, ring, little finger


def openness(kp: np.ndarray) -> float:
    """kp (21, 2) px -> openness ratio (NaN if the hand is missing)."""
    if np.isnan(kp[0, 0]):
        return float("nan")
    w = kp[0]
    return float(np.mean([np.linalg.norm(kp[t] - w) / max(np.linalg.norm(kp[m] - w), 1e-6) for t, m in FINGERS]))


def both_open(kp_px: np.ndarray, thr: float = 1.6) -> np.ndarray:
    """kp_px (T, 2, 21, 2) -> (T,) bool: two hands present and both open."""
    op = np.array([[openness(kp_px[i, s]) for s in range(kp_px.shape[1])] for i in range(len(kp_px))])
    with np.errstate(invalid="ignore"):
        return (~np.isnan(op)).all(1) & (np.nan_to_num(op, nan=0.0).min(1) > thr)


def runs(flag: np.ndarray, fps: float, smooth_s: float = 0.5, min_s: float = 1.5, max_gap_s: float = 0.5) -> list[tuple[float, float]]:
    """Time intervals [a, b] (s) where the smoothed flag is mostly on, gaps < max_gap_s bridged, >= min_s long."""
    k = max(1, int(round(smooth_s * fps)))
    f = np.convolve(flag.astype(float), np.ones(k) / k, mode="same") > 0.5
    iv, a = [], None
    for i, v in enumerate(np.r_[f, False]):
        if v and a is None:
            a = i
        elif not v and a is not None:
            iv.append([a, i - 1])
            a = None
    merged = []
    for a, b in iv:
        if merged and (a - merged[-1][1]) / fps <= max_gap_s:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(round(a / fps, 2), round((b + 1) / fps, 2)) for a, b in merged if (b + 1 - a) / fps >= min_s]


def start_end_signals(tpv_kp: np.ndarray, fpv_kp: np.ndarray, fps: float, min_s: float = 1.0, edge_frac: float = 0.4,
                      **kw) -> dict[str, tuple[float, float] | None]:
    """Start/end gestures: intervals where at least one view sees both hands open (the wearer does not always
    look at their hands; the end gesture is often short). Start = first such interval beginning in the first
    ``edge_frac`` of the clip, end = last one ending in the last ``edge_frac``; None when absent."""
    n = min(len(tpv_kp), len(fpv_kp))
    bt, bf = both_open(tpv_kp[:n]), both_open(fpv_kp[:n])
    dur = n / fps
    out = {"start": None, "end": None}
    # strict (both views agree) first: tighter bounds; one view only as a fallback
    for flag in (bt & bf, bt | bf):
        r = runs(flag, fps, min_s=min_s, **kw)
        starts = [iv for iv in r if iv[0] <= edge_frac * dur]
        ends = [iv for iv in r if iv[1] >= (1 - edge_frac) * dur and (not starts or iv != starts[0])]
        if out["start"] is None and starts:
            out["start"] = starts[0]
        if out["end"] is None and ends:
            out["end"] = ends[-1]
    return out


def hand_side_fpv(fpv_kp: np.ndarray, t_s: np.ndarray, t0: float, t1: float, width: int = 1920) -> tuple[str, float]:
    """Which demonstrator hand works during [t0, t1], from the egocentric view (left hand on the image left).

    The working hand = the slot whose wrist moves most in the interval; side from its wrist x vs image centre.
    Returns (side, confidence = fraction of its frames on that side); ("unknown", 0) without data.
    """
    m = (t_s >= t0) & (t_s <= t1)
    best, best_motion = None, -1.0
    for s in range(fpv_kp.shape[1]):
        w = fpv_kp[m, s, 0]
        w = w[~np.isnan(w[:, 0])]
        if len(w) < 5:
            continue
        motion = float(np.linalg.norm(np.diff(w, axis=0), axis=1).sum())
        if motion > best_motion:
            best, best_motion = w, motion
    if best is None:
        return "unknown", 0.0
    right = best[:, 0] > width / 2
    side = "right" if right.mean() >= 0.5 else "left"
    return side, float(max(right.mean(), 1 - right.mean()))
