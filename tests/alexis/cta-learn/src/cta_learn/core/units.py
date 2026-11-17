"""The ONLY place where joint values change unit.

Canonical unit inside cta_learn: radians, vectors ordered as ``Embodiment.joint_names``.

External formats:

* ``"rad"``     – radians (canonical).
* ``"deg"``     – degrees (old LeRobot calibration, ``main_*`` datasets, USD limits).
* ``"lerobot"`` – LeRobot normalized range: ``[-100, 100]`` for body joints
  (``MotorNormMode.RANGE_M100_100``) and ``[0, 100]`` for the gripper
  (``RANGE_0_100``), mapped linearly onto the embodiment joint limits.

Every conversion validates its input range and raises :class:`UnitRangeError`
instead of silently producing out-of-distribution values. This is the guard
against the failure of the previous SO-101 attempt, where a degrees dataset was
fed with normalized states.

Note: on the real robot, LeRobot normalizes with the per-robot calibration
(min/max recorded at calibration time). Mapping ``"lerobot"`` onto the USD joint
limits is the simulation convention; aligning it with a real calibration file
is part of phase P1.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from cta_learn.core.embodiment import Embodiment

Format = Literal["rad", "deg", "lerobot"]
FORMATS: tuple[str, ...] = ("rad", "deg", "lerobot")

#: Tolerance outside joint limits, as a fraction of each joint's range
#: (simulation can overshoot limits slightly; real encoders are noisy).
DEFAULT_LIMIT_TOL = 0.05


class UnitRangeError(ValueError):
    """Values are not plausible for the declared format (wrong unit or wrong joint order)."""


def _as_array(values, emb: Embodiment) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape[-1] != emb.num_joints:
        raise ValueError(
            f"last dimension is {arr.shape[-1]}, expected {emb.num_joints} joints ({emb.joint_names})"
        )
    return arr


def _norm_bounds(emb: Embodiment) -> tuple[np.ndarray, np.ndarray]:
    lo = np.array([-100.0 if j.lerobot_norm == "m100_100" else 0.0 for j in emb.joints])
    hi = np.full(emb.num_joints, 100.0)
    return lo, hi


def format_bounds(fmt: Format, emb: Embodiment) -> tuple[np.ndarray, np.ndarray]:
    """Nominal per-joint (lower, upper) bounds of a format."""
    if fmt == "rad":
        return emb.lower, emb.upper
    if fmt == "deg":
        return np.degrees(emb.lower), np.degrees(emb.upper)
    if fmt == "lerobot":
        return _norm_bounds(emb)
    raise ValueError(f"unknown format '{fmt}', expected one of {FORMATS}")


def validate(values, fmt: Format, emb: Embodiment, tol: float = DEFAULT_LIMIT_TOL) -> np.ndarray:
    """Return ``values`` as an array, or raise :class:`UnitRangeError` naming the offending joints."""
    arr = _as_array(values, emb)
    lo, hi = format_bounds(fmt, emb)
    margin = tol * (hi - lo)
    flat = arr.reshape(-1, emb.num_joints)
    bad_low = (flat < lo - margin).any(axis=0)
    bad_high = (flat > hi + margin).any(axis=0)
    bad = bad_low | bad_high
    if bad.any():
        details = ", ".join(
            f"{name}: [{flat[:, i].min():.3g}, {flat[:, i].max():.3g}] not within [{lo[i]:.3g}, {hi[i]:.3g}]"
            for i, name in enumerate(emb.joint_names)
            if bad[i]
        )
        raise UnitRangeError(f"values are not valid '{fmt}' for embodiment '{emb.name}': {details}")
    return arr


def to_rad(values, fmt: Format, emb: Embodiment, tol: float = DEFAULT_LIMIT_TOL) -> np.ndarray:
    """Convert external joint values to canonical radians (validated)."""
    arr = validate(values, fmt, emb, tol)
    if fmt == "rad":
        return arr.copy()
    if fmt == "deg":
        return np.radians(arr)
    lo_n, hi_n = _norm_bounds(emb)
    frac = (arr - lo_n) / (hi_n - lo_n)
    return emb.lower + frac * (emb.upper - emb.lower)


def from_rad(q_rad, fmt: Format, emb: Embodiment, tol: float = DEFAULT_LIMIT_TOL) -> np.ndarray:
    """Convert canonical radians to an external format (input validated in radians)."""
    arr = validate(q_rad, "rad", emb, tol)
    if fmt == "rad":
        return arr.copy()
    if fmt == "deg":
        return np.degrees(arr)
    if fmt == "lerobot":
        lo_n, hi_n = _norm_bounds(emb)
        frac = (arr - emb.lower) / (emb.upper - emb.lower)
        return lo_n + frac * (hi_n - lo_n)
    raise ValueError(f"unknown format '{fmt}', expected one of {FORMATS}")


def convert(values, src: Format, dst: Format, emb: Embodiment, tol: float = DEFAULT_LIMIT_TOL) -> np.ndarray:
    """Convert between any two formats, always through validated radians."""
    return from_rad(to_rad(values, src, emb, tol), dst, emb, tol)


def check_dataset_stats(
    stats_min, stats_max, fmt: Format, emb: Embodiment, tol: float = DEFAULT_LIMIT_TOL
) -> None:
    """Check a dataset's per-joint min/max (e.g. LeRobot ``meta/stats.json``) against a declared format.

    Run this before training or inference on any dataset: a degrees dataset declared
    as ``"lerobot"`` fails here instead of producing a policy that dives into the table.
    """
    validate(np.stack([np.asarray(stats_min), np.asarray(stats_max)]), fmt, emb, tol)
