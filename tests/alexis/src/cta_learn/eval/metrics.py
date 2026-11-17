"""Evaluation helpers (pure numpy): trial sampling, action clipping, outcome classification, statistics."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

# Offsets from the nominal poses, same convention as the scene's reset events (m, rad).
TRAIN_CUBE = {"x": (-0.04, 0.04), "y": (-0.08, 0.08), "yaw": (-math.radians(45), math.radians(45))}
TRAIN_TARGET = {"x": (-0.04, 0.04), "y": (-0.03, 0.03)}
# "Slightly outside" the training range: at least one cube coordinate beyond it, target unchanged.
OOD_CUBE = {"x": (-0.06, 0.06), "y": (-0.10, 0.10), "yaw": (-math.radians(60), math.radians(60))}

LIFT_THRESHOLD = 0.02  # m above the cube's initial height = "was lifted"
TARGET_TOL = 0.03       # m, same xy tolerance as the success term


def _uniform(rng, rng_dict):
    return {k: float(rng.uniform(*v)) for k, v in rng_dict.items()}


def _inside(sample, box) -> bool:
    return all(box[k][0] <= sample[k] <= box[k][1] for k in box)


def make_trials(n_train: int, n_ood: int, seed: int) -> list[dict]:
    """Deterministic list of initial conditions: ``n_train`` inside the training range, then ``n_ood`` outside."""
    rng = np.random.default_rng(seed)
    trials = []
    for i in range(n_train):
        trials.append({"trial": i, "split": "train_range", "cube": _uniform(rng, TRAIN_CUBE),
                       "target": _uniform(rng, TRAIN_TARGET)})
    for i in range(n_ood):
        while True:
            cube = _uniform(rng, OOD_CUBE)
            if not _inside(cube, TRAIN_CUBE):
                break
        trials.append({"trial": n_train + i, "split": "out_of_range", "cube": cube,
                       "target": _uniform(rng, TRAIN_TARGET)})
    return trials


def clip_to_limits(action: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, bool]:
    """Clip a joint-position command (rad) to the hard joint limits; also report whether it had to."""
    a = np.asarray(action, dtype=np.float64)
    clipped = np.clip(a, lower, upper)
    return clipped.astype(np.float32), bool(np.any(clipped != a))


def classify_outcome(success: bool, off_table: bool, cube_z: np.ndarray, cube_xy_final: np.ndarray,
                     target_xy: np.ndarray, holding_final: bool) -> str:
    """Name what happened in one episode, from privileged sim state (evaluation only).

    success · cube_off_table · never_lifted · in_target_no_success · held_at_timeout · placed_outside
    (in_target_no_success: cube within the target xy tolerance but the success term never fired, i.e.
    not resting (z or speed) or the gripper still within 5 cm of the cube)
    """
    if success:
        return "success"
    if off_table:
        return "cube_off_table"
    cube_z = np.asarray(cube_z)
    if cube_z.max() - cube_z[0] < LIFT_THRESHOLD:
        return "never_lifted"
    in_target = float(np.linalg.norm(np.asarray(cube_xy_final) - np.asarray(target_xy))) < TARGET_TOL
    if in_target:
        return "in_target_no_success"
    if holding_final:
        return "held_at_timeout"
    return "placed_outside"


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson score interval for a success rate k/n (good at small n and near 0 / 1)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarize(records: list[dict]) -> dict:
    """Success rate + Wilson CI + outcome counts, overall and per split."""
    def block(rs):
        k = sum(r["outcome"] == "success" for r in rs)
        lo, hi = wilson_interval(k, len(rs))
        return {"n": len(rs), "successes": k, "success_rate": k / len(rs) if rs else 0.0,
                "ci95": [round(lo, 3), round(hi, 3)], "outcomes": dict(Counter(r["outcome"] for r in rs))}

    out = {"all": block(records)}
    for split in sorted({r["split"] for r in records}):
        out[split] = block([r for r in records if r["split"] == split])
    return out


def paired_table(policy: list[dict], expert: list[dict]) -> dict:
    """Cross-tabulate policy vs expert success on the same trials."""
    e = {r["trial"]: r["outcome"] == "success" for r in expert}
    t = Counter()
    for r in policy:
        if r["trial"] in e:
            t[f"policy_{'ok' if r['outcome'] == 'success' else 'fail'}__expert_{'ok' if e[r['trial']] else 'fail'}"] += 1
    return dict(t)
