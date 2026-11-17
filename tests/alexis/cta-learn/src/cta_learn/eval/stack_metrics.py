"""P3 evaluation helpers (pure numpy): trials for the profile-stack task and outcome classification.

Same protocol as P0 (cta_learn.eval.metrics): a deterministic list of start conditions, n_train inside the
training ranges then n_ood "slightly outside" (the layout's range_ood boxes, at least one coordinate outside the
training box), each played by the policy AND by the scripted expert (paired reference). Offsets are relative to
each part's nominal start (layout start_xyyaw), as the scene's reset events use them.
"""
from __future__ import annotations

import numpy as np

PARTS = ("profile_short", "profile_long")
KEYS = ("x", "y", "yaw")
LIFT_THRESHOLD = 0.015   # m: a part whose bottom rose this much was picked up


def boxes(layout: dict, key: str) -> dict:
    """{part: {"x": (lo, hi), "y": ..., "yaw": ...}} half-ranges of the layout as symmetric boxes."""
    return {p: {k: (-layout["objects"][p][key][i], layout["objects"][p][key][i]) for i, k in enumerate(KEYS)} for p in PARTS}


def _sample(rng, box) -> dict:
    return {p: {k: float(rng.uniform(*box[p][k])) for k in KEYS} for p in PARTS}


def _inside(sample, box) -> bool:
    return all(box[p][k][0] <= sample[p][k] <= box[p][k][1] for p in PARTS for k in KEYS)


def make_stack_trials(layout: dict, n_train: int, n_ood: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    train, ood = boxes(layout, "range"), boxes(layout, "range_ood")
    trials = [{"trial": i, "split": "train_range", "offsets": _sample(rng, train)} for i in range(n_train)]
    for i in range(n_ood):
        while True:
            s = _sample(rng, ood)
            if not _inside(s, train):
                break
        trials.append({"trial": n_train + i, "split": "out_of_range", "offsets": s})
    return trials


def classify_stack_outcome(success: bool, dropped: dict, lifted: dict, final_checks: dict, holding_final: bool) -> str:
    """Name what happened in one episode, from privileged sim state (evaluation only), in task order.

    success · short_dropped · long_dropped · short_never_lifted · short_not_placed · long_never_lifted ·
    long_fell · long_not_on_short · held_at_timeout · stack_not_released
    ``final_checks``: the checks of retarget.profile_stack.stack_success on the last state (bools).
    """
    if success:
        return "success"
    for p, name in (("profile_short", "short"), ("profile_long", "long")):
        if dropped[p]:
            return f"{name}_dropped"
    if not lifted["profile_short"]:
        return "short_never_lifted"
    if not (final_checks["short_lying"] and final_checks["short_on_table"] and final_checks["short_in_zone"]):
        return "short_not_placed"
    if not lifted["profile_long"]:
        return "long_never_lifted"
    if not final_checks["long_upright"]:
        return "long_fell"
    if not final_checks["long_on_short"]:
        return "held_at_timeout" if holding_final else "long_not_on_short"
    return "stack_not_released"   # stack right, but the gripper stayed within 5 cm or the parts kept moving
