"""Consistency checks between the simulator and the embodiment description."""

from __future__ import annotations

import numpy as np

from cta_learn.core import Embodiment


def verify_robot_matches_embodiment(robot, emb: Embodiment, atol: float = 1e-3) -> list[str]:
    """Return a list of problems (empty = OK): missing joints or limits that differ from the YAML."""
    problems: list[str] = []
    sim_names = list(robot.data.joint_names)
    limits = robot.data.joint_pos_limits.torch[0].cpu().numpy()  # (num_joints, 2), radians
    for j in emb.joints:
        if j.name not in sim_names:
            problems.append(f"joint '{j.name}' missing in sim (sim joints: {sim_names})")
            continue
        lo, hi = limits[sim_names.index(j.name)]
        if not (np.isclose(lo, j.lower, atol=atol) and np.isclose(hi, j.upper, atol=atol)):
            problems.append(
                f"joint '{j.name}': sim limits [{np.degrees(lo):.2f}, {np.degrees(hi):.2f}] deg "
                f"!= yaml [{np.degrees(j.lower):.2f}, {np.degrees(j.upper):.2f}] deg"
            )
    extra = set(sim_names) - {j.name for j in emb.joints}
    if extra:
        problems.append(f"sim has joints not in the embodiment: {sorted(extra)}")
    return problems
