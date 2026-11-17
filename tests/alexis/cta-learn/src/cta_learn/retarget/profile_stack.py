"""M4 (first version) — project the profile TaskSpec onto the SO-101 workspace (pure python, no Isaac).

The human demo happens on a 0.8 m table with 40 mm profiles; the SO-101 reaches ~0.3 m and grasps ~30 mm
parts. What is kept is the TASK (ADR-001: reproduce the task, not the human's motions): the final relations
(short part lying on the table in a place zone, long part upright on top of it with the demonstrated offset in
the short part's frame and relative yaw), the order of the steps and the instruction. What changes:
  D10 — parts scaled down so the SO-101 surely grasps them (section 25 mm, lengths 50 / 75 mm, 20 / 25 g);
  D11 — the short part STARTS LYING (no tipping over: hard for a 5-joint arm grasping from above);
  places — start / goal zones laid out in front of the robot (robot frame of the P0 scene: base at the origin,
  workspace along +x, table top at z = 0); start poses randomised, the short part's goal zone fixed
  ("at the front" of the demo), the long part's goal follows the short part wherever it ends.
"""
from __future__ import annotations

import math

from cta_learn.core.taskspec import TaskSpec

HUMAN_SECTION = 0.04  # m, section of the demo's profiles (P2.3)

DEFAULT_PLACES = {
    # robot env frame (m, rad); "range" = uniform half-widths for training, "range_ood" for the out-of-range
    # evaluation trials (P0 convention: 40 trials in range + 10 slightly outside)
    # Starts kept >= 5 cm from the gripper's rest position (TCP at (0.195, 0.021, 0.071) at the start pose,
    # measured by check_profile_scene): a 75 mm upright part spawned under it was knocked over at reset.
    "profile_short": {"start": (0.28, -0.03, 0.0), "range": (0.025, 0.03, math.radians(30)),
                      "range_ood": (0.04, 0.05, math.radians(50))},
    "profile_long": {"start": (0.19, -0.11, 0.0), "range": (0.02, 0.025, math.radians(45)),
                     "range_ood": (0.03, 0.035, math.radians(45))},
    "goal_short": (0.21, 0.10, 0.0),   # centre of the place zone, short part's long axis along robot y at yaw 0
}


def so101_layout(spec: TaskSpec, section: float = 0.025, lengths: dict[str, float] | None = None,
                 masses: dict[str, float] | None = None, places: dict | None = None) -> dict:
    lengths = lengths or {"profile_short": 0.05, "profile_long": 0.075}
    masses = masses or {"profile_short": 0.02, "profile_long": 0.025}
    places = places or DEFAULT_PLACES
    scale = section / HUMAN_SECTION
    steps = sorted(spec.steps, key=lambda s: s.index)
    if [s.object for s in steps] != ["profile_short", "profile_long"]:
        raise ValueError("this layout expects step 0 = profile_short, step 1 = profile_long")
    s0, s1 = steps
    if s0.goal.relation.type != "on_table" or s0.goal.pose.posture != "lying":
        raise ValueError("step 0 must lay profile_short on the table")
    r1 = s1.goal.relation
    if r1.type != "on_top_of" or r1.ref != "profile_short" or s1.goal.pose.posture != "upright":
        raise ValueError("step 1 must stand profile_long upright on top of profile_short")
    objects, deviations = {}, []
    for o in spec.objects:
        L = lengths[o.id]
        posture = o.initial.posture
        if o.id == "profile_short" and posture != "lying":
            deviations.append(f"D11: {o.id} starts lying (demo: {posture}), no tipping over")
            posture = "lying"
        # box extents in the part's own frame: lying = length along local x; upright = length along local z
        size = [L, section, section] if posture == "lying" else [section, section, L]
        p = places[o.id]
        objects[o.id] = {"size_m": size, "length_m": L, "section_m": section, "mass_kg": masses[o.id],
                         "start_posture": posture, "start_xyyaw": list(p["start"]), "range": list(p["range"]),
                         "range_ood": list(p["range_ood"]), "demo_size_m": o.size_m}
    deviations.append(f"D10: parts scaled to section {section * 1000:.0f} mm, lengths "
                      f"{', '.join(f'{k} {v * 1000:.0f} mm' for k, v in lengths.items())} (demo section 40 mm)")
    goals = {
        "profile_short": {"posture": "lying", "relation": "on_table", "xyyaw": list(places["goal_short"]),
                          "tol_xy_m": 0.02, "tol_yaw_deg": 20.0},
        "profile_long": {"posture": "upright", "relation": "on_top_of", "ref": "profile_short",
                         "offset_xy_m": [round(v * scale, 4) for v in r1.offset_xy_m], "yaw_rel_deg": r1.yaw_rel_deg,
                         "tol_xy_m": 0.012, "tol_upright_deg": 15.0},
    }
    return {"task_id": spec.task_id + "_so101", "from_taskspec": spec.task_id, "instruction": spec.instruction,
            "order": [s.object for s in steps], "objects": objects, "goals": goals, "deviations": deviations,
            "frame": "robot env frame of CTA-SO101 scenes: base at origin, workspace +x, table top z = 0 (m, rad)"}


# ---------------------------------------------------------------- success (array-agnostic: numpy or torch)

def rotate_xyzw(q, v, xp):
    """Rotate vectors v (N, 3) by quaternions q (N, 4) in XYZW order (Isaac Lab 3.0 convention)."""
    qv, w = q[:, :3], q[:, 3:4]
    t = 2 * xp.linalg.cross(qv, v)  # linalg.cross: last axis in both numpy (>= 2.0) and torch
    return v + w * t + xp.linalg.cross(qv, t)


def stack_success(short_pos, short_quat, long_pos, long_quat, tcp_pos, goal_xy, layout: dict, xp, *,
                  release_distance: float = 0.05):
    """Per-env success of the stack task + the individual checks (dict of bool arrays).

    Positions in the env frame (table top z = 0). xp = numpy or torch (same calls)."""
    so, lo = layout["objects"]["profile_short"], layout["objects"]["profile_long"]
    gs, gl = layout["goals"]["profile_short"], layout["goals"]["profile_long"]
    n = short_pos.shape[0]
    ex = xp.zeros_like(short_pos); ex[:, 0] = 1.0   # short part's long axis = local x when it starts lying
    ez = xp.zeros_like(long_pos); ez[:, 2] = 1.0    # long part's long axis = local z when it starts upright
    a_s = rotate_xyzw(short_quat, ex, xp)
    a_l = rotate_xyzw(long_quat, ez, xp)
    sec = so["section_m"]
    checks = {
        "short_lying": xp.abs(a_s[:, 2]) < math.sin(math.radians(20.0)),
        "short_on_table": short_pos[:, 2] < sec / 2 + 0.01,
        "short_in_zone": ((short_pos[:, 0] - goal_xy[:, 0]) ** 2 + (short_pos[:, 1] - goal_xy[:, 1]) ** 2) ** 0.5 < gs["tol_xy_m"],
        "long_upright": xp.abs(a_l[:, 2]) > math.cos(math.radians(gl["tol_upright_deg"])),
        "long_on_short": (xp.abs(long_pos[:, 2] - (sec + lo["length_m"] / 2)) < 0.01)
                         & (((long_pos[:, 0] - short_pos[:, 0]) ** 2 + (long_pos[:, 1] - short_pos[:, 1]) ** 2) ** 0.5 < gl["tol_xy_m"]),
        "released": (((tcp_pos - long_pos) ** 2).sum(1) ** 0.5 > release_distance)
                    & (((tcp_pos - short_pos) ** 2).sum(1) ** 0.5 > release_distance),
    }
    ok = checks["short_lying"]
    for k in checks:
        ok = ok & checks[k]
    assert ok.shape[0] == n
    return ok, checks
