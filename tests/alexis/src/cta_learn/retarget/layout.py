"""M4 (generic) — project ANY box-part TaskSpec onto the SO-101 workspace (pure python, no Isaac).

Replaces the profile-only ``retarget.profile_stack.so101_layout`` for new tasks (that one stays: the published P3
results were made with it). What is kept is the TASK (ADR-001): the order of the steps, the final relations
("on the table in a place zone", "on top of a part placed before" with the demonstrated offset in the ref part's
frame and relative yaw) and the instruction. What changes, decided in a small per-task ADAPTATION config (YAML,
reviewed by a human: no code per task); everything it does not give is computed:

  size     — default: the part scaled UNIFORMLY so its smallest size = ``grasp_size_m`` (25 mm: what the SO-101
             gripper surely grasps, D10); or ``size_m`` given (longest first, e.g. the D10 profile lengths).
  mass     — default ``default_mass_kg`` (light, D9); or ``mass_kg``.
  posture  — start / goal posture of the demo; "unknown" (the video could not tell, see core.taskspec) becomes
             "lying", the most stable one; ``start_posture`` overrides the demo (e.g. D11: start lying). Without
             an override, a part whose start and goal postures differ starts in its goal posture (no turning-over
             gesture in the library; the start place in the robot cell is ours to set).
  places   — default: starts taken in order from ``start_pool``, the first on-table goal zone at ``goal_zone``
             (robot env frame: base at the origin, workspace +x, table top z = 0; m, rad); or given per part.

Every change from the demo is written in ``deviations`` (shown at the human review).
Part frame in the scene: the box is spawned with ``size_m`` = its extents along its local x, y, z AT THE START
POSTURE (lying: (a, b, c); on_edge: (a, c, b); upright: (b, c, a) for sizes a >= b >= c).
"""
from __future__ import annotations

import math

from cta_learn.core.taskspec import TaskSpec

TOL = 1e-6
DEFAULTS = {
    "grasp_size_m": 0.025,
    "default_mass_kg": 0.02,
    # nominal starts and their uniform half-ranges (x, y, yaw); kept >= 5 cm from the gripper's rest TCP
    # (0.195, 0.021, 0.071) (P3.1); same values as the profile task (DEFAULT_PLACES of retarget.profile_stack)
    "start_pool": [
        {"start": [0.28, -0.03, 0.0], "range": [0.025, 0.03, math.radians(30)], "range_ood": [0.04, 0.05, math.radians(50)]},
        {"start": [0.19, -0.11, 0.0], "range": [0.02, 0.025, math.radians(45)], "range_ood": [0.03, 0.035, math.radians(45)]},
    ],
    "goal_zone": [0.21, 0.10, 0.0],
    "tol_zone_xy_m": 0.02,
    "tol_zone_yaw_deg": 20.0,
    "tol_on_top_xy_m": 0.012,
    "tol_posture_deg": 15.0,
}


def extents_at(size_sorted: list[float], posture: str) -> list[float]:
    """Extents along the part's local (x, y, z) at a posture, for sizes a >= b >= c."""
    a, b, c = size_sorted
    return {"lying": [a, b, c], "on_edge": [a, c, b], "upright": [b, c, a]}[posture]


def vertical_size(size_sorted: list[float], posture: str) -> float:
    return extents_at(size_sorted, posture)[2]


def robot_layout(spec: TaskSpec, adapt: dict | None = None) -> dict:
    adapt = adapt or {}
    cfg = {**DEFAULTS, **{k: v for k, v in adapt.items() if k != "objects"}}
    per = adapt.get("objects", {})
    steps = sorted(spec.steps, key=lambda s: s.index)
    step_of = {s.object: s for s in steps}
    deviations: list[str] = []
    objects, scale = {}, {}
    pool = list(cfg["start_pool"])
    for k, o in enumerate(sorted(spec.objects, key=lambda o: [s.object for s in steps].index(o.id) if o.id in step_of else 99)):
        a = per.get(o.id, {})
        demo = sorted(o.size_m, reverse=True)
        if "size_m" in a:
            size = sorted(a["size_m"], reverse=True)
            deviations.append(f"{o.id}: size {fmt_mm(size)} (demo {fmt_mm(demo)}), set in the adaptation config")
        else:
            f = cfg["grasp_size_m"] / demo[2]
            size = [round(v * f, 4) for v in demo]
            deviations.append(f"{o.id}: scaled x{f:.3f} to {fmt_mm(size)} (demo {fmt_mm(demo)}) so the SO-101 grasps it")
        scale[o.id] = size[2] / demo[2]
        posture = a.get("start_posture", o.initial.posture)
        goal_posture = a.get("goal_posture", step_of[o.id].goal.pose.posture) if o.id in step_of else "unknown"
        if posture != o.initial.posture:
            deviations.append(f"{o.id}: starts {posture} (demo: {o.initial.posture})"
                              + (f", {a['start_posture_reason']}" if a.get("start_posture_reason") else ""))
        elif goal_posture != "unknown" and posture != goal_posture:
            # the robot cell's start place is ours to set: the part starts as it must arrive, since turning a part
            # over is not in the gesture library (as D11 decided by hand for the profiles)
            deviations.append(f"{o.id}: starts {goal_posture} as at arrival (demo: {posture}), no turning over")
            posture = goal_posture
        if posture == "unknown":
            posture = "lying"
            deviations.append(f"{o.id}: start posture unknown in the video -> lying (most stable)")
        if "start_xyyaw" in a:
            place = {"start": a["start_xyyaw"], "range": a["range"], "range_ood": a["range_ood"]}
        else:
            if not pool:
                raise ValueError(f"{o.id}: no start place left in start_pool; give start_xyyaw in the adaptation config")
            place = pool.pop(0)
        mass = float(a.get("mass_kg", cfg["default_mass_kg"]))
        objects[o.id] = {"size_m": extents_at(size, posture), "box_m": size, "mass_kg": mass, "start_posture": posture,
                         "start_xyyaw": list(place["start"]), "range": list(place["range"]),
                         "range_ood": list(place["range_ood"]), "demo_size_m": demo}
    goals, zone_used = {}, False
    for s in steps:
        g, a = s.goal, per.get(s.object, {})
        posture = a.get("goal_posture", g.pose.posture)
        if posture == "unknown":
            posture = objects[s.object]["start_posture"]
            deviations.append(f"{s.object}: goal posture unknown in the video -> {posture} (as it starts)")
        box = objects[s.object]["box_m"]
        goal = {"posture": posture, "relation": g.relation.type, "vertical_m": vertical_size(box, posture),
                "tol_posture_deg": float(a.get("tol_posture_deg", cfg["tol_posture_deg"]))}
        if g.relation.type == "on_table":
            if "goal_xyyaw" in a:
                xyyaw = a["goal_xyyaw"]
            elif not zone_used:
                xyyaw, zone_used = cfg["goal_zone"], True
            else:
                raise ValueError(f"{s.object}: a 2nd on-table goal needs goal_xyyaw in the adaptation config")
            goal.update({"xyyaw": list(xyyaw), "tol_xy_m": float(a.get("tol_xy_m", cfg["tol_zone_xy_m"])),
                         "tol_yaw_deg": float(a.get("tol_yaw_deg", cfg["tol_zone_yaw_deg"]))})
        else:  # on_top_of: offset in the ref part's frame, scaled like the ref part
            ref = g.relation.ref
            goal.update({"ref": ref, "offset_xy_m": [round(v * scale[ref], 4) for v in g.relation.offset_xy_m],
                         "yaw_rel_deg": g.relation.yaw_rel_deg,
                         "tol_xy_m": float(a.get("tol_xy_m", cfg["tol_on_top_xy_m"]))})
        goals[s.object] = goal
    return {"task_id": spec.task_id + "_so101", "from_taskspec": spec.task_id, "instruction": spec.instruction,
            "order": [s.object for s in steps], "objects": objects, "goals": goals,
            "grasp_width_m": min(min(o["box_m"]) for o in objects.values()), "deviations": deviations,
            "frame": "robot env frame of CTA-SO101 scenes: base at origin, workspace +x, table top z = 0 (m, rad)"}


def fmt_mm(size) -> str:
    return " x ".join(f"{v * 1000:.0f}" for v in size) + " mm"


# ---------------------------------------------------------------- success (array-agnostic: numpy or torch)

def rotate_xyzw(q, v, xp):
    """Rotate vectors v (N, 3) by quaternions q (N, 4) in XYZW order (Isaac Lab 3.0 convention)."""
    qv, w = q[:, :3], q[:, 3:4]
    t = 2 * xp.linalg.cross(qv, v)
    return v + w * t + xp.linalg.cross(qv, t)


def _axis_z(quat, k: int, xp):
    """|z component| of the part's local axis k in the world, per env."""
    e = xp.zeros_like(quat[:, :3])
    e[:, k] = 1.0
    return xp.abs(rotate_xyzw(quat, e, xp)[:, 2])


def posture_ok(quat, size_local: list[float], vertical: float, tol_deg: float, xp):
    """The part rests with a face of height ``vertical`` up. One local axis has that size: it must be vertical
    (within tol). Two have it (square section, e.g. a lying profile): the third must be horizontal (within tol).
    Three (cube): always true."""
    match = [k for k in range(3) if abs(size_local[k] - vertical) < 1e-6]
    if len(match) == 1:
        return _axis_z(quat, match[0], xp) > math.cos(math.radians(tol_deg))
    if len(match) == 2:
        other = ({0, 1, 2} - set(match)).pop()
        return _axis_z(quat, other, xp) < math.sin(math.radians(tol_deg))
    if len(match) == 3:
        return _axis_z(quat, 2, xp) > -1.0
    raise ValueError(f"no local axis of size {vertical} in {size_local}")


def task_success(pos: dict, quat: dict, tcp_pos, zone_xy: dict, layout: dict, xp, *, release_distance: float = 0.05,
                 z_tol: float = 0.01):
    """Per-env success of a layout's goals + the individual checks (dict of bool arrays).

    pos / quat: {part: (N, 3) / (N, 4) XYZW} in the env frame (table top z = 0); zone_xy: {part: (N, 2)} centre of
    the place zone of each on-table goal. Checks, for each goal in order: posture, resting height (on the table /
    on the top of its ref part), place (in its zone / at the demonstrated offset over its ref part); then the
    gripper away from every part."""
    checks = {}
    for name in layout["order"]:
        g, o = layout["goals"][name], layout["objects"][name]
        p = pos[name]
        checks[f"{name}_posture"] = posture_ok(quat[name], o["size_m"], g["vertical_m"], g["tol_posture_deg"], xp)
        if g["relation"] == "on_table":
            checks[f"{name}_on_table"] = p[:, 2] < g["vertical_m"] / 2 + z_tol
            d = ((p[:, 0] - zone_xy[name][:, 0]) ** 2 + (p[:, 1] - zone_xy[name][:, 1]) ** 2) ** 0.5
            checks[f"{name}_in_zone"] = d < g["tol_xy_m"]
        else:
            ref, gr = g["ref"], layout["goals"][g["ref"]]
            pr = pos[ref]
            top = pr[:, 2] + gr["vertical_m"] / 2
            checks[f"{name}_on_{ref}"] = xp.abs(p[:, 2] - (top + g["vertical_m"] / 2)) < z_tol
            # demonstrated offset, in the ref part's frame (its local x, y axes projected on the table)
            ex = xp.zeros_like(p); ex[:, 0] = 1.0
            ax = rotate_xyzw(quat[ref], ex, xp)
            n = (ax[:, 0] ** 2 + ax[:, 1] ** 2) ** 0.5 + 1e-9
            cx, sx = ax[:, 0] / n, ax[:, 1] / n
            ox, oy = g["offset_xy_m"]
            tx, ty = pr[:, 0] + cx * ox - sx * oy, pr[:, 1] + sx * ox + cx * oy
            checks[f"{name}_at_offset"] = ((p[:, 0] - tx) ** 2 + (p[:, 1] - ty) ** 2) ** 0.5 < g["tol_xy_m"]
    away = None
    for name in layout["order"]:
        a = ((tcp_pos - pos[name]) ** 2).sum(1) ** 0.5 > release_distance
        away = a if away is None else away & a
    checks["released"] = away
    ok = None
    for v in checks.values():
        ok = v if ok is None else ok & v
    return ok, checks
