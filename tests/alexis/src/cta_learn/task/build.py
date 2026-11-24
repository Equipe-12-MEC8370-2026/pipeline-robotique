"""M3 — build a TaskSpec from the perception outputs of one demonstration (pure numpy, no I/O here).

Inputs: objects.json of P2.4d (rests and moves per part, table frame), the part sizes, the demonstrator's
hands (TPV + FPV) and the start/end gestures. One step per part, ordered by grasp time:
  initial pose = first rest, goal = last rest; the step's move = the one that DISPLACES the part most (rest before
  -> rest after), not the longest one: a part picked up to be checked and put back in place is handled, not
  moved (2nd CTA task: the block lifted and checked after the screwing). relation = "on_top_of" when the goal rests
  on the top of a part placed earlier (base at the height of its top, over its footprint), else "on_table";
  phases reach / grasp / transport / release (+ retreat after the last step).
Posture (see core.taskspec): from the vertical axis of the fitted box; "unknown" when the fits of the rest
disagree (``posture_ambiguous``). objects.json files written before the generic box fit (upright flag only) are
still read.
"""
from __future__ import annotations

import numpy as np

from cta_learn.core.taskspec import Goal, ObjectSpec, Phase, Pose, Relation, Step, TaskSpec
from cta_learn.task.segment import hand_side_fpv

TOL = 1e-3  # m, size comparisons


def rest_pose(rest: dict) -> dict:
    """Pose dict of a rest of objects.json, carrying the rest's ``posture_ambiguous`` flag."""
    return {**rest["pose"], "posture_ambiguous": bool(rest.get("posture_ambiguous", False))}


def posture_name(p: dict, size: list[float]) -> str:
    """Posture of a rest pose dict of objects.json, for a part of sizes ``size``."""
    if p.get("posture_ambiguous"):
        return "unknown"
    if "extents_m" not in p:  # older objects.json: profiles only
        return "upright" if p["upright"] else "lying"
    L, M, S = sorted(size, reverse=True)
    v = p["extents_m"][2]
    if abs(v - S) < TOL:
        return "lying"
    if abs(v - L) < TOL and L - M > TOL:
        return "upright"
    return "on_edge"


def _pose(p: dict, size: list[float]) -> Pose:
    return Pose(round(p["x"], 4), round(p["y"], 4), round(p["yaw_deg"], 1), posture_name(p, size), round(p["z_base"], 4))


def _vertical(p: dict, size: list[float]) -> float:
    """Height of a part at rest (its fitted box's vertical size; older files: profile convention)."""
    if "extents_m" in p:
        return float(p["extents_m"][2])
    return float(max(size)) if p["upright"] else float(min(size))


def _period(p: dict) -> float:
    return float(p.get("yaw_period_deg", 90.0 if p["upright"] else 180.0))


def relative_offset(goal: dict, ref: dict) -> tuple[list[float], float]:
    """Goal centre offset expressed in the ref part's frame (x along its length) and yaw relative to it."""
    c, s = np.cos(np.radians(ref["yaw_deg"])), np.sin(np.radians(ref["yaw_deg"]))
    dx, dy = goal["x"] - ref["x"], goal["y"] - ref["y"]
    return [round(float(c * dx + s * dy), 4), round(float(-s * dx + c * dy), 4)], round(float((goal["yaw_deg"] - ref["yaw_deg"]) % _period(goal)), 1)


def main_move(o: dict) -> dict:
    """The move that displaces the part most: move k goes from rest k to rest k + 1."""
    rests, moves = o["rests"], o["moves"]
    if len(rests) != len(moves) + 1:  # not the layout of object_trajectories: fall back to the longest move
        return max(moves, key=lambda m: m["t_release"] - m["t_grasp"])
    d = [np.hypot(rests[k + 1]["pose"]["x"] - rests[k]["pose"]["x"], rests[k + 1]["pose"]["y"] - rests[k]["pose"]["y"])
         for k in range(len(moves))]
    return moves[int(np.argmax(d))]


def _over(goal: dict, ref: dict, ref_size: list[float], margin: float = 0.01) -> bool:
    """Is the goal centre over the footprint of the ref part (fitted extents; older files: profile lying)?"""
    ext = ref.get("extents_m") or ([max(ref_size), min(ref_size)] if not ref["upright"] else [min(ref_size)] * 2)
    c, s = np.cos(np.radians(ref["yaw_deg"])), np.sin(np.radians(ref["yaw_deg"]))
    dx, dy = goal["x"] - ref["x"], goal["y"] - ref["y"]
    return abs(c * dx + s * dy) <= ext[0] / 2 + margin and abs(-s * dx + c * dy) <= ext[1] / 2 + margin


def build_taskspec(objects_json: dict, sizes: dict[str, list[float]], masses: dict[str, float], fpv_kp: np.ndarray,
                   fpv_t: np.ndarray, signals: dict, task_id: str, instruction: dict[str, str], source: dict,
                   descriptions: dict[str, str] | None = None) -> TaskSpec:
    objs, order = [], []
    for name, o in objects_json["objects"].items():
        if not o["rests"] or not o["moves"]:
            raise ValueError(f"{name}: needs at least one rest and one move")
        L = sorted(sizes[name], reverse=True)
        objs.append(ObjectSpec(name, "box", [float(v) for v in L], float(masses[name]), _pose(rest_pose(o["rests"][0]), L),
                               (descriptions or {}).get(name, "")))
        mv = main_move(o)
        order.append((mv["t_grasp"], name, mv, rest_pose(o["rests"][-1])))
    order.sort()
    steps, placed = [], {}
    prev_end = signals["start"][1] if signals.get("start") else order[0][0]
    for k, (tg, name, mv, goal) in enumerate(order):
        rel = Relation("on_table")
        if goal["z_base"] > 1e-6:
            for ref_name, ref in placed.items():
                top = ref["z_base"] + _vertical(ref, sizes[ref_name])
                if abs(goal["z_base"] - top) < TOL and _over(goal, ref, sizes[ref_name]):
                    off, yaw_rel = relative_offset(goal, ref)
                    rel = Relation("on_top_of", ref_name, off, yaw_rel)
        m0, m1 = mv["t_motion"]
        phases = [Phase("reach", [round(min(prev_end, tg), 2), round(tg, 2)])]
        if m0 > tg:
            phases.append(Phase("grasp", [round(tg, 2), round(m0, 2)]))
        phases.append(Phase("transport", [round(max(m0, tg), 2), round(m1, 2)]))
        if mv["t_release"] > m1:
            phases.append(Phase("release", [round(m1, 2), round(mv["t_release"], 2)]))
        is_last = k == len(order) - 1
        if is_last and signals.get("end") and signals["end"][0] > mv["t_release"]:
            phases.append(Phase("retreat", [round(mv["t_release"], 2), round(signals["end"][0], 2)]))
        side, conf = hand_side_fpv(fpv_kp, fpv_t, tg, mv["t_release"])
        gp = _pose(goal, sizes[name])
        steps.append(Step(k, name, side, round(conf, 2), round(tg, 2), round(mv["t_release"], 2), phases,
                          Goal(gp, rel, 0.02, 45.0 if gp.posture == "upright" else 15.0)))
        placed[name] = goal
        prev_end = mv["t_release"]
    success = []
    for s in steps:
        g = s.goal
        post = "" if g.pose.posture == "unknown" else f" {g.pose.posture}"
        if g.relation.type == "on_top_of":
            success.append(f"{s.object}{post} on top of {g.relation.ref} (centre within {g.tolerance_xy_m * 100:.0f} cm "
                           f"of offset {g.relation.offset_xy_m} m in its frame), released")
        else:
            success.append(f"{s.object}{post} on the table, released")
    return TaskSpec(task_id, instruction, objects_json.get("frame", "table frame"), objs, steps, success,
                    {k: (list(v) if v else None) for k, v in signals.items()}, source).validate()
