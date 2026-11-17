"""M3 — build a TaskSpec from the perception outputs of one demonstration (pure numpy, no I/O here).

Inputs: objects.json of P2.4d (rests and moves per part, table frame), the part sizes, the demonstrator's
hands (TPV + FPV) and the start/end gestures. One step per part, ordered by grasp time:
  initial pose = first rest, goal = last rest, relation = "on_top_of" when the goal rests on a part placed
  earlier (raised base over its footprint), else "on_table"; phases reach / grasp / transport / release
  (+ retreat after the last step).
"""
from __future__ import annotations

import numpy as np

from cta_learn.core.taskspec import Goal, ObjectSpec, Phase, Pose, Relation, Step, TaskSpec
from cta_learn.task.segment import hand_side_fpv


def _pose(p: dict) -> Pose:
    return Pose(round(p["x"], 4), round(p["y"], 4), round(p["yaw_deg"], 1), "upright" if p["upright"] else "lying",
                round(p["z_base"], 4))


def relative_offset(goal: dict, ref: dict) -> tuple[list[float], float]:
    """Goal centre offset expressed in the ref part's frame (x along its length) and yaw relative to it."""
    c, s = np.cos(np.radians(ref["yaw_deg"])), np.sin(np.radians(ref["yaw_deg"]))
    dx, dy = goal["x"] - ref["x"], goal["y"] - ref["y"]
    per = 90.0 if goal["upright"] else 180.0
    return [round(float(c * dx + s * dy), 4), round(float(-s * dx + c * dy), 4)], round(float((goal["yaw_deg"] - ref["yaw_deg"]) % per), 1)


def build_taskspec(objects_json: dict, sizes: dict[str, list[float]], masses: dict[str, float], fpv_kp: np.ndarray,
                   fpv_t: np.ndarray, signals: dict, task_id: str, instruction: dict[str, str], source: dict,
                   descriptions: dict[str, str] | None = None) -> TaskSpec:
    objs, order = [], []
    for name, o in objects_json["objects"].items():
        if not o["rests"] or not o["moves"]:
            raise ValueError(f"{name}: needs at least one rest and one move")
        L = sorted(sizes[name], reverse=True)
        objs.append(ObjectSpec(name, "box", [float(v) for v in L], float(masses[name]), _pose(o["rests"][0]["pose"]),
                               (descriptions or {}).get(name, "")))
        mv = max(o["moves"], key=lambda m: m["t_release"] - m["t_grasp"])  # the main manipulation of this part
        order.append((mv["t_grasp"], name, mv, o["rests"][-1]["pose"]))
    order.sort()
    steps, placed = [], {}
    prev_end = signals["start"][1] if signals.get("start") else order[0][0]
    for k, (tg, name, mv, goal) in enumerate(order):
        rel = Relation("on_table")
        if goal["z_base"] > 1e-6:
            for ref_name, ref in placed.items():
                if not ref["upright"] and abs(goal["z_base"] - min(sizes[ref_name])) < 1e-3:
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
        upright = goal["upright"]
        steps.append(Step(k, name, side, round(conf, 2), round(tg, 2), round(mv["t_release"], 2), phases,
                          Goal(_pose(goal), rel, 0.02, 45.0 if upright else 15.0)))
        placed[name] = goal
        prev_end = mv["t_release"]
    success = []
    for s in steps:
        g = s.goal
        if g.relation.type == "on_top_of":
            success.append(f"{s.object} {g.pose.posture} on top of {g.relation.ref} (centre within {g.tolerance_xy_m * 100:.0f} cm "
                           f"of offset {g.relation.offset_xy_m} m in its frame), released")
        else:
            success.append(f"{s.object} {g.pose.posture} on the table, released")
    return TaskSpec(task_id, instruction, objects_json.get("frame", "table frame"), objs, steps, success,
                    {k: (list(v) if v else None) for k, v in signals.items()}, source).validate()
