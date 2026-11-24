"""Gesture library (D13): generic gestures chained from a task's SO-101 layout (cta_learn.retarget.layout).

One gesture today, "pick_place_box": grasp a box-like part from above and put it down, on the table in a zone or on
top of a part placed before, at the demonstrated offset and relative yaw. It is the P3 phase machine
(datagen.stack_expert.StackExpert: approach, descend, close, lift, transport, lower, release, [clear, slide],
retreat), with its parameters derived from the part's geometry instead of the profile names:

  closing axis   across the narrower horizontal side (yaw symmetry 90 deg for a square footprint, else 180 deg);
  tall part      height > 1.5 x its narrower horizontal side (an upright profile): grip around the centre of
                 mass (TCP 35 mm up), slide the open gripper out sideways before rising (P3.3: rising straight up
                 tipped it); otherwise (lying profile, block): fingertips 2 mm above the table, rise straight up;
  heights        approach, lift and retreat from the part's height and its support's height (as in P3).

A step the library cannot do raises GestureMissing (the TaskSpec then needs a new gesture, written once, or the
MimicGen option of D13): a goal posture different from the start posture (turning a part over), a part wider than
the gripper opening.
"""
from __future__ import annotations

import math

from cta_learn.datagen.stack_expert import FINGERTIP_BELOW_TCP, StepPlan

GRIPPER_MAX_WIDTH = 0.035   # m, widest part the SO-101 closes on with margin (P0 cube 30 mm, P3 bars 25 mm)
TALL_RATIO = 1.5
GESTURES = ("pick_place_box",)


class GestureMissing(NotImplementedError):
    pass


def pick_place_box(layout: dict, obj: str) -> StepPlan:
    """StepPlan of the pick-and-place of ``obj`` (generic layout of retarget.layout)."""
    o, g = layout["objects"][obj], layout["goals"][obj]
    if g["posture"] != o["start_posture"]:
        raise GestureMissing(f"{obj}: goes from {o['start_posture']} to {g['posture']}: turning a part over is not in "
                             f"the gesture library ({', '.join(GESTURES)})")
    ex, ey, height = o["size_m"]
    width = min(ex, ey)
    if width > GRIPPER_MAX_WIDTH:
        raise GestureMissing(f"{obj}: narrower horizontal side {width * 1000:.0f} mm > gripper {GRIPPER_MAX_WIDTH * 1000:.0f} mm")
    square = abs(ex - ey) < 1e-6
    symmetry = math.pi / 2 if square else math.pi
    grip_yaw = 0.0 if (square or ex < ey) else math.pi / 2   # closing axis along the narrower local axis
    tall = height > TALL_RATIO * width
    grasp_z = 0.035 if tall else 0.002 + FINGERTIP_BELOW_TCP
    if g["relation"] == "on_table":
        support, txy, tyaw, support_h = None, tuple(g["xyyaw"][:2]), g["xyyaw"][2], 0.0
    else:
        support, txy, tyaw = g["ref"], tuple(g["offset_xy_m"]), math.radians(g["yaw_rel_deg"] or 0.0)
        support_h = layout["goals"][support]["vertical_m"]
    top_after = support_h + height
    return StepPlan(
        obj=obj, half_height=height / 2, height=height, symmetry=symmetry, grip_yaw=grip_yaw, grasp_z=grasp_z,
        approach_z=max(0.05 + 0.03, height + FINGERTIP_BELOW_TCP + 0.008),
        support=support, target_xy=txy, target_yaw=tyaw,
        lift_clear=0.02 if support else 0.03,
        retreat_z=0.09 if tall else max(0.08, top_after + FINGERTIP_BELOW_TCP + 0.02),
        slide=width / 2 + 0.045 if tall else 0.0,
    )


def plan_task(layout: dict) -> list[StepPlan]:
    """The task's steps, in order, each as a gesture of the library (only pick_place_box today)."""
    return [pick_place_box(layout, obj) for obj in layout["order"]]
