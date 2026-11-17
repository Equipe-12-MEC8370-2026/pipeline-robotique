"""Scripted expert for CTA-SO101-ProfileStack-v0 (P3.2): the TaskSpec's steps as chained pick-and-places.

The plan comes from the SO-101 layout generated from the TaskSpec (configs/tasks/so101_profile_stack.yaml):
the ORDER of the steps, each part's posture and size, where it goes (the short part: a fixed zone on the table;
the long part: on top of the short part WHERE IT ACTUALLY LIES, with the demo's offset and relative yaw).
Each step runs the P0 phase machine (approach, descend, close, lift, transport, lower, release, retreat) with
heights derived from the part geometry. Simulator-free (numpy), unit-tested; the IK is the P0 one.

Gripper conventions (P0): ``yaw`` = world yaw of the gripper closing axis; the TCP sits between the fingers so
that a centred part of section 25 mm is 5 mm from the fixed finger, 10 mm above the fingertips.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from cta_learn.datagen.pick_place_expert import READY_POSE_DEG, ExpertCommand

PHASES = ("ready", "approach", "descend", "close", "lift", "transport", "lower", "release", "clear", "slide", "retreat", "done")
VERTICAL_PHASES = ("descend", "close", "lower", "release", "clear", "slide")
TILT_WEIGHT_HIGH, TILT_WEIGHT_LOW = 0.3, 0.03
FINGERTIP_BELOW_TCP = 0.010  # m


def wrap(angle: float, period: float) -> float:
    """Map an angle to [-period/2, period/2)."""
    return (angle + period / 2) % period - period / 2


def bearing(pos) -> float:
    """Direction of a point seen from the robot base (env frame: base at the origin)."""
    return math.atan2(float(pos[1]), float(pos[0]))


def rot2(yaw: float, v) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


@dataclass
class StepPlan:
    """One pick-and-place, all heights in metres."""

    obj: str
    half_height: float        # part centre above its bottom face
    height: float             # part height as it stands (bottom to top)
    symmetry: float           # yaw symmetry of the part seen from above: pi (lying bar), pi/2 (square section)
    grip_yaw: float           # gripper closing axis relative to the part's local x (pi/2: across a lying bar)
    grasp_z: float            # TCP above the part's bottom when closing
    approach_z: float         # TCP above the part's bottom before descending (fingertips clear of its top)
    support: str | None       # None = table; else the part it is placed on
    target_xy: tuple[float, float]  # absolute (support None) or offset in the support's frame
    target_yaw: float         # absolute part yaw (support None) or relative to the support's yaw
    lift_clear: float         # part bottom above the support's top while transporting
    retreat_z: float          # absolute TCP height after release (and after the slide, if any)
    slide: float = 0.0        # upright part: after opening, slide the open gripper this far horizontally, in the
                              # plane of the fingers (perpendicular to the closing axis, toward the base), before
                              # rising. Rising straight up tilted the wrist and a finger tipped the part (P3.3).


@dataclass
class StackParams:
    """Per-episode variations (sampled) shared by the steps."""

    speed: float = 0.12               # max TCP speed (m/s)
    joint_speed: float = 0.5          # ready phase (rad/s)
    grasp_offset_xy: tuple[float, float] = (0.0, 0.0)
    grasp_dz: float = 0.0             # added to every grasp height
    approach_dz: float = 0.0
    lift_dz: float = 0.0
    retreat_dz: float = 0.0
    place_drop: float = 0.003         # part bottom this much above its support when released
    lift_min: float = 0.07            # TCP never transports lower than this
    slide_speed: float = 0.05         # m/s, open gripper sliding away from an upright part
    clear_dist: float = 0.010         # before the slide: move the fixed finger this far away from the released part
    clear_speed: float = 0.03         # (it leaned ~4 deg on that finger and the slide dragged it off the stack, P3.3)
    gripper_open: float = math.radians(35.0)
    gripper_closed: float = math.radians(-10.0)
    gripper_release: float = math.radians(60.0)
    close_time: float = 1.2
    release_time: float = 0.8
    reach_tol: float = 0.008
    descend_tol: float = 0.004
    ready_tol_deg: float = 3.0
    phase_timeout: float = 6.0
    lift_check: float = 0.015         # part bottom above where it was grasped, else regrasp
    max_regrasps: int = 2

    @classmethod
    def sample(cls, rng: np.random.Generator) -> "StackParams":
        return cls(
            speed=rng.uniform(0.10, 0.15),
            grasp_offset_xy=tuple(rng.uniform(-0.002, 0.002, size=2)),
            grasp_dz=rng.uniform(-0.002, 0.002),
            approach_dz=rng.uniform(0.0, 0.004),
            lift_dz=rng.uniform(0.0, 0.01),
            retreat_dz=rng.uniform(0.0, 0.01),
            place_drop=rng.uniform(0.002, 0.005),
        )


def plan_from_layout(layout: dict) -> list[StepPlan]:
    """TaskSpec-derived layout -> one StepPlan per step, in the TaskSpec's order."""
    plans = []
    for obj in layout["order"]:
        o, g = layout["objects"][obj], layout["goals"][obj]
        lying = o["start_posture"] == "lying"
        sec, length = o["section_m"], o["length_m"]
        height = sec if lying else length
        if lying:   # grasp across the bar, fingertips 2 mm above the table (P0 grasp height)
            grasp_z, symmetry, grip_yaw = 0.012, math.pi, math.pi / 2
        else:       # upright: fingertips ~25 mm up, the grip around the centre of mass (less swinging)
            grasp_z, symmetry, grip_yaw = 0.035, math.pi / 2, 0.0
        if g["relation"] == "on_table":
            support, txy, tyaw = None, tuple(g["xyyaw"][:2]), g["xyyaw"][2]
            support_h = 0.0
        else:
            support, txy, tyaw = g["ref"], tuple(g["offset_xy_m"]), math.radians(g["yaw_rel_deg"])
            support_h = layout["objects"][support]["section_m"]  # the support lies: its height = its section
        top_after = support_h + height
        plans.append(StepPlan(
            obj=obj, half_height=height / 2, height=height, symmetry=symmetry, grip_yaw=grip_yaw, grasp_z=grasp_z,
            # fingertips 8 mm above the top of an upright part: higher, near the base, the wrist reaches its limit
            # and the TCP stalled ~1 cm off (3/31 approach timeouts at TCP z 0.108-0.112, P3.3)
            approach_z=max(0.05 + 0.03, height + FINGERTIP_BELOW_TCP + 0.008),
            support=support, target_xy=txy, target_yaw=tyaw,
            lift_clear=0.02 if support else 0.03,
            # lying part: rise straight up (fingertips clear of its top); upright part: slide out sideways first
            retreat_z=max(0.08, top_after + FINGERTIP_BELOW_TCP + 0.02) if lying else 0.09,
            slide=0.0 if lying else sec / 2 + 0.045,
        ))
    return plans


@dataclass
class StackExpert:
    plans: list[StepPlan]
    params: StackParams = field(default_factory=StackParams)
    step_index: int = 0
    phase: str = "ready"
    t_phase: float = 0.0
    failed: bool = False
    failure_reason: str = ""
    regrasps: int = 0
    _cmd: np.ndarray | None = None
    _q_cmd: np.ndarray | None = None
    _yaw: float = 0.0
    _grasp: dict | None = None    # fixed at the start of the descent: xy, bottom z, part yaw, gripper yaw
    _place: dict | None = None    # fixed at the start of the transport: xy, TCP z, gripper yaw
    _hold_off: np.ndarray | None = None  # measured TCP - part bottom (xy) while carrying (filtered)
    _rel: dict | None = None      # release point, fixed-finger clearance point, slide end point (xy)
    last_goal: np.ndarray | None = None  # goal of the current phase before rate limiting (diagnostics)

    def reset(self, tcp_pos, parts: dict, params: StackParams | None = None, arm_q=None) -> None:
        if params is not None:
            self.params = params
        self.step_index, self.phase, self.t_phase = 0, "ready", 0.0
        self.failed, self.failure_reason, self.regrasps = False, "", 0
        self._cmd = np.asarray(tcp_pos, dtype=np.float64).copy()
        self._q_cmd = None if arm_q is None else np.asarray(arm_q, dtype=np.float64).copy()
        self._grasp = self._place = self._hold_off = self._rel = None
        self._yaw = self._grasp_yaw(parts)

    @property
    def done(self) -> bool:
        return self.phase == "done"

    @property
    def plan(self) -> StepPlan:
        return self.plans[min(self.step_index, len(self.plans) - 1)]

    # ---------------------------------------------------------------- geometry
    def _grasp_yaw(self, parts: dict) -> float:
        """Closing-axis yaw for the grasp. The part's symmetry gives several equivalent yaws: take the one closest
        to the arm's bearing (direction of the part seen from the base), i.e. the smallest wrist roll. Taking the
        one closest to 0 needed ~60 deg of roll for an upright part at -30 deg bearing turned +35 deg, and the IK
        stalled ~1 cm off target (the TCP is off the roll axis): 3/31 approach timeouts (P3.3)."""
        p = self.plan
        b = bearing(parts[p.obj][0])
        return b + wrap(parts[p.obj][1] + p.grip_yaw - b, p.symmetry)

    def _bottom_xy(self, parts: dict) -> np.ndarray:
        """xy of the centre of the part's bottom face (parts[obj][2], if given, = world direction of its local z)."""
        p = self.plan
        pos = np.asarray(parts[p.obj][0], dtype=np.float64)
        if len(parts[p.obj]) < 3:
            return pos[:2]
        return pos[:2] - np.asarray(parts[p.obj][2], dtype=np.float64)[:2] * p.half_height

    def _support_top(self, parts: dict) -> float:
        p = self.plan
        if p.support is None:
            return 0.0
        sp = next(q for q in self.plans if q.obj == p.support)
        return float(parts[p.support][0][2]) + sp.half_height

    def _make_place(self, parts: dict) -> dict:
        p, g, prm = self.plan, self._grasp, self.params
        if p.support is None:
            xy, part_yaw = np.asarray(p.target_xy, dtype=np.float64), p.target_yaw
        else:
            s_pos, s_yaw = parts[p.support][:2]
            xy = np.asarray(s_pos[:2], dtype=np.float64) + rot2(s_yaw, p.target_xy)
            part_yaw = s_yaw + p.target_yaw
        # among the equivalent final yaws (part symmetry) reachable by turning the gripper at most 90 deg, the one
        # closest to the arm's bearing at the target (small wrist roll). Without the 90 deg limit the lying bar
        # (180 deg symmetry) was often turned half a turn in transport, up to the wrist roll limit (P3.3).
        b = math.atan2(xy[1], xy[0])
        t0 = wrap(part_yaw - g["part_yaw"], p.symmetry)
        turns = [t0 + k * p.symmetry for k in range(-4, 5) if abs(t0 + k * p.symmetry) <= math.pi / 2 + 1e-9]
        turn = min(turns, key=lambda tr: abs(wrap(g["yaw"] + tr - b, 2 * math.pi)))
        top = self._support_top(parts)
        tcp_z = top + p.grasp_z + prm.grasp_dz + prm.place_drop
        lift = max(prm.lift_min, top + p.grasp_z + prm.grasp_dz + p.lift_clear) + prm.lift_dz
        return {"xy": xy, "z": tcp_z, "lift": lift, "yaw": g["yaw"] + turn, "retreat": p.retreat_z + prm.retreat_dz}

    def _make_rel(self) -> dict:
        """Release point = where the TCP was sent; clearance = away from the fixed finger (gripper -x: the fixed
        finger is on the -x side of the part); slide = in the fingers' plane, toward the robot base."""
        pl, p, prm = self._place, self.plan, self.params
        xy = self._cmd[:2].copy()
        closing = np.array([math.cos(pl["yaw"]), math.sin(pl["yaw"])])
        clear_xy = xy - prm.clear_dist * closing if p.slide > 0 else xy
        u = np.array([-closing[1], closing[0]])
        if float(u @ xy) > 0:
            u = -u
        return {"xy": xy, "clear_xy": clear_xy, "out_xy": clear_xy + p.slide * u}

    def _goal(self, parts: dict) -> tuple[np.ndarray, float, float]:
        p, prm = self.plan, self.params
        pos = np.asarray(parts[p.obj][0], dtype=np.float64)
        off = np.asarray(prm.grasp_offset_xy, dtype=np.float64)
        bottom = pos[2] - p.half_height
        ph = self.phase
        if ph == "approach":
            return np.array([*(pos[:2] + off), bottom + p.approach_z + prm.approach_dz]), prm.gripper_open, self._yaw
        g = self._grasp
        if ph == "descend":
            return np.array([*g["xy"], g["bottom"] + p.grasp_z + prm.grasp_dz]), prm.gripper_open, g["yaw"]
        if ph == "close":
            return np.array([*g["xy"], g["bottom"] + p.grasp_z + prm.grasp_dz]), prm.gripper_closed, g["yaw"]
        if ph == "lift":
            return np.array([*g["xy"], self._place["lift"]]), prm.gripper_closed, g["yaw"]
        pl = self._place
        # carried part: aim its BOTTOM at the target, not the TCP (it sits a few mm off-centre in the fingers)
        carry_xy = pl["xy"] + (self._hold_off if self._hold_off is not None else 0.0)
        if ph == "transport":
            return np.array([*carry_xy, pl["lift"]]), prm.gripper_closed, pl["yaw"]
        if ph == "lower":
            return np.array([*carry_xy, pl["z"]]), prm.gripper_closed, pl["yaw"]
        r = self._rel
        if ph == "release":
            return np.array([*r["xy"], pl["z"]]), prm.gripper_release, pl["yaw"]
        if ph == "clear":
            return np.array([*r["clear_xy"], pl["z"]]), prm.gripper_release, pl["yaw"]
        if ph == "slide":
            return np.array([*r["out_xy"], pl["z"]]), prm.gripper_release, pl["yaw"]
        if ph in ("retreat", "done"):
            return np.array([*r["out_xy"], pl["retreat"]]), prm.gripper_release, pl["yaw"]
        raise ValueError(ph)

    # ---------------------------------------------------------------- control
    def _set_phase(self, phase: str) -> None:
        self.phase, self.t_phase = phase, 0.0

    def _step_ready(self, tcp_pos, arm_q, dt) -> ExpertCommand:
        prm = self.params
        ready = np.radians(READY_POSE_DEG)
        if self._q_cmd is None:
            self._q_cmd = np.asarray(arm_q if arm_q is not None else ready, dtype=np.float64).copy()
        self._q_cmd = self._q_cmd + np.clip(ready - self._q_cmd, -prm.joint_speed * dt, prm.joint_speed * dt)
        self.t_phase += dt
        q_now = self._q_cmd if arm_q is None else np.asarray(arm_q, dtype=np.float64)
        cmd = ExpertCommand(tcp_goal=tcp_pos.copy(), yaw=self._yaw, gripper=prm.gripper_open, phase="ready",
                            joint_goal=self._q_cmd.copy())
        if np.max(np.abs(ready - q_now)) < math.radians(prm.ready_tol_deg):
            self._set_phase("approach")
            self._cmd = tcp_pos.copy()
        elif self.t_phase > prm.phase_timeout:
            self.failed, self.failure_reason = True, "timeout in phase 'ready'"
        return cmd

    def step(self, tcp_pos, parts: dict, dt: float, arm_q=None) -> ExpertCommand:
        """``parts``: {obj: (centre position (3,) env frame, yaw rad[, world direction of the part's local z (3,)])}
        measured now. With the direction, the carried part's bottom (not its centre) is aimed at the target."""
        tcp_pos = np.asarray(tcp_pos, dtype=np.float64)
        if self.phase == "ready":
            return self._step_ready(tcp_pos, arm_q, dt)
        p, prm = self.plan, self.params
        if self.phase == "approach":
            self._yaw = self._grasp_yaw(parts)   # follow the part until the descent starts
        if self.phase == "descend" and self._grasp is None:
            pos = np.asarray(parts[p.obj][0], dtype=np.float64)
            self._grasp = {"xy": pos[:2] + np.asarray(prm.grasp_offset_xy), "bottom": pos[2] - p.half_height,
                           "part_yaw": parts[p.obj][1], "yaw": self._yaw}
            self._place = self._make_place(parts)  # lift height needs the support; refreshed at transport
        if self.phase == "transport" and self.t_phase == 0.0:
            self._place = self._make_place(parts)
        if self.phase in ("lift", "transport"):   # measure the part's offset in the grip (frozen during 'lower')
            off = tcp_pos[:2] - self._bottom_xy(parts)
            self._hold_off = off if self._hold_off is None else 0.8 * self._hold_off + 0.2 * off
        if self.phase == "release" and self._rel is None:
            self._rel = self._make_rel()

        goal, grip, yaw = self._goal(parts)
        self.last_goal = goal
        delta = goal - self._cmd
        dist = float(np.linalg.norm(delta))
        max_step = {"slide": prm.slide_speed, "clear": prm.clear_speed}.get(self.phase, prm.speed) * dt
        self._cmd = goal.copy() if dist <= max_step else self._cmd + delta / dist * max_step
        self.t_phase += dt
        tol = prm.descend_tol if self.phase == "descend" else prm.reach_tol
        reached = dist <= max_step and float(np.linalg.norm(goal - tcp_pos)) < tol

        if self.phase == "lift" and reached:
            bottom = float(parts[p.obj][0][2]) - p.half_height
            if bottom - self._grasp["bottom"] < prm.lift_check:   # missed grasp: reopen and start the step over
                if self.regrasps >= prm.max_regrasps:
                    self.failed, self.failure_reason = True, f"step {self.step_index}: grasp missed too many times"
                else:
                    self.regrasps += 1
                    self._grasp = self._place = self._hold_off = None
                    self._set_phase("approach")
                reached = False
        phase_now = self.phase
        if self.phase == "close":
            if self.t_phase >= prm.close_time:
                self._set_phase("lift")
        elif self.phase == "release":
            if self.t_phase >= prm.release_time:
                self._set_phase("clear" if p.slide > 0 else "retreat")
        elif self.phase == "retreat" and reached:
            if self.step_index + 1 < len(self.plans):
                self.step_index += 1
                self._grasp = self._place = self._hold_off = self._rel = None
                self._set_phase("approach")
                self._yaw = yaw
            else:
                self._set_phase("done")
        elif self.phase not in ("done", "retreat") and reached:
            self._set_phase(PHASES[PHASES.index(self.phase) + 1])
        if self.phase != "done" and self.t_phase > prm.phase_timeout:
            self.failed, self.failure_reason = True, f"step {self.step_index}: timeout in phase '{self.phase}'"
        tilt = TILT_WEIGHT_HIGH if phase_now in VERTICAL_PHASES else TILT_WEIGHT_LOW
        return ExpertCommand(tcp_goal=self._cmd.copy(), yaw=yaw, gripper=grip, phase=phase_now, tilt_weight=tilt)
