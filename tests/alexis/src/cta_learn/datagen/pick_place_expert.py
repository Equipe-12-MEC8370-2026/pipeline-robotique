"""Scripted expert for the pick-and-place task (P0.3).

The expert reads privileged state (cube, target and TCP positions) and outputs, at each
policy step, either a joint-space goal (phase ``ready``) or a Cartesian goal for the tool
centre point (TCP) with a gripper yaw, plus a gripper command. Joint angles for Cartesian
goals are computed by the simulator-side IK (``cta_learn.sim.ik``). The logic here is
simulator-free and unit-testable.

Why a ``ready`` phase: from the folded start pose, a local IK falls into the folded
(elbow-up) branch where a vertical gripper cannot reach the workspace (measured on a
forward-kinematics grid, 2026-11-11). Moving first, in joint space, to a pose of the
extended branch keeps the IK on solutions that reach the whole cube area.

Later (architecture C, P3), the hard-coded sequence of waypoints below is what a
``TaskSpec`` extracted from human video will provide.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

PHASES = ("ready", "approach", "descend", "close", "lift", "transport", "lower", "release", "retreat", "done")
CUBE_HALF = 0.015
# Extended-branch pose, gripper vertical, TCP at ~(0.22, 0.02, 0.09) m (pan, lift, elbow, wrist_flex, wrist_roll).
READY_POSE_DEG = (0.0, -10.0, 20.0, 75.0, 0.0)


@dataclass
class ExpertParams:
    """Per-episode parameters; :meth:`sample` draws variations so demonstrations are not all identical."""

    approach_height: float = 0.05     # TCP height above the cube top before descending (m)
    grasp_height: float = 0.012       # TCP height (absolute, table top = 0) when closing (m): fingertips
                                      # 2 mm above the table, gripping the lower 2/3 of the cube
    grasp_offset_xy: tuple[float, float] = (0.0, 0.0)
    lift_height: float = 0.07         # TCP height while transporting (m); > ~0.10 hits the wrist limit
    place_height: float = 0.030       # TCP height when releasing (m)
    retreat_height: float = 0.08      # TCP height after release; the success term needs the TCP > 5 cm from
                                      # the cube centre (z = 0.015), so it must be > 0.065: P0.3 drew it in
                                      # [0.06, 0.08] and every TCP below ~0.064 timed out (fixed 2026-11-12)
    speed: float = 0.14               # max TCP speed (m/s)
    joint_speed: float = 0.5          # max joint speed in the ready phase (rad/s), EE-3.1
    gripper_open: float = math.radians(35.0)
    gripper_closed: float = math.radians(-10.0)
    gripper_release: float = math.radians(60.0)  # open wider to release (frees a wedged part)
    close_time: float = 1.2           # wait while closing (s); the 1 N.m gripper needs ~0.5 s
    release_time: float = 0.8
    reach_tol: float = 0.008          # TCP-to-goal distance to consider a waypoint reached (m)
    descend_tol: float = 0.004        # tighter for the final descent: the cube must sit 5 mm from the fixed finger
    ready_tol_deg: float = 3.0
    phase_timeout: float = 6.0        # s spent in one phase before declaring failure
    lift_check: float = 0.015         # after 'lift', the cube must be this much above the table, else regrasp
    max_regrasps: int = 2
    first_grasp_offset_xy: tuple[float, float] = (0.0, 0.0)  # extra offset on the FIRST grasp only: injects a
                                                             # missed grasp to generate recovery demos

    @classmethod
    def sample(cls, rng: np.random.Generator) -> "ExpertParams":
        return cls(
            approach_height=rng.uniform(0.04, 0.06),
            grasp_height=0.012 + rng.uniform(-0.002, 0.002),
            grasp_offset_xy=tuple(rng.uniform(-0.003, 0.003, size=2)),
            lift_height=rng.uniform(0.06, 0.08),
            place_height=rng.uniform(0.025, 0.040),
            retreat_height=rng.uniform(0.075, 0.09),
            speed=rng.uniform(0.10, 0.18),
        )


def wrap_to_quarter_turn(yaw: float) -> float:
    """A square cube looks the same every 90 deg: map its yaw to [-45, 45) deg."""
    return (yaw + math.pi / 4) % (math.pi / 2) - math.pi / 4


def yaw_from_quat_xyzw(q) -> float:
    x, y, z, w = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


@dataclass
class ExpertCommand:
    tcp_goal: np.ndarray            # (3,) env frame, metres (ignored when joint_goal is set)
    yaw: float                      # rad, world yaw of the gripper closing axis
    gripper: float                  # rad, joint target
    phase: str
    joint_goal: np.ndarray | None = None  # (5,) arm joint targets in rad (ready phase only)
    tilt_weight: float = 0.03       # how strongly the IK keeps the gripper vertical (high near the table)


# Phases where the gripper must be vertical (close to the cube / target); elsewhere verticality is relaxed.
VERTICAL_PHASES = ("descend", "close", "lower", "release")
TILT_WEIGHT_HIGH, TILT_WEIGHT_LOW = 0.3, 0.03


@dataclass
class PickPlaceExpert:
    params: ExpertParams = field(default_factory=ExpertParams)
    phase: str = "ready"
    t_phase: float = 0.0
    failed: bool = False
    failure_reason: str = ""
    _cmd: np.ndarray | None = None      # moving Cartesian command (rate-limited toward the goal)
    _q_cmd: np.ndarray | None = None    # moving joint command (ready phase)
    _yaw: float = 0.0
    _grasp_xy: np.ndarray | None = None
    regrasps: int = 0                   # missed grasps detected (and retried) in this episode

    def reset(self, tcp_pos, cube_pos, cube_yaw: float, params: ExpertParams | None = None, arm_q=None) -> None:
        if params is not None:
            self.params = params
        self.phase, self.t_phase, self.failed, self.failure_reason = "ready", 0.0, False, ""
        self._cmd = np.asarray(tcp_pos, dtype=np.float64).copy()
        self._q_cmd = None if arm_q is None else np.asarray(arm_q, dtype=np.float64).copy()
        self._yaw = wrap_to_quarter_turn(cube_yaw)
        self._grasp_xy = None
        self.regrasps = 0

    @property
    def done(self) -> bool:
        return self.phase == "done"

    # ---------------------------------------------------------------- core
    def _goal(self, cube_pos: np.ndarray, target_pos: np.ndarray) -> tuple[np.ndarray, float]:
        p = self.params
        cube_xy = cube_pos[:2] + self._offset()
        if self.phase == "approach":
            return np.array([*cube_xy, 2 * CUBE_HALF + p.approach_height]), p.gripper_open
        if self.phase == "descend":
            return np.array([*cube_xy, p.grasp_height]), p.gripper_open
        if self.phase == "close":
            return np.array([*self._grasp_xy, p.grasp_height]), p.gripper_closed
        if self.phase == "lift":
            return np.array([*self._grasp_xy, p.lift_height]), p.gripper_closed
        if self.phase == "transport":
            return np.array([*target_pos[:2], p.lift_height]), p.gripper_closed
        if self.phase == "lower":
            return np.array([*target_pos[:2], p.place_height]), p.gripper_closed
        if self.phase == "release":
            return np.array([*target_pos[:2], p.place_height]), p.gripper_release
        if self.phase in ("retreat", "done"):
            return np.array([*target_pos[:2], p.retreat_height]), p.gripper_release
        raise ValueError(self.phase)

    def _offset(self) -> np.ndarray:
        off = np.asarray(self.params.grasp_offset_xy, dtype=np.float64)
        return off + np.asarray(self.params.first_grasp_offset_xy) if self.regrasps == 0 else off

    def _next_phase(self) -> None:
        i = PHASES.index(self.phase)
        self.phase = PHASES[min(i + 1, len(PHASES) - 1)]
        self.t_phase = 0.0

    def _step_ready(self, tcp_pos: np.ndarray, arm_q, dt: float) -> ExpertCommand:
        p = self.params
        ready = np.radians(READY_POSE_DEG)
        if self._q_cmd is None:
            self._q_cmd = np.asarray(arm_q if arm_q is not None else ready, dtype=np.float64).copy()
        delta = ready - self._q_cmd
        self._q_cmd = self._q_cmd + np.clip(delta, -p.joint_speed * dt, p.joint_speed * dt)
        self.t_phase += dt
        q_now = self._q_cmd if arm_q is None else np.asarray(arm_q, dtype=np.float64)
        cmd = ExpertCommand(tcp_goal=tcp_pos.copy(), yaw=self._yaw, gripper=p.gripper_open, phase="ready",
                            joint_goal=self._q_cmd.copy())
        if np.max(np.abs(ready - q_now)) < math.radians(p.ready_tol_deg):
            self._next_phase()
            self._cmd = tcp_pos.copy()  # Cartesian control starts from where the TCP is
        elif self.t_phase > p.phase_timeout:
            self.failed, self.failure_reason = True, "timeout in phase 'ready'"
        return cmd

    def step(self, tcp_pos, cube_pos, target_pos, dt: float, arm_q=None, cube_yaw: float | None = None) -> ExpertCommand:
        tcp_pos, cube_pos, target_pos = (np.asarray(v, dtype=np.float64) for v in (tcp_pos, cube_pos, target_pos))
        if self.phase == "ready":
            return self._step_ready(tcp_pos, arm_q, dt)
        p = self.params
        if self.phase in ("descend", "close") and self._grasp_xy is None:
            self._grasp_xy = cube_pos[:2] + self._offset()

        goal, grip = self._goal(cube_pos, target_pos)
        delta = goal - self._cmd
        dist = float(np.linalg.norm(delta))
        max_step = p.speed * dt
        self._cmd = goal.copy() if dist <= max_step else self._cmd + delta / dist * max_step

        self.t_phase += dt
        tol = p.descend_tol if self.phase == "descend" else p.reach_tol
        reached = dist <= max_step and float(np.linalg.norm(goal - tcp_pos)) < tol
        if self.phase == "lift" and reached and cube_pos[2] - CUBE_HALF < p.lift_check:
            # missed grasp: the TCP is up but the cube stayed on the table -> reopen and start over
            if self.regrasps >= p.max_regrasps:
                self.failed, self.failure_reason = True, "grasp missed too many times"
            else:
                self.regrasps += 1
                self.phase, self.t_phase, self._grasp_xy = "approach", 0.0, None
                if cube_yaw is not None:
                    self._yaw = wrap_to_quarter_turn(cube_yaw)
            reached = False
        if self.phase == "close":
            if self.t_phase >= p.close_time:
                self._next_phase()
        elif self.phase == "release":
            if self.t_phase >= p.release_time:
                self._next_phase()
        elif self.phase != "done" and reached:
            self._next_phase()
        if self.phase != "done" and self.t_phase > p.phase_timeout:
            self.failed, self.failure_reason = True, f"timeout in phase '{self.phase}'"
        tilt = TILT_WEIGHT_HIGH if self.phase in VERTICAL_PHASES else TILT_WEIGHT_LOW
        return ExpertCommand(tcp_goal=self._cmd.copy(), yaw=self._yaw, gripper=grip, phase=self.phase, tilt_weight=tilt)
