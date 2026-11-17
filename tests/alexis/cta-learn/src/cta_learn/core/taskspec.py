"""TaskSpec — robot-independent description of a demonstrated task (contract between the video half and the
robot half of architecture C, ADR-001).

Plain dataclasses + JSON (stdlib only) so the same file is read in .venv-learn, .venv-perception and
isaacsim-env (where nothing is installed). Hand-editable: a wrong extraction is fixed by editing the JSON,
then ``TaskSpec.load`` re-validates it.

Frame: the table frame of the TPV calibration (origin front-right corner, x back, y left, z up), metres,
degrees. Goals are stored twice: the absolute pose seen in the demo, and the relation that generalises
(e.g. "upright on top of profile_short, offset (dx, dy) in its frame") — P3 randomises scenes and keeps relations.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCHEMA_VERSION = "0.1"
POSTURES = ("lying", "upright")
PHASES = ("reach", "grasp", "transport", "place", "release", "retreat")


class TaskSpecError(ValueError):
    pass


@dataclass
class Pose:
    x: float
    y: float
    yaw_deg: float
    posture: str            # "lying" | "upright"
    z_base: float = 0.0     # height of the part's base above the table

    def validate(self, where: str) -> None:
        if self.posture not in POSTURES:
            raise TaskSpecError(f"{where}: posture must be one of {POSTURES}, got {self.posture!r}")
        if self.z_base < 0:
            raise TaskSpecError(f"{where}: z_base < 0")


@dataclass
class ObjectSpec:
    id: str
    shape: str                  # "box" (bounding box of the part)
    size_m: list[float]         # [length, width, height] of the part, length = longest
    mass_kg: float
    initial: Pose
    description: str = ""

    def validate(self) -> None:
        if self.shape != "box":
            raise TaskSpecError(f"object {self.id}: only shape 'box' is supported")
        if len(self.size_m) != 3 or min(self.size_m) <= 0:
            raise TaskSpecError(f"object {self.id}: size_m must be 3 positive numbers")
        if self.mass_kg <= 0:
            raise TaskSpecError(f"object {self.id}: mass_kg must be > 0")
        self.initial.validate(f"object {self.id} initial")


@dataclass
class Relation:
    type: str                   # "on_table" | "on_top_of"
    ref: str | None = None      # object id for on_top_of
    offset_xy_m: list[float] | None = None   # centre offset in the ref object's frame (x along its length)
    yaw_rel_deg: float | None = None         # yaw relative to the ref object


@dataclass
class Goal:
    pose: Pose                  # absolute, as demonstrated
    relation: Relation
    tolerance_xy_m: float = 0.02
    tolerance_yaw_deg: float = 15.0


@dataclass
class Phase:
    name: str                   # one of PHASES
    t: list[float]              # [start, end] seconds in the synced clips


@dataclass
class Step:
    index: int
    object: str
    hand: str                   # "left" | "right" | "unknown" (the demonstrator's hand)
    hand_confidence: float
    t_grasp: float
    t_release: float
    phases: list[Phase]
    goal: Goal


@dataclass
class TaskSpec:
    task_id: str
    instruction: dict[str, str]                 # {"fr": ..., "en": ...}
    frame: str
    objects: list[ObjectSpec]
    steps: list[Step]
    success: list[str]                          # human-readable criteria derived from the goals
    signals: dict[str, list[float] | None] = field(default_factory=dict)  # demonstrator start/end gestures
    source: dict = field(default_factory=dict)  # provenance (pair, files, calibration, versions)
    schema_version: str = SCHEMA_VERSION

    # ---------- validation ----------
    def validate(self) -> "TaskSpec":
        if self.schema_version != SCHEMA_VERSION:
            raise TaskSpecError(f"schema_version {self.schema_version} != {SCHEMA_VERSION}")
        if not self.instruction.get("en"):
            raise TaskSpecError("instruction['en'] is required (text prompt of language-conditioned policies)")
        ids = [o.id for o in self.objects]
        if len(set(ids)) != len(ids):
            raise TaskSpecError("object ids must be unique")
        for o in self.objects:
            o.validate()
        if [s.index for s in self.steps] != list(range(len(self.steps))):
            raise TaskSpecError("steps must be indexed 0..n-1 in order")
        placed: set[str] = set()
        prev_release = -1e9
        for s in self.steps:
            w = f"step {s.index}"
            if s.object not in ids:
                raise TaskSpecError(f"{w}: unknown object {s.object!r}")
            if s.hand not in ("left", "right", "unknown") or not 0 <= s.hand_confidence <= 1:
                raise TaskSpecError(f"{w}: bad hand / hand_confidence")
            if not s.t_grasp < s.t_release:
                raise TaskSpecError(f"{w}: t_grasp must be < t_release")
            if s.t_grasp < prev_release - 1e-6:
                raise TaskSpecError(f"{w}: starts before the previous step's release")
            prev_release = s.t_release
            last = -1e9
            for ph in s.phases:
                if ph.name not in PHASES:
                    raise TaskSpecError(f"{w}: unknown phase {ph.name!r}")
                if not (len(ph.t) == 2 and ph.t[0] <= ph.t[1] and ph.t[0] >= last - 1e-6):
                    raise TaskSpecError(f"{w}: phase {ph.name} times must be ordered and non-overlapping")
                last = ph.t[1]
            s.goal.pose.validate(f"{w} goal")
            r = s.goal.relation
            if r.type == "on_top_of":
                if r.ref not in placed:
                    raise TaskSpecError(f"{w}: on_top_of {r.ref!r} must refer to an object placed by an earlier step")
                if r.offset_xy_m is None or r.yaw_rel_deg is None:
                    raise TaskSpecError(f"{w}: on_top_of needs offset_xy_m and yaw_rel_deg")
            elif r.type != "on_table":
                raise TaskSpecError(f"{w}: relation type must be on_table or on_top_of")
            placed.add(s.object)
        return self

    # ---------- (de)serialisation ----------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TaskSpec":
        def pose(p):
            return Pose(**p)
        objs = [ObjectSpec(**{**o, "initial": pose(o["initial"])}) for o in d["objects"]]
        steps = [Step(**{**s, "phases": [Phase(**p) for p in s["phases"]],
                         "goal": Goal(**{**s["goal"], "pose": pose(s["goal"]["pose"]), "relation": Relation(**s["goal"]["relation"])})})
                 for s in d["steps"]]
        return cls(**{**d, "objects": objs, "steps": steps}).validate()

    def save(self, path: Path) -> None:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "TaskSpec":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
