"""Robot description shared by every module (sim, training, real robot).

An :class:`Embodiment` lists arms, joints (with limits in radians) and cameras.
It is loaded from ``configs/embodiments/<name>.yaml`` and is the single source
of truth for joint order, joint names and camera names.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import yaml

from cta_learn import CONFIGS_DIR

JointKind = Literal["revolute", "gripper"]
LeRobotNorm = Literal["m100_100", "0_100"]


@dataclass(frozen=True)
class JointSpec:
    name: str
    kind: JointKind
    lower: float  # radians
    upper: float  # radians
    lerobot_norm: LeRobotNorm

    def __post_init__(self) -> None:
        if not self.lower < self.upper:
            raise ValueError(f"joint '{self.name}': lower ({self.lower}) must be < upper ({self.upper})")
        if self.kind not in ("revolute", "gripper"):
            raise ValueError(f"joint '{self.name}': unknown kind '{self.kind}'")
        if self.lerobot_norm not in ("m100_100", "0_100"):
            raise ValueError(f"joint '{self.name}': unknown lerobot_norm '{self.lerobot_norm}'")


@dataclass(frozen=True)
class ArmSpec:
    name: str
    ee_body: str
    joints: tuple[JointSpec, ...]


@dataclass(frozen=True)
class CameraSpec:
    name: str
    width: int
    height: int
    fps: int
    mount: str  # "world" or the body the camera is attached to


@dataclass(frozen=True)
class Embodiment:
    name: str
    control_hz: int
    arms: tuple[ArmSpec, ...]
    cameras: tuple[CameraSpec, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.arms:
            raise ValueError("an embodiment needs at least one arm")
        names = self.joint_names
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate joint names in embodiment '{self.name}': {names}")
        cams = [c.name for c in self.cameras]
        if len(set(cams)) != len(cams):
            raise ValueError(f"duplicate camera names in embodiment '{self.name}': {cams}")

    @property
    def joints(self) -> tuple[JointSpec, ...]:
        return tuple(j for arm in self.arms for j in arm.joints)

    @property
    def joint_names(self) -> list[str]:
        """Flat joint names in vector order. Multi-arm names are prefixed: ``left.shoulder_pan``."""
        if len(self.arms) == 1:
            return [j.name for j in self.arms[0].joints]
        return [f"{arm.name}.{j.name}" for arm in self.arms for j in arm.joints]

    @property
    def num_joints(self) -> int:
        return len(self.joints)

    @property
    def lower(self) -> np.ndarray:
        return np.array([j.lower for j in self.joints], dtype=np.float64)

    @property
    def upper(self) -> np.ndarray:
        return np.array([j.upper for j in self.joints], dtype=np.float64)

    @property
    def camera_names(self) -> list[str]:
        return [c.name for c in self.cameras]


def _parse_joint(raw: dict) -> JointSpec:
    lo_deg, hi_deg = raw["limits_deg"]
    return JointSpec(
        name=raw["name"],
        kind=raw["kind"],
        lower=math.radians(float(lo_deg)),
        upper=math.radians(float(hi_deg)),
        lerobot_norm=str(raw["lerobot_norm"]),
    )


def embodiment_from_dict(raw: dict) -> Embodiment:
    arms = tuple(
        ArmSpec(name=a["name"], ee_body=a["ee_body"], joints=tuple(_parse_joint(j) for j in a["joints"]))
        for a in raw["arms"]
    )
    cameras = tuple(CameraSpec(**c) for c in raw.get("cameras", []))
    return Embodiment(name=raw["name"], control_hz=int(raw["control_hz"]), arms=arms, cameras=cameras)


def load_embodiment(name_or_path: str | Path) -> Embodiment:
    """Load an embodiment by name (``"so101"``) or by YAML path."""
    path = Path(name_or_path)
    if path.suffix not in (".yaml", ".yml"):
        path = CONFIGS_DIR / "embodiments" / f"{name_or_path}.yaml"
    with open(path, encoding="utf-8") as f:
        return embodiment_from_dict(yaml.safe_load(f))
