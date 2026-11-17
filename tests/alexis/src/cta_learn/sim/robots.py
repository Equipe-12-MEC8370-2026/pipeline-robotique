"""Robot articulation configs built from the embodiment YAML + the copied USD."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from cta_learn import REPO_ROOT
from cta_learn.core import load_embodiment

SO101_USD = REPO_ROOT / "assets" / "robots" / "so101" / "so101_follower.usd"
SO101 = load_embodiment("so101")

# Folded start pose, close to LeRobot's SO-101 rest pose, kept inside the joint limits (degrees).
SO101_START_POSE_DEG = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -95.0,
    "elbow_flex": 85.0,
    "wrist_flex": 50.0,
    "wrist_roll": 0.0,
    "gripper": 0.0,
}


def so101_cfg(prim_path: str = "{ENV_REGEX_NS}/Robot") -> ArticulationCfg:
    """SO-101 follower. Actuator gains start from LeIsaac's values (to be tuned against the real servos in P1)."""
    arm_joints = [j.name for j in SO101.joints if j.kind == "revolute"]
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(SO101_USD),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),  # xyzw; overridden by the task
            joint_pos={k: math.radians(v) for k, v in SO101_START_POSE_DEG.items()},
        ),
        actuators={
            "sts3215-arm": ImplicitActuatorCfg(
                joint_names_expr=arm_joints,
                # LeIsaac's stiffness 17.8 N.m/rad let the arm sag ~1 deg under gravity and the wrist be
                # pushed back ~12 mm by the grasp force, so the jaw swept past the cube (2026-11-11).
                # 80 N.m/rad with a 3 N.m cap (~STS3215 stall torque) is closer to a position-servoed
                # STS3215. Both values must be identified on the real arm in P1.
                effort_limit_sim=3.0,
                velocity_limit_sim=10.0,
                stiffness=80.0,
                damping=2.0,
            ),
            "sts3215-gripper": ImplicitActuatorCfg(
                joint_names_expr=["gripper"],
                # Grasp bench (scripts/sim/grasp_bench.py, 2026-11-11, 20 trials each, 30 g cube):
                #   2 N.m -> 5 %, 0.5 -> 5 %, 0.2 -> 85 %, 0.1 -> 100 %, 0.07 -> 100 % (40/40 on a 10 g cube).
                # A strong pinch squeezes the cube up into the V-shaped fingers (wedged) or pushes it away;
                # a soft pinch holds it. LeIsaac uses mass / 0.15 N.m for the same reason. On the real STS3215
                # this maps to its torque-limit register (to verify in P1).
                effort_limit_sim=0.1,
                velocity_limit_sim=10.0,
                stiffness=17.8,
                damping=0.60,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )
