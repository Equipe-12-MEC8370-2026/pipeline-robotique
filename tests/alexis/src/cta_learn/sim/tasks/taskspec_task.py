"""CTA-SO101-TaskSpec-v0: the scene of ANY task, built from its SO-101 layout (no code per task).

The layout (configs/tasks/so101_<task_id>.yaml) is written by scripts/sim/make_robot_layout.py from the TaskSpec and
its adaptation config (cta_learn.retarget.layout). Pick it with the environment variable CTA_LAYOUT (path, absolute
or relative to the repo; default: the profile task). For each part of the layout: a box (size, mass, aluminium
look), a reset event placing it uniformly in its start range, a privileged observation (position, quaternion) for
the scripted expert, an off-table termination. Success: cta_learn.sim.mdp.taskspec_success (the layout's goals,
parts at rest and gripper away for 1 s). Same table, robot, cameras, rates and action space as P0 / P3; the P3
profile scene (profile_stack.py) is left as it is (published results).
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
import yaml
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg

from cta_learn import CONFIGS_DIR, REPO_ROOT
from cta_learn.core.geometry import quat_from_euler_deg
from cta_learn.sim import mdp
from cta_learn.sim.tasks import pick_place as p0

TASK_ID = "CTA-SO101-TaskSpec-v0"
DEFAULT_LAYOUT = CONFIGS_DIR / "tasks" / "so101_cta_profile_upright.yaml"
ALU = (0.78, 0.79, 0.82)
SUCCESS_HOLD_STEPS = 30
SECONDS_PER_STEP = 20.0   # P0: 20 s for one pick-and-place, P3: 40 s for two
_P0_SCENE = p0.PickPlaceSceneCfg(num_envs=1, env_spacing=2.0)
_P0_EVENTS = p0.EventCfg()
_ROBOT_ORDERED = SceneEntityCfg("robot", joint_names=p0.JOINT_ORDER, preserve_order=True)


def layout_path() -> Path:
    p = Path(os.environ.get("CTA_LAYOUT", DEFAULT_LAYOUT))
    return p if p.is_absolute() else REPO_ROOT / p


def load_layout(path: Path | None = None) -> dict:
    return yaml.safe_load((path or layout_path()).read_text(encoding="utf-8"))


def _range(o: dict, key: str = "range") -> dict:
    dx, dy, dyaw = o[key]
    return {"x": (-dx, dx), "y": (-dy, dy), "yaw": (-dyaw, dyaw)}


def prim_name(name: str) -> str:
    return "".join(w.capitalize() for w in name.split("_"))


def part_cfg(name: str, o: dict) -> RigidObjectCfg:
    x, y, yaw = o["start_xyyaw"]
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + prim_name(name),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(x, y, o["size_m"][2] / 2 + 0.001),
                                                  rot=quat_from_euler_deg(0.0, 0.0, math.degrees(yaw))),
        spawn=sim_utils.CuboidCfg(
            size=tuple(o["size_m"]),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16, solver_velocity_iteration_count=1, max_depenetration_velocity=1.0
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=o["mass_kg"]),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=ALU, metallic=0.6, roughness=0.35),
        ),
    )


@configclass
class TaskSpecSceneCfg(InteractiveSceneCfg):
    ground = _P0_SCENE.ground
    light = _P0_SCENE.light
    table = _P0_SCENE.table
    robot = _P0_SCENE.robot
    # parts: added by TaskSpecEnvCfg.__post_init__ (one attribute per part of the layout)
    ee_frame = FrameTransformerCfg(prim_path="{ENV_REGEX_NS}/Robot/base", debug_vis=False, target_frames=[])
    front = _P0_SCENE.front
    wrist = _P0_SCENE.wrist


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_ordered, params={"asset_cfg": _ROBOT_ORDERED})
        front = ObsTerm(func=base_mdp.image, params={"sensor_cfg": SceneEntityCfg("front"), "data_type": "rgb", "normalize": False})
        wrist = ObsTerm(func=base_mdp.image, params={"sensor_cfg": SceneEntityCfg("wrist"), "data_type": "rgb", "normalize": False})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class ExpertCfg(ObsGroup):
        """Privileged state for the scripted expert (parts added per layout). Never given to a policy."""

        tcp_pos = ObsTerm(func=mdp.frame_pos_env, params={"frame_cfg": SceneEntityCfg("ee_frame"), "name": "tcp"})
        gripper_quat = ObsTerm(func=mdp.frame_quat, params={"frame_cfg": SceneEntityCfg("ee_frame"), "name": "gripper"})
        joint_pos = ObsTerm(func=mdp.joint_pos_ordered, params={"asset_cfg": _ROBOT_ORDERED})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    expert: ExpertCfg = ExpertCfg()


@configclass
class EventCfg:
    # robot only; the part events, then forget_contacts (must come AFTER them) and the visual randomisations are
    # appended in TaskSpecEnvCfg.__post_init__ (the event manager runs them in attribute order)
    reset_robot = EventTerm(func=mdp.reset_articulations_to_default, mode="reset")


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)


@configclass
class RewardsCfg:
    """No rewards: imitation learning and evaluation only."""


@configclass
class TaskSpecEnvCfg(ManagerBasedRLEnvCfg):
    scene: TaskSpecSceneCfg = TaskSpecSceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: p0.ActionsCfg = p0.ActionsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    task_description: str = ""

    def use_ood_ranges(self) -> None:
        """Out-of-range evaluation trials: wider start ranges."""
        for name in self.layout["order"]:
            getattr(self.events, f"place_{name}").params["pose_range"] = _range(self.layout["objects"][name], "range_ood")

    def __post_init__(self) -> None:
        self.layout = load_layout()
        lay = self.layout
        names = list(lay["order"])
        self.task_description = lay["instruction"]["en"]
        # TCP: the part's face 5 mm from the fixed finger (P0 rule), 10 mm above the fingertips
        tcp = (p0.FIXED_FINGER_INNER_X + 0.005 + lay["grasp_width_m"] / 2, 0.0, p0.FIXED_FINGER_TIP_Z + 0.010)
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base", debug_vis=False,
            target_frames=[FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper", name="gripper"),
                           FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper", name="tcp",
                                                        offset=OffsetCfg(pos=tcp))])
        for n in names:
            o = lay["objects"][n]
            setattr(self.scene, n, part_cfg(n, o))
            setattr(self.observations.expert, f"{n}_pos", ObsTerm(func=mdp.object_pos_env, params={"asset_cfg": SceneEntityCfg(n)}))
            setattr(self.observations.expert, f"{n}_quat", ObsTerm(func=mdp.object_quat, params={"asset_cfg": SceneEntityCfg(n)}))
            g = lay["goals"][n]
            if g["relation"] == "on_table":
                setattr(self.observations.expert, f"{n}_goal_xy", ObsTerm(func=mdp.goal_xy_const, params={"xy": tuple(g["xyyaw"][:2])}))
            setattr(self.events, f"place_{n}", EventTerm(func=base_mdp.reset_root_state_uniform, mode="reset",
                    params={"pose_range": _range(o), "velocity_range": {}, "asset_cfg": SceneEntityCfg(n)}))
            setattr(self.terminations, f"{n}_dropped", DoneTerm(func=mdp.object_off_table, params={"object_cfg": SceneEntityCfg(n)}))
        self.events.forget_contacts = EventTerm(func=mdp.break_rigid_contacts, mode="reset", params={"asset_names": tuple(names)})
        self.events.light = _P0_EVENTS.light
        self.events.table_color = EventTerm(func=mdp.randomize_shader_color, mode="reset",
                                            params={"prim_suffix": "Table", "color_low": (0.22, 0.22, 0.22), "color_high": (0.52, 0.50, 0.48)})
        self.events.front_camera = _P0_EVENTS.front_camera
        self.terminations.success = DoneTerm(func=mdp.taskspec_success,
                                             params={"ee_frame_cfg": SceneEntityCfg("ee_frame"), "layout": lay,
                                                     "hold_steps": SUCCESS_HOLD_STEPS})
        self.decimation = 2
        self.episode_length_s = SECONDS_PER_STEP * len(names)
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
        self.sim.render.rendering_mode = "balanced"
        self.sim.physics = PhysxCfg(bounce_threshold_velocity=0.01, friction_correlation_distance=0.00625)
        self.viewer.eye = (0.75, -0.55, 0.55)
        self.viewer.lookat = (0.15, 0.0, 0.0)
