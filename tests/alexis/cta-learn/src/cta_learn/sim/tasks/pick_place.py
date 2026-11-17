"""CTA-SO101-PickPlace-v0: pick a cube and place it in a target zone (P0 task).

Layout (env frame, metres): table top at z = 0, robot base at the origin, workspace
in front of the robot along +X. Policy runs at 30 Hz (physics 60 Hz, decimation 2),
same rate as the cameras. Actions are absolute joint targets in radians, ordered as
``configs/embodiments/so101.yaml``.
"""

from __future__ import annotations

import math

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg, TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg

from cta_learn.core.geometry import look_at_quat, quat_from_euler_deg
from cta_learn.sim import mdp
from cta_learn.sim.robots import SO101, so101_cfg

TASK_ID = "CTA-SO101-PickPlace-v0"
TASK_DESCRIPTION = "Pick up the red cube and place it on the green square."

# ----------------------------------------------------------------- layout
# The SO-101 USD extends along -Y at zero yaw (measured by check_scene: grasp point at y = -0.28 m);
# +90 deg about Z turns its workspace to +X.
ROBOT_YAW_DEG = 90.0
# The USD origin is 0.0301 m BELOW the bottom of the base mesh (measured with a USD bounding box):
# lower the root so the base sits on the table top (z = 0). Checked by check_scene.
ROBOT_BASE_Z = -0.0301
CUBE_SIZE_FOR_TCP = 0.03
# Tool centre point in the gripper body frame (m) = centre of the grasp between the two fingers.
# Measured on the USD finger meshes (2026-11-11, scratch probe "probe_fingers"):
#   fixed finger: tip at z = -0.1044, inner face at x ~ -0.008, y in [-0.007, 0.006]
#   moving jaw (opens along +X): inner face x ~ 0.008 at 0 deg, 0.033 at 20 deg, 0.049 at 35 deg
# -> at 35 deg the gap is ~57 mm, centred at x ~ 0.020; a 30 mm cube is held at ~11 deg.
# TCP = where a 30 mm cube's centre sits when its face is 5 mm from the fixed finger (x = -0.008 + 0.005
# + 0.015 = 0.012), 10 mm above the fingertips. Centring the cube in the open gap failed: the moving jaw
# had to push it ~13 mm and the cube tipped over and escaped (frame sequence, 2026-11-11). Close to the
# fixed finger, the jaw squeezes it almost without pushing. (The LeIsaac "jaw" offset used at first was
# an object-detection point, not the fingertip: it put the TCP 16 mm too high.)
FIXED_FINGER_TIP_Z = -0.1044
FIXED_FINGER_INNER_X = -0.008
TCP_OFFSET = (FIXED_FINGER_INNER_X + 0.005 + CUBE_SIZE_FOR_TCP / 2, 0.0, FIXED_FINGER_TIP_Z + 0.010)
CUBE_SIZE = 0.03
CUBE_MASS = 0.03
CUBE_NOMINAL = (0.22, -0.06)
CUBE_RANGE = {"x": (-0.04, 0.04), "y": (-0.08, 0.08), "yaw": (-math.radians(45), math.radians(45))}
TARGET_SIZE = 0.06
TARGET_NOMINAL = (0.22, 0.12)
TARGET_RANGE = {"x": (-0.04, 0.04), "y": (-0.03, 0.03)}
FRONT_CAM_EYE = (0.60, 0.0, 0.42)
FRONT_CAM_LOOKAT = (0.18, 0.02, 0.0)
FRONT_CAM_QUAT = look_at_quat(FRONT_CAM_EYE, FRONT_CAM_LOOKAT)  # 'world' convention, xyzw
JOINT_ORDER = [j.name for j in SO101.joints]
ARM_JOINTS = [j.name for j in SO101.joints if j.kind == "revolute"]
CAM_W, CAM_H = SO101.cameras[0].width, SO101.cameras[0].height


def _pinhole(focal_length: float) -> sim_utils.PinholeCameraCfg:
    return sim_utils.PinholeCameraCfg(
        focal_length=focal_length,
        focus_distance=400.0,
        horizontal_aperture=36.0,
        clipping_range=(0.01, 20.0),
        lock_camera=True,
    )


@configclass
class PickPlaceSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.75)),
        spawn=sim_utils.GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=1500.0),
    )
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.20, 0.0, -0.02)),
        spawn=sim_utils.CuboidCfg(
            size=(0.80, 0.70, 0.04),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.8),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.55, 0.55)),
        ),
    )
    robot = so101_cfg().replace(
        init_state=so101_cfg().init_state.replace(
            pos=(0.0, 0.0, ROBOT_BASE_Z), rot=quat_from_euler_deg(0.0, 0.0, ROBOT_YAW_DEG)
        )
    )
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(*CUBE_NOMINAL, CUBE_SIZE / 2 + 0.001)),
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE,) * 3,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16, solver_velocity_iteration_count=1, max_depenetration_velocity=1.0
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=CUBE_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.1, 0.1)),
        ),
    )
    target = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(*TARGET_NOMINAL, 0.0011)),
        spawn=sim_utils.CuboidCfg(
            size=(TARGET_SIZE, TARGET_SIZE, 0.002),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.7, 0.2)),
        ),
    )
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        debug_vis=False,
        # NOTE: the FrameTransformer may reorder target frames; always look them up by NAME
        # (cta_learn.sim.mdp.frame_index), never by position.
        target_frames=[
            FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper", name="gripper"),
            # tool centre point: where the fingertips meet when the gripper is closed, fixed in the gripper
            # body frame (measured 2026-11-11: jaw contact point at gripper = -10 deg).
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/gripper", name="tcp", offset=OffsetCfg(pos=TCP_OFFSET)
            ),
        ],
    )
    front = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/front_camera",
        offset=TiledCameraCfg.OffsetCfg(pos=FRONT_CAM_EYE, rot=FRONT_CAM_QUAT, convention="world"),
        data_types=["rgb"],
        spawn=_pinhole(24.0),
        width=CAM_W,
        height=CAM_H,
        update_period=1 / 30.0,
    )
    wrist = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper/wrist_camera",
        # LeIsaac literal (0.0, -0.9954, 0.0, -0.09585) was WXYZ; converted here to XYZW.
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, -0.03), rot=(-0.9954, 0.0, -0.09585, 0.0), convention="ros"),
        data_types=["rgb"],
        spawn=_pinhole(18.0),
        width=CAM_W,
        height=CAM_H,
        update_period=1 / 30.0,
    )


@configclass
class ActionsCfg:
    # absolute joint targets in radians; arm (5) then gripper (1) = embodiment order
    arm = base_mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=ARM_JOINTS, scale=1.0, use_default_offset=False, preserve_order=True
    )
    gripper = base_mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=["gripper"], scale=1.0, use_default_offset=False
    )


_ROBOT_ORDERED = SceneEntityCfg("robot", joint_names=JOINT_ORDER, preserve_order=True)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """What a policy (and the real robot) sees."""

        joint_pos = ObsTerm(func=mdp.joint_pos_ordered, params={"asset_cfg": _ROBOT_ORDERED})
        front = ObsTerm(
            func=base_mdp.image, params={"sensor_cfg": SceneEntityCfg("front"), "data_type": "rgb", "normalize": False}
        )
        wrist = ObsTerm(
            func=base_mdp.image, params={"sensor_cfg": SceneEntityCfg("wrist"), "data_type": "rgb", "normalize": False}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class ExpertCfg(ObsGroup):
        """Privileged state for the scripted expert (P0.3). Never given to a policy."""

        cube_pos = ObsTerm(func=mdp.object_pos_env, params={"asset_cfg": SceneEntityCfg("cube")})
        cube_quat = ObsTerm(func=mdp.object_quat, params={"asset_cfg": SceneEntityCfg("cube")})
        target_pos = ObsTerm(func=mdp.object_pos_env, params={"asset_cfg": SceneEntityCfg("target")})
        tcp_pos = ObsTerm(func=mdp.frame_pos_env, params={"frame_cfg": SceneEntityCfg("ee_frame"), "name": "tcp"})
        gripper_quat = ObsTerm(
            func=mdp.frame_quat, params={"frame_cfg": SceneEntityCfg("ee_frame"), "name": "gripper"}
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_ordered, params={"asset_cfg": _ROBOT_ORDERED})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    expert: ExpertCfg = ExpertCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")
    cube = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": CUBE_RANGE, "velocity_range": {}, "asset_cfg": SceneEntityCfg("cube")},
    )
    target = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": TARGET_RANGE, "velocity_range": {}, "asset_cfg": SceneEntityCfg("target")},
    )
    # (tested 2026-11-11: finger friction 1.5 via randomize_rigid_body_material did not help: 0/10)
    light = EventTerm(
        func=mdp.randomize_dome_light,
        mode="reset",
        params={"prim_path": "/World/Light", "intensity_range": (700.0, 2500.0), "color_range": (0.8, 1.0)},
    )
    table_color = EventTerm(
        func=mdp.randomize_shader_color,
        mode="reset",
        params={"prim_suffix": "Table", "color_low": (0.25, 0.25, 0.25), "color_high": (0.75, 0.72, 0.68)},
    )
    front_camera = EventTerm(
        func=mdp.jitter_camera_pose,
        mode="reset",
        params={
            "camera_cfg": SceneEntityCfg("front"),
            "nominal_pos": FRONT_CAM_EYE,
            "nominal_quat": FRONT_CAM_QUAT,
            "pos_range": 0.015,
            "rot_range_deg": 2.0,
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    success = DoneTerm(
        func=mdp.object_placed_in_target,
        params={
            "object_cfg": SceneEntityCfg("cube"),
            "target_cfg": SceneEntityCfg("target"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        },
    )
    cube_dropped = DoneTerm(func=mdp.object_off_table, params={"object_cfg": SceneEntityCfg("cube")})


@configclass
class RewardsCfg:
    """No rewards: this task is for imitation learning and evaluation."""


@configclass
class PickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    scene: PickPlaceSceneCfg = PickPlaceSceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    rewards: RewardsCfg = RewardsCfg()

    task_description: str = TASK_DESCRIPTION

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 20.0
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
        # "balanced" keeps realistic shading at ~30 policy steps/s with 1 env on the RTX 4070 SUPER;
        # "performance" is ~3 % faster but flatter (measured 2026-11-11, outputs/p0_2).
        self.sim.render.rendering_mode = "balanced"
        self.sim.physics = PhysxCfg(bounce_threshold_velocity=0.01, friction_correlation_distance=0.00625)
        self.viewer.eye = (0.75, -0.55, 0.55)
        self.viewer.lookat = (0.15, 0.0, 0.0)
