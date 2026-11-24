"""CTA-SO101-PickPlaceReal-v0: the P0 cube task made to look like the user's real setup (P4, 2026-11-24).

Same layout, robot, physics, expert and success criterion as CTA-SO101-PickPlace-v0 (``pick_place.py``, unchanged:
its results are published). Differences, all visual:
  - printed wrist camera mount (+ simplified camera) on the gripper, placed from geometry (``sim/wrist_mount.py``);
  - wrist camera moved to the centre of that mount, looking toward the fingertips; its pose is jittered per episode
    (lens height and assembly are estimates); still a 90 deg pinhole (the real fisheye image is undistorted to it);
  - orange cube, white paper zone, wooden table (texture cropped from the user's photo), colours measured on the
    user's photos and lightly randomised;
  - front camera (same IMX335 fisheye as the wrist, undistorted on the real side to a 90 deg pinhole) moved closer,
    option "D" chosen by the user on renders (2026-11-24): the real table is not deep. Jittered as in P0.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from cta_learn import REPO_ROOT
from cta_learn.core.geometry import look_at_quat
from cta_learn.sim import mdp
from cta_learn.sim.tasks.pick_place import EventCfg, PickPlaceEnvCfg, PickPlaceSceneCfg, _pinhole
from cta_learn.sim.wrist_mount import MOUNT_USD, WRIST_JITTER, mat_to_quat_xyzw, mount_transform, wrist_camera_offset

TASK_ID = "CTA-SO101-PickPlaceReal-v0"
TASK_DESCRIPTION = "Pick up the orange cube and place it on the white square."
TABLE_WOOD_USD = REPO_ROOT / "assets" / "props" / "table_wood.usd"

# Linear RGB (UsdPreviewSurface). Cube: orange PLA spool photo, median sRGB (236, 98, 1) -> linear (0.84, 0.12, 0.0003).
CUBE_ORANGE = (0.84, 0.12, 0.003)
CUBE_ORANGE_LOW, CUBE_ORANGE_HIGH = (0.70, 0.08, 0.0), (0.95, 0.20, 0.02)
# Zone: plain white printer paper (not photographed): off-white, randomised in brightness.
PAPER_LOW, PAPER_HIGH = (0.62, 0.62, 0.58), (0.92, 0.92, 0.90)
PAPER_WHITE = (0.80, 0.80, 0.77)

_T = mount_transform()
MOUNT_POS = tuple(float(v) for v in _T[:3, 3])
MOUNT_QUAT = mat_to_quat_xyzw(_T[:3, :3])
WRIST_POS, WRIST_QUAT_ROS = wrist_camera_offset()
# front camera, option D (env frame, m): 20 cm closer than P0 (0.60, 0, 0.42), aimed at the workspace centre
FRONT_CAM_EYE = (0.40, 0.0, 0.38)
FRONT_CAM_LOOKAT = (0.22, 0.02, 0.0)
FRONT_CAM_QUAT = look_at_quat(FRONT_CAM_EYE, FRONT_CAM_LOOKAT)  # 'world' convention, xyzw
FRONT_FOCAL_MM = 18.0  # 36 mm aperture -> 90 deg horizontal, like the wrist camera (P0 front: 24 mm, 74 deg)


@configclass
class PickPlaceRealSceneCfg(PickPlaceSceneCfg):
    table_wood = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableWood",
        # 0.2 mm above the physical table top (z = 0): visual only, the cube rests on the cuboid underneath
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.20, 0.0, 0.0002)),
        spawn=sim_utils.UsdFileCfg(usd_path=str(TABLE_WOOD_USD)),
    )
    camera_mount = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper/camera_mount",
        init_state=AssetBaseCfg.InitialStateCfg(pos=MOUNT_POS, rot=MOUNT_QUAT),
        spawn=sim_utils.UsdFileCfg(usd_path=str(MOUNT_USD)),
    )

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        self.cube.spawn.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=CUBE_ORANGE)
        self.target.spawn.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=PAPER_WHITE, roughness=0.9)
        self.wrist.offset = TiledCameraCfg.OffsetCfg(pos=WRIST_POS, rot=WRIST_QUAT_ROS, convention="ros")
        self.front.offset = TiledCameraCfg.OffsetCfg(pos=FRONT_CAM_EYE, rot=FRONT_CAM_QUAT, convention="world")
        self.front.spawn = _pinhole(FRONT_FOCAL_MM)


@configclass
class PickPlaceRealEventCfg(EventCfg):
    table_color = None  # the grey cuboid is hidden under the wooden top
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
    table_wood = EventTerm(
        func=mdp.randomize_texture_scale,
        mode="reset",
        params={"prim_suffix": "TableWood", "brightness_range": (0.75, 1.3), "tint_range": 0.08},
    )
    cube_color = EventTerm(
        func=mdp.randomize_shader_color,
        mode="reset",
        params={"prim_suffix": "Cube", "color_low": CUBE_ORANGE_LOW, "color_high": CUBE_ORANGE_HIGH},
    )
    paper_color = EventTerm(
        func=mdp.randomize_shader_color,
        mode="reset",
        params={"prim_suffix": "Target", "color_low": PAPER_LOW, "color_high": PAPER_HIGH},
    )
    # the real arm is printed in white PLA (user's photo, Thingiverse photos); the LeIsaac USD is yellow
    robot_color = EventTerm(
        func=mdp.randomize_mdl_color,
        mode="reset",
        params={
            "shader_suffix": "Robot/Looks/material_a_3d_printed/Shader",
            "input_name": "diffuse_color_constant",
            "color_low": (0.72, 0.72, 0.70),
            "color_high": (0.92, 0.92, 0.90),
        },
    )
    wrist_camera = EventTerm(
        func=mdp.jitter_attached_camera,
        mode="reset",
        params={
            "camera_cfg": SceneEntityCfg("wrist"),
            "nominal_pos": WRIST_POS,
            "nominal_quat_ros": WRIST_QUAT_ROS,
            "pos_range": WRIST_JITTER["pos_m"],
            "along_axis_range": WRIST_JITTER["along_axis_m"],
            "rot_range_deg": WRIST_JITTER["rot_deg"],
        },
    )


@configclass
class PickPlaceRealEnvCfg(PickPlaceEnvCfg):
    scene: PickPlaceRealSceneCfg = PickPlaceRealSceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=False)
    events: PickPlaceRealEventCfg = PickPlaceRealEventCfg()
    task_description: str = TASK_DESCRIPTION
