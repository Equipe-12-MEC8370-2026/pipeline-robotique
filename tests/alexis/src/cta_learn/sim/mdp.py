"""Custom MDP terms (observations, terminations, events) for cta_learn tasks.

All poses returned here are expressed in the env frame (world minus env origin).
Quaternions are (x, y, z, w).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

# ---------------------------------------------------------------- observations


def object_pos_env(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    obj = env.scene[asset_cfg.name]
    return obj.data.root_pos_w.torch - env.scene.env_origins


def object_quat(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.root_quat_w.torch


def frame_index(frame, name: str) -> int:
    """Index of a FrameTransformer target by name (the sensor may reorder targets, e.g. alphabetically)."""
    names = list(frame.data.target_frame_names)
    if name not in names:
        raise KeyError(f"frame '{name}' not in {names}")
    return names.index(name)


def frame_pos_env(env, frame_cfg: SceneEntityCfg, name: str) -> torch.Tensor:
    frame = env.scene[frame_cfg.name]
    return frame.data.target_pos_w.torch[:, frame_index(frame, name), :] - env.scene.env_origins


def frame_quat(env, frame_cfg: SceneEntityCfg, name: str) -> torch.Tensor:
    frame = env.scene[frame_cfg.name]
    return frame.data.target_quat_w.torch[:, frame_index(frame, name), :]


def joint_pos_ordered(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Joint positions (rad) in the order of ``asset_cfg.joint_names`` (use preserve_order=True)."""
    return env.scene[asset_cfg.name].data.joint_pos.torch[:, asset_cfg.joint_ids]


# ---------------------------------------------------------------- terminations


def object_placed_in_target(
    env,
    object_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    xy_tolerance: float = 0.03,
    max_height: float = 0.03,
    max_speed: float = 0.05,
    release_distance: float = 0.05,
) -> torch.Tensor:
    """Success: object resting on the table inside the target zone, gripper moved away (released)."""
    obj = env.scene[object_cfg.name]
    tgt = env.scene[target_cfg.name]
    obj_pos = obj.data.root_pos_w.torch
    xy_err = torch.linalg.vector_norm(obj_pos[:, :2] - tgt.data.root_pos_w.torch[:, :2], dim=1)
    resting = (obj_pos[:, 2] - env.scene.env_origins[:, 2] < max_height) & (
        torch.linalg.vector_norm(obj.data.root_lin_vel_w.torch, dim=1) < max_speed
    )
    frame = env.scene[ee_frame_cfg.name]
    ee_pos = frame.data.target_pos_w.torch[:, frame_index(frame, "tcp"), :]
    released = torch.linalg.vector_norm(ee_pos - obj_pos, dim=1) > release_distance
    return (xy_err < xy_tolerance) & resting & released


def object_off_table(env, object_cfg: SceneEntityCfg, min_height: float = -0.05) -> torch.Tensor:
    obj = env.scene[object_cfg.name]
    return obj.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2] < min_height


# ---------------------------------------------------------------- events


def randomize_dome_light(
    env,
    env_ids: torch.Tensor | None,
    prim_path: str,
    intensity_range: tuple[float, float],
    color_range: tuple[float, float],
):
    """Global dome light: random intensity and slightly tinted white (shared by all envs)."""
    import omni.usd
    from pxr import Gf

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"light prim not found: {prim_path}")
    lo, hi = intensity_range
    intensity = float(torch.empty(1).uniform_(lo, hi))
    rgb = torch.empty(3).uniform_(*color_range).tolist()
    prim.GetAttribute("inputs:intensity").Set(intensity)
    prim.GetAttribute("inputs:color").Set(Gf.Vec3f(*rgb))


def randomize_shader_color(
    env,
    env_ids: torch.Tensor | None,
    prim_suffix: str,
    color_low: Sequence[float],
    color_high: Sequence[float],
):
    """Set ``inputs:diffuseColor`` of every UsdPreviewSurface under ``/World/envs/env_<i>/<prim_suffix>``."""
    import omni.usd
    from pxr import Gf, Usd, UsdShade

    stage = omni.usd.get_context().get_stage()
    ids = range(env.num_envs) if env_ids is None else env_ids.tolist()
    low, high = torch.tensor(color_low), torch.tensor(color_high)
    for i in ids:
        root = stage.GetPrimAtPath(f"/World/envs/env_{i}/{prim_suffix}")
        if not root.IsValid():
            raise RuntimeError(f"prim not found: /World/envs/env_{i}/{prim_suffix}")
        rgb = (low + torch.rand(3) * (high - low)).tolist()
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdShade.Shader):
                inp = UsdShade.Shader(prim).GetInput("diffuseColor")
                if inp:
                    inp.Set(Gf.Vec3f(*rgb))


def jitter_camera_pose(
    env,
    env_ids: torch.Tensor | None,
    camera_cfg: SceneEntityCfg,
    nominal_pos: Sequence[float],
    nominal_quat: Sequence[float],
    pos_range: float,
    rot_range_deg: float,
):
    """Re-place a world-mounted camera around its nominal pose (env frame, 'world' convention)."""
    cam = env.scene[camera_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = len(env_ids)
    dev = env.device
    pos = torch.tensor(nominal_pos, device=dev).repeat(n, 1)
    pos += (torch.rand(n, 3, device=dev) * 2 - 1) * pos_range
    pos += env.scene.env_origins[env_ids]
    ang = (torch.rand(n, 3, device=dev) * 2 - 1) * torch.deg2rad(torch.tensor(rot_range_deg, device=dev))
    dq = math_utils.quat_from_euler_xyz(ang[:, 0], ang[:, 1], ang[:, 2])
    quat = math_utils.quat_mul(torch.tensor(nominal_quat, device=dev).repeat(n, 1), dq)
    cam.set_world_poses(pos, quat, env_ids=env_ids, convention="world")


def jitter_attached_camera(
    env,
    env_ids: torch.Tensor | None,
    camera_cfg: SceneEntityCfg,
    nominal_pos: Sequence[float],
    nominal_quat_ros: Sequence[float],
    pos_range: float,
    along_axis_range: float,
    rot_range_deg: float,
):
    """Re-place a camera mounted on a robot link around its nominal LOCAL pose (parent-link frame, ROS convention:
    +Z forward). Position: uniform cube +-pos_range plus +-along_axis_range along the optical axis; rotation: small
    XYZ Euler angles +-rot_range_deg in the camera frame. Used for the P4 wrist camera, whose real pose is only
    estimated (lens height, print and assembly tolerances)."""
    import warp as wp
    from isaaclab.utils.math import convert_camera_frame_orientation_convention

    cam = env.scene[camera_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = len(env_ids)
    dev = env.device
    q0 = torch.tensor(nominal_quat_ros, device=dev, dtype=torch.float32).repeat(n, 1)
    fwd = math_utils.quat_apply(q0, torch.tensor([0.0, 0.0, 1.0], device=dev).repeat(n, 1))
    pos = torch.tensor(nominal_pos, device=dev, dtype=torch.float32).repeat(n, 1)
    pos += (torch.rand(n, 3, device=dev) * 2 - 1) * pos_range
    pos += fwd * ((torch.rand(n, 1, device=dev) * 2 - 1) * along_axis_range)
    ang = (torch.rand(n, 3, device=dev) * 2 - 1) * math.radians(rot_range_deg)
    quat = math_utils.quat_mul(q0, math_utils.quat_from_euler_xyz(ang[:, 0], ang[:, 1], ang[:, 2]))
    quat = convert_camera_frame_orientation_convention(quat, origin="ros", target="opengl")
    cam._view.set_local_poses(  # noqa: SLF001 (Camera exposes no local-pose setter)
        wp.from_torch(pos.contiguous(), dtype=wp.vec3f),
        wp.from_torch(quat.contiguous(), dtype=wp.vec4f),
        cam._resolve_env_ids_wp(env_ids),  # noqa: SLF001
    )


def randomize_mdl_color(
    env,
    env_ids: torch.Tensor | None,
    shader_suffix: str,
    input_name: str,
    color_low: Sequence[float],
    color_high: Sequence[float],
):
    """Set a colour input of ONE MDL shader (``/World/envs/env_<i>/<shader_suffix>``), e.g. the 3D-printed parts of the
    SO-101 USD (OmniPBR ``diffuse_color_constant``), uniformly in [low, high] per channel."""
    import omni.usd
    from pxr import Gf, Sdf, UsdShade

    stage = omni.usd.get_context().get_stage()
    ids = range(env.num_envs) if env_ids is None else env_ids.tolist()
    low, high = torch.tensor(color_low), torch.tensor(color_high)
    for i in ids:
        prim = stage.GetPrimAtPath(f"/World/envs/env_{i}/{shader_suffix}")
        if not prim.IsValid():
            raise RuntimeError(f"prim not found: /World/envs/env_{i}/{shader_suffix}")
        rgb = (low + torch.rand(3) * (high - low)).tolist()
        UsdShade.Shader(prim).CreateInput(input_name, Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))


def randomize_texture_scale(
    env,
    env_ids: torch.Tensor | None,
    prim_suffix: str,
    brightness_range: tuple[float, float],
    tint_range: float,
):
    """Multiply the colour of every UsdUVTexture under ``/World/envs/env_<i>/<prim_suffix>`` (``inputs:scale``):
    one brightness factor for the 3 channels times a per-channel tint in [1 - tint_range, 1 + tint_range]."""
    import omni.usd
    from pxr import Gf, Usd, UsdShade

    stage = omni.usd.get_context().get_stage()
    ids = range(env.num_envs) if env_ids is None else env_ids.tolist()
    for i in ids:
        root = stage.GetPrimAtPath(f"/World/envs/env_{i}/{prim_suffix}")
        if not root.IsValid():
            raise RuntimeError(f"prim not found: /World/envs/env_{i}/{prim_suffix}")
        b = brightness_range[0] + torch.rand(1).item() * (brightness_range[1] - brightness_range[0])
        tint = (1 + (torch.rand(3) * 2 - 1) * tint_range).tolist()
        done = 0
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdShade.Shader) and UsdShade.Shader(prim).GetShaderId() == "UsdUVTexture":
                UsdShade.Shader(prim).GetInput("scale").Set(Gf.Vec4f(b * tint[0], b * tint[1], b * tint[2], 1.0))
                done += 1
        if not done:
            raise RuntimeError(f"no UsdUVTexture under /World/envs/env_{i}/{prim_suffix}")


def profile_stack_success(env, short_cfg: SceneEntityCfg, long_cfg: SceneEntityCfg, ee_frame_cfg: SceneEntityCfg,
                          layout: dict, max_speed: float = 0.05, max_ang_speed: float = 0.5,
                          hold_steps: int = 0) -> torch.Tensor:
    """P3 success: the checks of cta_learn.retarget.profile_stack.stack_success (short part lying in its zone,
    long part upright on top of it, gripper away) + both parts at rest (linear and angular speed).

    ``hold_steps`` > 0: the condition must hold for that many CONSECUTIVE calls (call once per env step). Without
    it, success fired as soon as the gripper was 5 cm away, while the long part could still be tipping over
    (seen by the user in the P3.2 demo, 2026-11-13). In that mode "at rest" is measured on the POSE change between
    two calls (same speed limits), not on PhysX's velocities: for a part standing on another one those jitter
    (0.03 m/s, 0.6 rad/s read) while the pose moves < 0.1 mm and < 0.1 deg in 3 s (P3.3 diagnosis)."""
    from cta_learn.retarget.profile_stack import stack_success

    s, l_ = env.scene[short_cfg.name], env.scene[long_cfg.name]
    o = env.scene.env_origins
    frame = env.scene[ee_frame_cfg.name]
    tcp = frame.data.target_pos_w.torch[:, frame_index(frame, "tcp"), :] - o
    goal = torch.tensor(layout["goals"]["profile_short"]["xyyaw"][:2], device=o.device).expand(o.shape[0], 2)
    ok, _ = stack_success(s.data.root_pos_w.torch - o, s.data.root_quat_w.torch, l_.data.root_pos_w.torch - o,
                          l_.data.root_quat_w.torch, tcp, goal, layout, torch)
    if hold_steps <= 0:
        still = (torch.linalg.vector_norm(s.data.root_lin_vel_w.torch, dim=1) < max_speed) & (
            torch.linalg.vector_norm(l_.data.root_lin_vel_w.torch, dim=1) < max_speed) & (
            torch.linalg.vector_norm(l_.data.root_ang_vel_w.torch, dim=1) < max_ang_speed)
        return ok & still
    pose = torch.cat([s.data.root_pos_w.torch, s.data.root_quat_w.torch, l_.data.root_pos_w.torch, l_.data.root_quat_w.torch], 1)
    key = f"_stack_success_state_{long_cfg.name}"
    prev, count = getattr(env, key, (None, None))
    if prev is None or prev.shape != pose.shape:
        prev, count = pose.clone(), torch.zeros(ok.shape, dtype=torch.int32, device=ok.device)
    dt = env.step_dt
    moved = torch.zeros_like(ok)
    for p0 in (0, 7):  # short, long: position step and rotation angle between the two quaternions
        dpos = torch.linalg.vector_norm(pose[:, p0:p0 + 3] - prev[:, p0:p0 + 3], dim=1)
        dot = (pose[:, p0 + 3:p0 + 7] * prev[:, p0 + 3:p0 + 7]).sum(1).abs().clamp(max=1.0)
        moved |= (dpos > max_speed * dt) | (2 * torch.acos(dot) > max_ang_speed * dt)
    ok = ok & ~moved
    count = torch.where(ok, count + 1, torch.zeros_like(count))
    setattr(env, key, (pose.clone(), count))
    return count >= hold_steps


def goal_xy_const(env, xy: tuple[float, float]) -> torch.Tensor:
    """Constant goal position (env frame) as an observation for the scripted expert."""
    return torch.tensor(xy, device=env.device).expand(env.num_envs, 2).clone()


def break_rigid_contacts(env, env_ids: torch.Tensor, asset_names: Sequence[str], park_height: float = 0.5) -> None:
    """Reset event, to apply AFTER the events that place ``asset_names``: forget their old contacts.

    PhysX keeps a contact pair's friction anchors (body-local points of the previous frame) while the pair stays in
    contact. A part teleported to a new pose ON the table was therefore pulled back toward its PREVIOUS pose at the
    first physics step (up to 2 cm / 20 deg; the 75 mm upright part sometimes fell). P3.1 diagnosis, 2026-11-13.
    Fix: park the parts in the air for one physics step (the pairs are lost), then write the sampled pose back.

    Side effects: one extra physics step (1/60 s) for the envs NOT being reset (none when all envs reset together,
    e.g. num_envs=1); the reset envs' articulations are written back to their default state after that step.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    assets = [env.scene[name] for name in asset_names]
    targets = [a.data.root_pose_w.torch[env_ids].clone() for a in assets]
    zero = torch.zeros(len(env_ids), 6, device=env.device)
    for k, (a, pose) in enumerate(zip(assets, targets)):
        parked = pose.clone()
        parked[:, 2] += park_height + 0.2 * k  # distinct heights: the parked parts do not touch each other
        a.write_root_pose_to_sim_index(root_pose=parked, env_ids=env_ids)
        a.write_root_velocity_to_sim_index(root_velocity=zero, env_ids=env_ids)
    env.sim.step(render=False)
    env.scene.update(dt=env.physics_dt)
    for a, pose in zip(assets, targets):
        a.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        a.write_root_velocity_to_sim_index(root_velocity=zero, env_ids=env_ids)
    reset_articulations_to_default(env, env_ids)


def reset_articulations_to_default(env, env_ids: torch.Tensor) -> None:
    """Like isaaclab's reset_scene_to_default, but for articulations (the robot) only.

    Rigid parts are placed by their own reset_root_state_uniform event. Teleporting them first to their default
    pose and then to the random one in the same reset left PhysX an impulse: a 75 mm upright part spawned tilted
    up to ~20 deg and slid 1-2 cm toward its default pose, sometimes falling (P3.1 diagnosis, 2026-11-13).
    """
    for art in env.scene.articulations.values():
        pose = art.data.default_root_pose.torch[env_ids].clone()
        vel = art.data.default_root_vel.torch[env_ids].clone()
        pose[:, :3] += env.scene.env_origins[env_ids]
        art.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        art.write_root_velocity_to_sim_index(root_velocity=vel, env_ids=env_ids)
        art.write_joint_position_to_sim_index(position=art.data.default_joint_pos.torch[env_ids].clone(), env_ids=env_ids)
        art.write_joint_velocity_to_sim_index(velocity=art.data.default_joint_vel.torch[env_ids].clone(), env_ids=env_ids)


def taskspec_success(env, ee_frame_cfg: SceneEntityCfg, layout: dict, max_speed: float = 0.05,
                     max_ang_speed: float = 0.5, hold_steps: int = 30) -> torch.Tensor:
    """Generic success (any layout of cta_learn.retarget.layout): its goals met (task_success), every part at rest
    and the gripper away, for ``hold_steps`` CONSECUTIVE calls (call once per env step). "At rest" is measured on
    the pose change between two calls, as in profile_stack_success (PhysX velocities jitter for stacked parts)."""
    from cta_learn.retarget.layout import task_success

    names = list(layout["order"])
    o = env.scene.env_origins
    frame = env.scene[ee_frame_cfg.name]
    tcp = frame.data.target_pos_w.torch[:, frame_index(frame, "tcp"), :] - o
    pos = {n: env.scene[n].data.root_pos_w.torch - o for n in names}
    quat = {n: env.scene[n].data.root_quat_w.torch for n in names}
    zone = {n: torch.tensor(g["xyyaw"][:2], device=o.device).expand(o.shape[0], 2)
            for n, g in layout["goals"].items() if g["relation"] == "on_table"}
    ok, _ = task_success(pos, quat, tcp, zone, layout, torch)
    pose = torch.cat([torch.cat([env.scene[n].data.root_pos_w.torch, quat[n]], 1) for n in names], 1)
    key = "_taskspec_success_state"
    prev, count = getattr(env, key, (None, None))
    if prev is None or prev.shape != pose.shape:
        prev, count = pose.clone(), torch.zeros(ok.shape, dtype=torch.int32, device=ok.device)
    dt = env.step_dt
    moved = torch.zeros_like(ok)
    for k in range(len(names)):
        p0 = 7 * k
        dpos = torch.linalg.vector_norm(pose[:, p0:p0 + 3] - prev[:, p0:p0 + 3], dim=1)
        dot = (pose[:, p0 + 3:p0 + 7] * prev[:, p0 + 3:p0 + 7]).sum(1).abs().clamp(max=1.0)
        moved |= (dpos > max_speed * dt) | (2 * torch.acos(dot) > max_ang_speed * dt)
    ok = ok & ~moved
    count = torch.where(ok, count + 1, torch.zeros_like(count))
    setattr(env, key, (pose.clone(), count))
    return count >= max(hold_steps, 1)
