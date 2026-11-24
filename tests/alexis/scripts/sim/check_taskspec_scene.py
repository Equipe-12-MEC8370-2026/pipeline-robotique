r"""Headless validation of CTA-SO101-TaskSpec-v0 for one layout (generic version of check_profile_scene.py).

PowerShell, from the repo root:
    $env:PYTHONPATH = "$PWD\src"; $env:CTA_LAYOUT = "configs/tasks/so101_cta_angle_block.yaml"
    & ..\isaacsim-env\Scripts\python.exe scripts\sim\check_taskspec_scene.py --headless --enable_cameras
Exit code 0 = all checks passed. Writes camera snapshots and a JSON report to outputs/taskspec_scene/<task_id>/.
Checks: robot joints; parts still after reset (PhysX contact memory); spawned in their ranges, at their start posture,
apart and resting; no success at reset; camera images; the goal state (every part teleported to its goal) fires
success and stays stable 4 s; a wrong state (first part 6 cm off its goal) does not; out-of-range trials widen the
ranges.
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--resets", type=int, default=10)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import cta_learn.sim.tasks  # noqa: E402,F401
from cta_learn import REPO_ROOT  # noqa: E402
from cta_learn.core.geometry import quat_from_euler_deg  # noqa: E402
from cta_learn.retarget.layout import posture_ok, rotate_xyzw, vertical_size  # noqa: E402
from cta_learn.sim import mdp  # noqa: E402
from cta_learn.sim.checks import verify_robot_matches_embodiment  # noqa: E402
from cta_learn.sim.robots import SO101, SO101_START_POSE_DEG  # noqa: E402
from cta_learn.sim.tasks.taskspec_task import TASK_ID, TaskSpecEnvCfg, layout_path  # noqa: E402

results: dict[str, dict] = {}


def check(name: str, ok: bool, detail: str = "") -> None:
    results[name] = {"ok": bool(ok), "detail": detail}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def save_png(img: np.ndarray, path) -> None:
    from PIL import Image
    Image.fromarray(img).save(path)


def hold(env, q, steps):
    out = None
    for _ in range(steps):
        out = env.step(q)
    return out


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz)


def goal_quat(size_local, vertical, yaw):
    """XYZW quaternion putting the local axis of size ``vertical`` up (z stays up if it already is), then yaw."""
    k = 2 if abs(size_local[2] - vertical) < 1e-6 else [i for i in range(3) if abs(size_local[i] - vertical) < 1e-6][0]
    tilt = {2: (0.0, 0.0, 0.0, 1.0), 0: quat_from_euler_deg(0.0, 90.0, 0.0), 1: quat_from_euler_deg(90.0, 0.0, 0.0)}[k]
    return qmul(quat_from_euler_deg(0.0, 0.0, math.degrees(yaw)), tilt)


def teleport(asset, xyz, quat_xyzw, n, dev, origins):
    pose = torch.zeros(n, 7, device=dev)
    pose[:, :3] = origins + torch.tensor(xyz, device=dev, dtype=torch.float32)
    pose[:, 3:7] = torch.tensor(quat_xyzw, device=dev, dtype=torch.float32)
    ids = torch.arange(n, device=dev)
    asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
    asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev), env_ids=ids)


def goal_poses(lay):
    """{part: (xyz, quat)} of the layout's goal state, parts in task order."""
    out = {}
    for name in lay["order"]:
        o, g = lay["objects"][name], lay["goals"][name]
        v = g["vertical_m"]
        if g["relation"] == "on_table":
            x, y, yaw = g["xyyaw"]
            out[name] = ((x, y, v / 2 + 0.0005), goal_quat(o["size_m"], v, yaw))
        else:
            (rx, ry, rz), rq = out[g["ref"]]
            ref_v = lay["goals"][g["ref"]]["vertical_m"]
            ax = rotate_xyzw(np.array([rq]), np.array([[1.0, 0, 0]]), np) [0]
            c, s = ax[:2] / (np.linalg.norm(ax[:2]) + 1e-9)
            ox, oy = g["offset_xy_m"]
            yaw_ref = math.atan2(s, c)
            out[name] = ((rx + c * ox - s * oy, ry + s * ox + c * oy, rz + ref_v / 2 + v / 2 + 0.001),
                         goal_quat(o["size_m"], v, yaw_ref + math.radians(g["yaw_rel_deg"] or 0.0)))
    return out


cfg = TaskSpecEnvCfg()
cfg.scene.num_envs = args.num_envs
lay = cfg.layout
names = list(lay["order"])
OUT = REPO_ROOT / "outputs" / "taskspec_scene" / lay["task_id"]
OUT.mkdir(parents=True, exist_ok=True)
SUCCESS_PARAMS = dict(cfg.terminations.success.params)
cfg.terminations.success = None  # watched directly (the term would auto-reset the env)
t0 = time.time()
env = gym.make(TASK_ID, cfg=cfg).unwrapped
check("build", True, f"{layout_path().name}: {len(names)} part(s) {names}, {args.num_envs} envs in {time.time() - t0:.1f} s")
robot = env.scene["robot"]
problems = verify_robot_matches_embodiment(robot, SO101)
check("joints_match_embodiment", not problems, "; ".join(problems) or "names and limits match so101.yaml")
n, dev, org = env.num_envs, env.device, env.scene.env_origins
parts = {k: env.scene[k] for k in names}
q_start = torch.tensor([math.radians(SO101_START_POSE_DEG[j.name]) for j in SO101.joints], device=dev).repeat(n, 1)


def success_now():
    return mdp.taskspec_success(env, **SUCCESS_PARAMS).cpu().numpy()


pos, post_ok, zs, speed, succ_reset, drift_mm, drift_deg, sep = {k: [] for k in names}, {k: [] for k in names}, {k: [] for k in names}, [], [], [], [], []
obs = None
for r in range(args.resets):
    env.reset()
    p0_ = {k: (a.data.root_pos_w.torch.clone(), a.data.root_quat_w.torch.clone()) for k, a in parts.items()}
    for _ in range(6):
        env.step(q_start)
        for k, a in parts.items():
            pp, qq = p0_[k]
            drift_mm.append(torch.linalg.vector_norm(a.data.root_pos_w.torch[:, :2] - pp[:, :2], dim=1).max().item() * 1000)
            dot = (a.data.root_quat_w.torch * qq).sum(-1).abs().clamp(max=1.0)
            drift_deg.append(torch.rad2deg(2 * torch.acos(dot)).max().item())
    obs, *_ = hold(env, q_start, 39)
    for k, a in parts.items():
        o = lay["objects"][k]
        pos[k].append((a.data.root_pos_w.torch - org).cpu().numpy())
        post_ok[k].append(posture_ok(a.data.root_quat_w.torch, o["size_m"], o["size_m"][2], 5.0, torch).cpu().numpy())
        zs[k].append(pos[k][-1][:, 2])
        speed.append(torch.linalg.vector_norm(a.data.root_lin_vel_w.torch, dim=1).max().item())
    for i, k in enumerate(names):
        for k2 in names[i + 1:]:
            sep.append(np.linalg.norm(pos[k][-1][:, :2] - pos[k2][-1][:, :2], axis=1).min())
    succ_reset.append(success_now())
    if r == 0:
        for cam in ("front", "wrist"):
            save_png(obs["policy"][cam][0].cpu().numpy(), OUT / f"{cam}_start.png")
check("still_after_reset", max(drift_mm) < 1.0 and max(drift_deg) < 1.0,
      f"max over the 6 frames after each reset: {max(drift_mm):.2f} mm (xy), {max(drift_deg):.2f} deg")
in_range, txt = True, []
for k in names:
    p, o = np.concatenate(pos[k]), lay["objects"][k]
    in_range &= all(np.all(np.abs(p[:, j] - o["start_xyyaw"][j]) <= o["range"][j] + 0.003) for j in (0, 1))
    txt.append(f"{k} x {p[:, 0].min():.3f}..{p[:, 0].max():.3f} y {p[:, 1].min():.3f}..{p[:, 1].max():.3f}")
check("spawn_in_ranges", in_range, "; ".join(txt))
check("start_postures", all(np.concatenate(post_ok[k]).all() for k in names),
      ", ".join(f"{k} {lay['objects'][k]['start_posture']}" for k in names))
z_ok = all(np.all(np.abs(np.concatenate(zs[k]) - lay["objects"][k]["size_m"][2] / 2) < 0.003) for k in names)
check("parts_apart_and_resting", z_ok and (not sep or min(sep) > 0.045) and max(speed) < 0.02,
      (f"min centre distance {min(sep) * 1000:.0f} mm; " if sep else "") + f"max speed after 1.5 s {max(speed):.3f} m/s")
check("no_success_at_reset", not np.concatenate(succ_reset).any(), f"{args.resets * n} resets")
for cam in ("front", "wrist"):
    img = obs["policy"][cam]
    check(f"image_{cam}", img.dtype == torch.uint8 and img.float().std().item() > 5.0, f"shape {tuple(img.shape)}, std {img.float().std().item():.1f}")

# goal state: success fires, then stays (stable)
gp = goal_poses(lay)
env.reset()
hold(env, q_start, 5)
for k in names:
    teleport(parts[k], gp[k][0], gp[k][1], n, dev, org)
fired = np.zeros(n, bool)
for _ in range(60):
    obs = env.step(q_start)
    fired |= success_now()
save_png(obs[0]["policy"]["front"][0].cpu().numpy(), OUT / "front_goal_state.png")
check("success_on_goal_state", fired.all(), f"fired per env: {fired.tolist()}")
hold(env, q_start, 60)
check("goal_state_is_stable", success_now().all(), "still a success after 4 s")

# wrong state: the first part 6 cm off its goal
env.reset()
hold(env, q_start, 5)
for k in names:
    xyz, q = gp[k]
    teleport(parts[k], (xyz[0] + 0.06, xyz[1], xyz[2]), q, n, dev, org)
    break
for k in names[1:]:
    teleport(parts[k], gp[k][0], gp[k][1], n, dev, org)
wrong = np.zeros(n, bool)
for _ in range(45):
    env.step(q_start)
    wrong |= success_now()
check("no_success_when_misplaced", not wrong.any(), f"fired per env: {wrong.tolist()}")

cfg2 = TaskSpecEnvCfg()
cfg2.use_ood_ranges()
k0 = names[0]
r_ood = getattr(cfg2.events, f"place_{k0}").params["pose_range"]["x"][1]
check("ood_ranges", r_ood > lay["objects"][k0]["range"][0], f"{k0} x half-range {lay['objects'][k0]['range'][0]} -> {r_ood} m")

report = {"layout": str(layout_path()), "checks": results, "num_envs": n}
(OUT / "check_taskspec_scene_report.json").write_text(json.dumps(report, indent=2))
ok = all(r["ok"] for r in results.values())
print(f"{sum(r['ok'] for r in results.values())}/{len(results)} checks passed ->", OUT)
env.close()
app.close()
raise SystemExit(0 if ok else 1)
