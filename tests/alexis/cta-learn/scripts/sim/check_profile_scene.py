"""Headless validation of CTA-SO101-ProfileStack-v0 (P3.1). Exit code 0 = all checks passed.

PowerShell, from the repo root:
    $env:PYTHONPATH = "$PWD\\src"; & ..\\isaacsim-env\\Scripts\\python.exe scripts\\sim\\check_profile_scene.py --headless --enable_cameras
Writes camera snapshots and a JSON report to outputs/p3_1/.
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
from cta_learn.retarget.profile_stack import rotate_xyzw  # noqa: E402
from cta_learn.sim.checks import verify_robot_matches_embodiment  # noqa: E402
from cta_learn.sim import mdp  # noqa: E402
from cta_learn.sim.robots import SO101, SO101_START_POSE_DEG  # noqa: E402
from cta_learn.sim.tasks.profile_stack import GOAL_SHORT_XY, LONG, SECTION, SHORT, TASK_ID, ProfileStackEnvCfg  # noqa: E402

OUT = REPO_ROOT / "outputs" / "p3_1"
OUT.mkdir(parents=True, exist_ok=True)
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


def axis_z(quat: torch.Tensor, local: tuple) -> np.ndarray:
    v = torch.tensor(local, dtype=quat.dtype, device=quat.device).expand(quat.shape[0], 3)
    return rotate_xyzw(quat, v, torch)[:, 2].abs().cpu().numpy()


def teleport(asset, xyz, quat_xyzw, n, dev, origins):
    pose = torch.zeros(n, 7, device=dev)
    pose[:, :3] = origins + torch.tensor(xyz, device=dev)
    pose[:, 3:7] = torch.tensor(quat_xyzw, device=dev)
    ids = torch.arange(n, device=dev)
    asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
    asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev), env_ids=ids)


cfg = ProfileStackEnvCfg()
cfg.scene.num_envs = args.num_envs
# the success TERM would auto-reset an env as soon as it fires (then the stack could not be watched):
# it is removed here and the same function is called directly
SUCCESS_PARAMS = dict(cfg.terminations.success.params)
cfg.terminations.success = None
t0 = time.time()
env = gym.make(TASK_ID, cfg=cfg).unwrapped
check("build", True, f"{args.num_envs} envs in {time.time() - t0:.1f} s")
robot = env.scene["robot"]
problems = verify_robot_matches_embodiment(robot, SO101)
check("joints_match_embodiment", not problems, "; ".join(problems) or "names and limits match so101.yaml")

n, dev, org = env.num_envs, env.device, env.scene.env_origins
short, long_ = env.scene["profile_short"], env.scene["profile_long"]
q_start = torch.tensor([math.radians(SO101_START_POSE_DEG[j.name]) for j in SO101.joints], device=dev).repeat(n, 1)

# 1-3) spawn ranges, postures, separation and settling over several resets
pos_s, pos_l, lying, upright, sep, settled_z, speed, succ_reset = [], [], [], [], [], [], [], []
drift_mm, drift_deg = [], []
for r in range(args.resets):
    env.reset()
    # the parts must not move in the first frames after a reset (PhysX contact memory bug, see mdp.break_rigid_contacts)
    p0_ = [(a.data.root_pos_w.torch.clone(), a.data.root_quat_w.torch.clone()) for a in (short, long_)]
    for _ in range(6):
        env.step(q_start)
        for a, (pp, qq) in zip((short, long_), p0_):
            drift_mm.append(torch.linalg.vector_norm(a.data.root_pos_w.torch[:, :2] - pp[:, :2], dim=1).max().item() * 1000)
            dot = (a.data.root_quat_w.torch * qq).sum(-1).abs().clamp(max=1.0)
            drift_deg.append(torch.rad2deg(2 * torch.acos(dot)).max().item())
    obs, *_ = hold(env, q_start, 39)
    ps = (short.data.root_pos_w.torch - org).cpu().numpy()
    pl = (long_.data.root_pos_w.torch - org).cpu().numpy()
    pos_s.append(ps); pos_l.append(pl)
    lying.append(axis_z(short.data.root_quat_w.torch, (1.0, 0, 0)))
    upright.append(axis_z(long_.data.root_quat_w.torch, (0, 0, 1.0)))
    sep.append(np.linalg.norm(ps[:, :2] - pl[:, :2], axis=1))
    settled_z.append(np.c_[ps[:, 2], pl[:, 2]])
    speed.append(max(torch.linalg.vector_norm(short.data.root_lin_vel_w.torch, dim=1).max().item(),
                     torch.linalg.vector_norm(long_.data.root_lin_vel_w.torch, dim=1).max().item()))
    succ_reset.append(mdp.profile_stack_success(env, **SUCCESS_PARAMS).cpu().numpy())
    if r == 0:
        tcp_rest = obs["expert"]["tcp_pos"][0].cpu().numpy()
        print(f"TCP at the start pose (env frame): {tcp_rest.round(3).tolist()}")
        for cam in ("front", "wrist"):
            save_png(obs["policy"][cam][0].cpu().numpy(), OUT / f"{cam}_start.png")
check("still_after_reset", max(drift_mm) < 1.0 and max(drift_deg) < 1.0,
      f"max over the 6 frames after each reset: {max(drift_mm):.2f} mm (xy), {max(drift_deg):.2f} deg")
pos_s, pos_l = np.concatenate(pos_s), np.concatenate(pos_l)
in_range = all(np.all(np.abs(p[:, k] - o["start_xyyaw"][k]) <= o["range"][k] + 0.003) for p, o in ((pos_s, SHORT), (pos_l, LONG)) for k in (0, 1))
check("spawn_in_ranges", in_range and np.ptp(pos_s[:, 1]) > 0.01,
      f"short x {pos_s[:, 0].min():.3f}..{pos_s[:, 0].max():.3f} y {pos_s[:, 1].min():.3f}..{pos_s[:, 1].max():.3f}; "
      f"long x {pos_l[:, 0].min():.3f}..{pos_l[:, 0].max():.3f} y {pos_l[:, 1].min():.3f}..{pos_l[:, 1].max():.3f}")
lying, upright = np.concatenate(lying), np.concatenate(upright)
check("start_postures", lying.max() < 0.05 and upright.min() > 0.99,
      f"short |axis_z| max {lying.max():.3f} (lying), long |axis_z| min {upright.min():.3f} (upright)")
sep, z = np.concatenate(sep), np.concatenate(settled_z)
check("parts_apart_and_resting", sep.min() > 0.045 and np.all(np.abs(z[:, 0] - SECTION / 2) < 0.003)
      and np.all(np.abs(z[:, 1] - LONG["length_m"] / 2) < 0.003) and max(speed) < 0.02,
      f"min centre distance {sep.min() * 1000:.0f} mm; z short {z[:, 0].mean() * 1000:.1f} mm, long {z[:, 1].mean() * 1000:.1f} mm; "
      f"max speed after 1.5 s {max(speed):.3f} m/s")
check("no_success_at_reset", not np.concatenate(succ_reset).any(), f"{args.resets * n} resets")

# 4) images
for cam in ("front", "wrist"):
    img = obs["policy"][cam]
    check(f"image_{cam}", img.dtype == torch.uint8 and img.float().std().item() > 5.0, f"shape {tuple(img.shape)}, std {img.float().std().item():.1f}")

# 5) stacked goal state: success fires and the long part stays standing on the short one (physics stability)
env.reset()
hold(env, q_start, 5)
gx, gy = GOAL_SHORT_XY
teleport(short, (gx, gy, SECTION / 2 + 0.0005), (0.0, 0.0, 0.0, 1.0), n, dev, org)
teleport(long_, (gx, gy, SECTION + LONG["length_m"] / 2 + 0.001), (0.0, 0.0, 0.0, 1.0), n, dev, org)
fired = np.zeros(n, bool)
for _ in range(60):
    obs = env.step(q_start)
    fired |= mdp.profile_stack_success(env, **SUCCESS_PARAMS).cpu().numpy()
save_png(obs[0]["policy"]["front"][0].cpu().numpy(), OUT / "front_goal_state.png")
save_png(obs[0]["policy"]["wrist"][0].cpu().numpy(), OUT / "wrist_goal_state.png")
check("success_on_goal_state", fired.all(), f"fired per env: {fired.tolist()}")
hold(env, q_start, 60)
up2 = axis_z(long_.data.root_quat_w.torch, (0, 0, 1.0))
zl = (long_.data.root_pos_w.torch[:, 2] - org[:, 2]).cpu().numpy()
still_ok = mdp.profile_stack_success(env, **SUCCESS_PARAMS).cpu().numpy()
check("stack_is_stable", still_ok.all() and up2.min() > 0.99 and np.all(np.abs(zl - (SECTION + LONG["length_m"] / 2)) < 0.003),
      f"after 4 s: long |axis_z| {up2.round(4).tolist()}, centre z {np.round(zl * 1000, 1).tolist()} mm")

# 6) a wrong final state does not count: long part standing next to the short part
env.reset()
hold(env, q_start, 5)
teleport(short, (gx, gy, SECTION / 2 + 0.0005), (0.0, 0.0, 0.0, 1.0), n, dev, org)
teleport(long_, (gx + 0.06, gy, LONG["length_m"] / 2 + 0.001), (0.0, 0.0, 0.0, 1.0), n, dev, org)
wrong = np.zeros(n, bool)
for _ in range(45):
    env.step(q_start)
    wrong |= mdp.profile_stack_success(env, **SUCCESS_PARAMS).cpu().numpy()
check("no_success_when_not_stacked", not wrong.any(), f"fired per env: {wrong.tolist()}")

# 7) out-of-range trials use wider ranges
cfg2_ranges = ProfileStackEnvCfg()
cfg2_ranges.use_ood_ranges()
r_ood = cfg2_ranges.events.profile_short.params["pose_range"]["x"][1]
check("ood_ranges", r_ood > SHORT["range"][0], f"short x half-range {SHORT['range'][0]} -> {r_ood} m (OOD)")

steps, t0 = 150, time.time()
hold(env, q_start, steps)
sps = steps / (time.time() - t0)
report = {"checks": results, "steps_per_s": sps, "num_envs": n, "max_alloc_MiB": round(torch.cuda.max_memory_allocated() / 2**20)}
(OUT / "check_profile_scene_report.json").write_text(json.dumps(report, indent=2))
ok = all(r["ok"] for r in results.values())
print(f"{sum(r['ok'] for r in results.values())}/{len(results)} checks passed;", f"{sps:.1f} steps/s with {n} envs ->", OUT)
env.close()
app.close()
raise SystemExit(0 if ok else 1)
