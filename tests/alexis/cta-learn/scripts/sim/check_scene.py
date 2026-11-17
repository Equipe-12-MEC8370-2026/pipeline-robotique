"""Headless validation of CTA-SO101-PickPlace-v0 (exit code 0 = all checks passed).

PowerShell, from the repo root:
    $env:PYTHONPATH = "$PWD\\src"; & ..\\isaacsim-env\\Scripts\\python.exe scripts\\sim\\check_scene.py --headless --enable_cameras
Writes camera snapshots and a JSON report to outputs/p0_2/.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=2)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

# ---- imports that need the running app
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import cta_learn.sim.tasks  # noqa: E402,F401  (registers the env)
from cta_learn import REPO_ROOT  # noqa: E402
from cta_learn.sim.checks import verify_robot_matches_embodiment  # noqa: E402
from cta_learn.sim.robots import SO101, SO101_START_POSE_DEG  # noqa: E402
from cta_learn.sim.tasks.pick_place import TASK_ID, PickPlaceEnvCfg  # noqa: E402

OUT = REPO_ROOT / "outputs" / "p0_2"
OUT.mkdir(parents=True, exist_ok=True)
results: dict[str, dict] = {}


def check(name: str, ok: bool, detail: str = "") -> None:
    results[name] = {"ok": bool(ok), "detail": detail}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def save_png(img: np.ndarray, path) -> None:
    try:
        from PIL import Image

        Image.fromarray(img).save(path)
    except ImportError:
        np.save(str(path).replace(".png", ".npy"), img)


def hold(env, q: torch.Tensor, steps: int):
    out = None
    for _ in range(steps):
        out = env.step(q)
    return out


cfg = PickPlaceEnvCfg()
cfg.scene.num_envs = args.num_envs
t0 = time.time()
env = gym.make(TASK_ID, cfg=cfg).unwrapped
check("build", True, f"{args.num_envs} envs in {time.time() - t0:.1f} s")

robot = env.scene["robot"]
print("sim joint names :", list(robot.data.joint_names))
print("sim body names  :", list(robot.data.body_names))
problems = verify_robot_matches_embodiment(robot, SO101)
check("joints_match_embodiment", not problems, "; ".join(problems) or "names and limits match so101.yaml")

# 0) the robot base sits on the table top (USD bounding box of the static base link)
import omni.usd  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

_bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
_base = omni.usd.get_context().get_stage().GetPrimAtPath("/World/envs/env_0/Robot/base")
base_min_z = _bbox.ComputeWorldBound(_base).ComputeAlignedRange().GetMin()[2] - float(env.scene.env_origins[0, 2])
check("robot_on_table", abs(base_min_z) < 0.003, f"bottom of base at z={base_min_z * 1000:.1f} mm (table top = 0)")

obs, _ = env.reset()
n = env.num_envs
q_start = torch.tensor([math.radians(SO101_START_POSE_DEG[j.name]) for j in SO101.joints], device=env.device)
q_start = q_start.repeat(n, 1)

# 1) settle: hold the start pose for 2 s, the cube must rest on the table
obs, *_ = hold(env, q_start, 60)
cube = env.scene["cube"]
cube_z = (cube.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]).cpu().numpy()
cube_v = torch.linalg.vector_norm(cube.data.root_lin_vel_w.torch, dim=1).cpu().numpy()
finite = all(torch.isfinite(t).all().item() for t in (robot.data.joint_pos.torch, cube.data.root_pos_w.torch))
check("cube_rests_on_table", finite and np.all(np.abs(cube_z - 0.015) < 0.004) and np.all(cube_v < 0.02),
      f"z={cube_z.round(4).tolist()} m, |v|={cube_v.round(4).tolist()} m/s")

# 2) joint tracking: the arm reaches the commanded start pose
err = torch.abs(obs["policy"]["joint_pos"] - q_start).max().item()
check("joint_tracking", err < math.radians(3.0), f"max error {math.degrees(err):.2f} deg after 2 s")

# 3) images
for cam in ("front", "wrist"):
    img = obs["policy"][cam]
    ok = tuple(img.shape) == (n, 480, 640, 3) and img.dtype == torch.uint8 and img.float().std().item() > 5.0
    check(f"image_{cam}", ok, f"shape={tuple(img.shape)} dtype={img.dtype} std={img.float().std().item():.1f}")
    save_png(img[0].cpu().numpy(), OUT / f"{cam}_start.png")

# 4) where does the arm reach? (decides ROBOT_YAW_DEG): extend the arm forward
q_reach = q_start.clone()
q_reach[:, 1] = math.radians(0.0)   # shoulder_lift
q_reach[:, 2] = math.radians(0.0)   # elbow_flex
q_reach[:, 3] = math.radians(0.0)   # wrist_flex
obs, *_ = hold(env, q_reach, 60)
grasp = obs["expert"]["tcp_pos"][0].cpu().numpy()
check("arm_reaches_plus_x", grasp[0] > 0.10 and abs(grasp[1]) < 0.08,
      f"TCP with lift=elbow=wrist=0: {grasp.round(3).tolist()} (env frame)")
save_png(obs["policy"]["front"][0].cpu().numpy(), OUT / "front_reach.png")
save_png(obs["policy"]["wrist"][0].cpu().numpy(), OUT / "wrist_reach.png")
hold(env, q_start, 45)

# 5) success: false at reset, true once the cube is put in the target and released
obs, _ = env.reset()
obs, *_ = hold(env, q_start, 5)
succ0 = env.termination_manager.get_term("success").cpu().numpy()
tgt = env.scene["target"].data.root_pos_w.torch.clone()
pose = cube.data.root_pose_w.torch.clone()
pose[:, 0:2] = tgt[:, 0:2]
pose[:, 2] = env.scene.env_origins[:, 2] + 0.016
pose[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device)
cube.write_root_pose_to_sim_index(root_pose=pose, env_ids=torch.arange(n, device=env.device))
cube.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=env.device),
                                      env_ids=torch.arange(n, device=env.device))
fired = np.zeros(n, dtype=bool)
for _ in range(30):
    env.step(q_start)
    fired |= env.termination_manager.get_term("success").cpu().numpy()
check("success_term", (not succ0.any()) and fired.all(),
      f"at reset={succ0.tolist()}, after placing={fired.tolist()}")

# 6) randomisation: cube pose changes between resets
positions = []
for _ in range(3):
    env.reset()
    positions.append((cube.data.root_pos_w.torch[0] - env.scene.env_origins[0]).cpu().numpy())
spread = float(np.ptp(np.stack(positions)[:, :2], axis=0).max())
check("randomisation", spread > 0.01, f"cube xy over 3 resets: {np.stack(positions)[:, :2].round(3).tolist()}")
obs, *_ = hold(env, q_start, 30)
save_png(obs["policy"]["front"][0].cpu().numpy(), OUT / "front_randomised.png")

# 7) speed and memory
steps = 150
t0 = time.time()
hold(env, q_start, steps)
dt = time.time() - t0
sps = steps / dt
if n == 1:
    check("realtime", sps >= 30.0, f"{sps:.1f} policy steps/s with 1 env (target >= 30)")
else:
    check("realtime", True, f"{sps:.1f} policy steps/s with {n} envs (target applies to 1 env; informative)")
mem = {"torch_max_alloc_MiB": round(torch.cuda.max_memory_allocated() / 2**20)}
try:
    import psutil

    mem["process_rss_MiB"] = round(psutil.Process().memory_info().rss / 2**20)
    mem["system_used_pct"] = psutil.virtual_memory().percent
except ImportError:
    pass
try:
    import subprocess

    mem["gpu_used_MiB"] = int(subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).split()[0])
except Exception:  # noqa: BLE001
    pass
print("memory:", mem)

report = {"checks": results, "memory": mem, "steps_per_s": sps, "num_envs": n}
(OUT / "check_scene_report.json").write_text(json.dumps(report, indent=2))
ok = all(r["ok"] for r in results.values())
print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED", "->", OUT)
env.close()
app.close()
raise SystemExit(0 if ok else 1)
