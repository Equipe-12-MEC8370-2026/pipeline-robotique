"""P3.3 gate 2: replay recorded actions open-loop in CTA-SO101-ProfileStack-v0 and check the stack still succeeds.

Restores the recorded initial poses of both parts and the robot joints, then applies the recorded actions unchanged
(and holds the last one up to 2 s: the GPU simulation is not bit-reproducible, success may come a few steps later).
Proves that the stored actions, not the expert, solve the task.

    $env:PYTHONPATH = "$PWD\\src"; & ..\\isaacsim-env\\Scripts\\python.exe scripts\\sim\\replay_profile_episodes.py --headless --episodes 0 100 250 499
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--raw", default="data/raw/p3_profile_stack")
parser.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2])
parser.add_argument("--extra_steps", type=int, default=60)
parser.add_argument("--report", default="outputs/p3_3/replay_report.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import json  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import cta_learn.sim.tasks  # noqa: E402,F401
from cta_learn import REPO_ROOT  # noqa: E402
from cta_learn.datagen.episode_io import load_episode  # noqa: E402
from cta_learn.sim import mdp  # noqa: E402
from cta_learn.sim.tasks.profile_stack import TASK_ID, ProfileStackEnvCfg  # noqa: E402

cfg = ProfileStackEnvCfg()
cfg.scene.num_envs = 1
cfg.scene.front = None
cfg.scene.wrist = None
cfg.observations.policy.front = None
cfg.observations.policy.wrist = None
cfg.events.front_camera = None
env = gym.make(TASK_ID, cfg=cfg).unwrapped
dev = env.device
ids = torch.tensor([0], device=dev, dtype=torch.int32)
origin = env.scene.env_origins[0]
robot = env.scene["robot"]
results = {}
for k in args.episodes:
    arrays, meta = load_episode(REPO_ROOT / args.raw / f"episode_{k:06d}")
    env.reset()
    for name, key in (("profile_short", "short"), ("profile_long", "long")):
        pose = torch.tensor(np.concatenate([arrays[f"{key}_pos"][0], arrays[f"{key}_quat"][0]]), device=dev,
                            dtype=torch.float32).unsqueeze(0)
        pose[:, :3] += origin
        env.scene[name].write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
        env.scene[name].write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=dev), env_ids=ids)
    # teleported parts on the table keep their old friction anchors: same fix as at reset
    mdp.break_rigid_contacts(env, ids, ("profile_short", "profile_long"))
    q0 = torch.tensor(arrays["state"][0], device=dev, dtype=torch.float32).unsqueeze(0)
    robot.write_joint_position_to_sim_index(position=q0, env_ids=ids)
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q0), env_ids=ids)
    actions = list(arrays["action"]) + [arrays["action"][-1]] * args.extra_steps
    ok, steps = False, 0
    for a in actions:
        _, _, term, trunc, _ = env.step(torch.tensor(a, device=dev, dtype=torch.float32).unsqueeze(0))
        steps += 1
        if env.termination_manager.get_term("success")[0]:
            ok = True
            break
        if term[0] or trunc[0]:
            break
    results[k] = {"success": ok, "steps": steps, "recorded_steps": len(arrays["action"])}
    print(f"REPLAY episode {k}: {'success' if ok else 'FAIL'} after {steps} steps (recorded {len(arrays['action'])})", flush=True)
summary = {"replayed": len(results), "successes": int(sum(r["success"] for r in results.values())), "per_episode": results}
print("REPLAY_SUMMARY", json.dumps(summary))
out = REPO_ROOT / args.report
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(summary, indent=2))
env.close()
app.close()
