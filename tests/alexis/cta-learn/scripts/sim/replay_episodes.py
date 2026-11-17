"""Gate 2 check: replay recorded actions open-loop and verify the task still succeeds.

Restores the recorded initial cube/target poses and robot joints, then applies the recorded
action sequence unchanged. Proves the stored actions (not the expert) solve the task.

    $env:PYTHONPATH = "$PWD\\src"; & ..\\isaacsim-env\\Scripts\\python.exe scripts\\sim\\replay_episodes.py --headless --episodes 0 7 42
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--raw", default="data/raw/p0_pickplace")
parser.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2])
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
from cta_learn.sim.tasks.pick_place import TASK_ID, PickPlaceEnvCfg  # noqa: E402

cfg = PickPlaceEnvCfg()
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
results = {}
for k in args.episodes:
    arrays, meta = load_episode(REPO_ROOT / args.raw / f"episode_{k:06d}")
    env.reset()
    for name, pos, quat in (("cube", arrays["cube_pos"][0], arrays["cube_quat"][0]),
                            ("target", arrays["target_pos"][0], np.array([0, 0, 0, 1.0]))):
        pose = torch.tensor(np.concatenate([pos, quat]), device=dev, dtype=torch.float32).unsqueeze(0)
        pose[:, :3] += origin
        env.scene[name].write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
        env.scene[name].write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=dev), env_ids=ids)
    q0 = torch.tensor(arrays["state"][0], device=dev, dtype=torch.float32).unsqueeze(0)
    robot = env.scene["robot"]
    robot.write_joint_position_to_sim_index(position=q0, env_ids=ids)
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q0), env_ids=ids)
    ok = False
    for a in arrays["action"]:
        _, _, term, trunc, _ = env.step(torch.tensor(a, device=dev, dtype=torch.float32).unsqueeze(0))
        if env.termination_manager.get_term("success")[0]:
            ok = True
            break
        if term[0] or trunc[0]:
            break
    results[k] = ok
    print(f"REPLAY episode {k}: {'success' if ok else 'FAIL'} ({len(arrays['action'])} steps)", flush=True)
summary = {"replayed": len(results), "successes": int(sum(results.values())), "per_episode": results}
print("REPLAY_SUMMARY", json.dumps(summary))
(REPO_ROOT / "outputs" / "p0_3" / "replay_report.json").write_text(json.dumps(summary, indent=2))
env.close()
app.close()
