"""Interactive viewer for CTA-SO101-PickPlace-v0 (visual validation).

PowerShell, from the repo root:
    $env:PYTHONPATH = "$PWD\\src"; & ..\\isaacsim-env\\Scripts\\python.exe scripts\\sim\\view_scene.py --enable_cameras --viz kit

The arm alternates between its folded start pose and an extended pose; the scene is
reset (new cube/target/light/table colour/camera jitter) every cycle. Each reset
saves the two camera images to outputs/p0_2/view_<n>_{front,wrist}.png.
Close the window to quit.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--cycle_s", type=float, default=6.0, help="seconds per start/extend/reset cycle")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import math  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import cta_learn.sim.tasks  # noqa: E402,F401
from cta_learn import REPO_ROOT  # noqa: E402
from cta_learn.sim.robots import SO101, SO101_START_POSE_DEG  # noqa: E402
from cta_learn.sim.tasks.pick_place import TASK_ID, PickPlaceEnvCfg  # noqa: E402

OUT = REPO_ROOT / "outputs" / "p0_2"
OUT.mkdir(parents=True, exist_ok=True)

cfg = PickPlaceEnvCfg()
cfg.scene.num_envs = 1
cfg.episode_length_s = 1e6  # the viewer controls resets
env = gym.make(TASK_ID, cfg=cfg).unwrapped

start = torch.tensor([[math.radians(SO101_START_POSE_DEG[j.name]) for j in SO101.joints]], device=env.device)
extend = start.clone()
extend[0, 1:4] = torch.tensor([math.radians(-20.0), math.radians(20.0), math.radians(60.0)])
extend[0, 5] = math.radians(60.0)  # open gripper

steps_per_cycle = int(args.cycle_s * 30)
obs, _ = env.reset()
step, n = 0, 0
while app.is_running():
    phase = (step % steps_per_cycle) / steps_per_cycle
    q = start if phase < 0.3 or phase > 0.8 else extend
    obs, *_ = env.step(q)
    step += 1
    if step % steps_per_cycle == 0:
        try:
            from PIL import Image

            for cam in ("front", "wrist"):
                Image.fromarray(obs["policy"][cam][0].cpu().numpy()).save(OUT / f"view_{n}_{cam}.png")
        except ImportError:
            pass
        n += 1
        obs, _ = env.reset()

env.close()
app.close()
