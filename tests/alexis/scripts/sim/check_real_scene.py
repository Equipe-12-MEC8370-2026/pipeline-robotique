"""Visual check of CTA-SO101-PickPlaceReal-v0 (P4): front, wrist and close-up images of the camera mount.

    $env:PYTHONPATH = "$PWD\\src"; & ..\\isaacsim-env\\Scripts\\python.exe scripts\\sim\\check_real_scene.py --headless --enable_cameras

Writes PNGs to outputs/p4_1/scene_check/ (3 resets: colours and wrist camera jitter change) and a JSON with the
mount pose read back from the stage (must equal the fitted transform).
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--resets", type=int, default=3)
parser.add_argument("--out", type=str, default="outputs/p4_1/scene_check")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import json  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402
from PIL import Image  # noqa: E402

import cta_learn.sim.tasks  # noqa: E402,F401
from cta_learn import REPO_ROOT  # noqa: E402
from cta_learn.sim.tasks.pick_place import CAM_H, CAM_W, _pinhole  # noqa: E402
from cta_learn.sim.tasks.pick_place_real import TASK_ID, PickPlaceRealEnvCfg  # noqa: E402
from cta_learn.sim.wrist_mount import mount_transform  # noqa: E402

out = REPO_ROOT / args.out
out.mkdir(parents=True, exist_ok=True)
cfg = PickPlaceRealEnvCfg()
cfg.seed = 3
cfg.scene.closeup = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/closeup_camera",
    offset=TiledCameraCfg.OffsetCfg(pos=(0.5, 0.5, 0.5), rot=(0.0, 0.0, 0.0, 1.0), convention="world"),
    data_types=["rgb"], spawn=_pinhole(24.0), width=CAM_W, height=CAM_H, update_period=0.0,
)
env = gym.make(TASK_ID, cfg=cfg).unwrapped
robot = env.scene["robot"]
close = env.scene["closeup"]
dev = env.device

# a pose with the gripper above the table, jaw open, so the wrist view shows fingers + table
hover = torch.tensor([[0.0, -0.35, 0.25, 1.2, 0.0, 0.6]], device=dev)  # rad, embodiment order
report = {}
for r in range(args.resets):
    obs, _ = env.reset()
    for _ in range(45):  # 1.5 s to reach the pose
        obs, *_ = env.step(hover)
    gp = robot.data.body_pos_w.torch[0, robot.body_names.index("gripper")]
    for view, eye in {"side": (0.10, 0.12, 0.05), "front": (0.13, -0.02, 0.04), "top": (0.02, 0.05, 0.14)}.items():
        close.set_world_poses_from_view(
            (gp + torch.tensor(eye, device=dev)).unsqueeze(0), (gp + torch.tensor([0, 0, -0.04], device=dev)).unsqueeze(0)
        )
        for _ in range(3):
            env.sim.render()
            close.update(dt=0.0, force_recompute=True)
        Image.fromarray(close.data.output["rgb"].torch[0].cpu().numpy()).save(out / f"reset{r}_closeup_{view}.png")
    obs, *_ = env.step(hover)
    for cam in ("front", "wrist"):
        Image.fromarray(obs["policy"][cam][0].cpu().numpy()).save(out / f"reset{r}_{cam}.png")

# mount pose on the stage vs the fitted transform
import omni.usd  # noqa: E402
from pxr import UsdGeom  # noqa: E402

stage = omni.usd.get_context().get_stage()
cache = UsdGeom.XformCache()
mount = np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath("/World/envs/env_0/Robot/gripper/camera_mount"))).T
link = np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath("/World/envs/env_0/Robot/gripper"))).T
rel = np.linalg.inv(link) @ mount
err_pos = float(np.linalg.norm(rel[:3, 3] - mount_transform()[:3, 3]))
err_rot = float(np.degrees(np.arccos(np.clip((np.trace(rel[:3, :3].T @ mount_transform()[:3, :3]) - 1) / 2, -1, 1))))
report["mount_vs_fit"] = {"pos_err_m": err_pos, "rot_err_deg": err_rot}
report["ok"] = err_pos < 1e-4 and err_rot < 0.1
(out / "check.json").write_text(json.dumps(report, indent=1))
print(json.dumps(report))
env.close()
app.close()
