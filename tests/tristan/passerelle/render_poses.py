r"""Images de référence des poses à reproduire à la main sur le vrai SO-101 (test 1 de la passerelle sim <-> réel).

Le bras simulé (scène P4 « comme le réel » d'Alexis) est amené dans chaque pose par des consignes articulaires absolues
(radians), puis photographié de côté et de face. Les angles visés ET atteints sont écrits sur l'image et dans poses.json.

Lancer depuis tests/alexis (Python d'Isaac) :
    $env:PYTHONPATH = "$PWD\src"; & C:\dev\env_isaac_lab\Scripts\python.exe ..\tristan\passerelle\render_poses.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=str, default=str(Path(__file__).resolve().parent / "poses"))
parser.add_argument("--only", nargs="+", default=None, help="préfixes des poses à (re)rendre, ex. C ; les autres restent")
AppLauncher.add_app_launcher_args(parser)
_opts = {o for _a in parser._actions for o in _a.option_strings}
for _flag in ("--headless", "--enable_cameras"):
    if _flag not in _opts:
        parser.add_argument(_flag, action="store_true")
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app

import json  # noqa: E402
import math  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import cta_learn.sim.tasks  # noqa: E402,F401
from cta_learn.sim.robots import SO101  # noqa: E402
from cta_learn.sim.tasks.pick_place import CAM_H, CAM_W, _pinhole  # noqa: E402
from cta_learn.sim.tasks.pick_place_real import TASK_ID, PickPlaceRealEnvCfg  # noqa: E402

# degrés, ordre de l'embodiment : shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
POSES_DEG = {
    "A_depart_replie": (0.0, -95.0, 85.0, 50.0, 0.0, 0.0),    # pose de départ de la sim (sim/robots.py)
    "B_ready_pince_ouverte": (0.0, -10.0, 20.0, 75.0, 0.0, 100.0),  # « ready » de l'expert, pince à sa limite
    # vérification : signe de shoulder_pan (+45) et wrist_roll (+90), zéro de wrist_flex (pince alignée sur l'avant-bras)
    "C_verif_pan45_roll90_poignet0": (45.0, -10.0, 20.0, 0.0, 90.0, 0.0),
}
# vues autour de la base du robot (m) : œil, point visé
VIEWS = {"cote": ((0.06, -0.36, 0.16), (0.06, 0.0, 0.13)), "face": ((0.38, -0.08, 0.22), (0.04, 0.0, 0.12)),
         "dessus": ((-0.12, 0.0, 0.62), (0.08, 0.0, 0.05))}

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
cfg = PickPlaceRealEnvCfg()
cfg.seed = 3
cfg.episode_length_s = 1000.0
cfg.scene.closeup = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/pose_camera",
    offset=TiledCameraCfg.OffsetCfg(pos=(0.5, 0.5, 0.5), rot=(0.0, 0.0, 0.0, 1.0), convention="world"),
    data_types=["rgb"], spawn=_pinhole(18.0), width=CAM_W, height=CAM_H, update_period=0.0,
)
env = gym.make(TASK_ID, cfg=cfg).unwrapped
robot = env.scene["robot"]
cam = env.scene["closeup"]
dev = env.device
names = list(SO101.joint_names)
jids = [robot.joint_names.index(n) for n in names]
base = robot.data.root_pos_w.torch[0].clone()

report_file = out / "poses.json"
report = json.loads(report_file.read_text(encoding="utf-8")) if report_file.is_file() else {"joint_order": names}
report.setdefault("poses", {})
todo = {p: d for p, d in POSES_DEG.items() if not args.only or any(p.startswith(o) for o in args.only)}
env.reset()
for pose, deg in todo.items():
    target = torch.tensor([[math.radians(v) for v in deg]], device=dev)
    for _ in range(150):  # 5 s à 30 Hz pour atteindre et stabiliser la pose
        env.step(target)
    reached = [math.degrees(v) for v in robot.data.joint_pos.torch[0, jids].tolist()]
    report["poses"][pose] = {"target_deg": list(deg), "reached_deg": [round(v, 1) for v in reached]}
    text = "  ".join(f"{n}={v:.0f}" for n, v in zip(names, reached))
    for view, (eye, look) in VIEWS.items():
        cam.set_world_poses_from_view((base + torch.tensor(eye, device=dev)).unsqueeze(0),
                                      (base + torch.tensor(look, device=dev)).unsqueeze(0))
        for _ in range(4):
            env.sim.render()
            cam.update(dt=0.0, force_recompute=True)
        img = Image.fromarray(cam.data.output["rgb"].torch[0].cpu().numpy())
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, img.width, 34), fill=(0, 0, 0))
        draw.text((6, 3), f"{pose} - vue {view}", fill=(255, 255, 255))
        draw.text((6, 18), text + "  (deg, sim)", fill=(255, 255, 0))
        img.save(out / f"{pose}_{view}.png")
    print(pose, report["poses"][pose])

report["poses"] = {p: report["poses"][p] for p in POSES_DEG if p in report["poses"]}  # ordre A, B, C…
report_file.write_text(json.dumps(report, indent=1), encoding="utf-8")
env.close()
app.close()
