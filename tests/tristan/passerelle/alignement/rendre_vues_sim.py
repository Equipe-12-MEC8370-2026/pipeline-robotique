r"""Images de RÉFÉRENCE pour aligner les vraies caméras : ce que la politique voit en sim (caméras « front » et
« wrist » de la scène P4, 640 x 480, exactement les observations envoyées à policy_server), dans une scène identique
à la scène réelle :
  * bras dans une pose de poses/poses.json (D par défaut : aller_a_pose.py la fait tenir au vrai bras) ;
  * cube et cible aux positions du début d'une démo (episode_000096 par défaut = placement du test 2) ;
  * AUCUNE variation aléatoire : caméras à leur pose nominale (tremblement 0), couleurs et lumière au milieu de
    leurs plages.
Sorties : cameras/ref_<pose>_front.png, cameras/ref_<pose>_wrist.png (+ ref_<pose>.json).

Depuis tests/alexis (Python d'Isaac) :
    $env:PYTHONPATH = "$PWD\src"; & C:\dev\env_isaac_lab\Scripts\python.exe ..\tristan\passerelle\alignement\rendre_vues_sim.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import chemins  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--pose", default="D")
parser.add_argument("--episode", default="episode_000096")
parser.add_argument("--raw", default=str(chemins.DEMOS_P4))
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
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import cta_learn.sim.tasks  # noqa: E402,F401
from cta_learn.sim.robots import SO101  # noqa: E402
from cta_learn.sim.tasks import pick_place_real as p4  # noqa: E402

poses = json.loads(chemins.POSES.read_text(encoding="utf-8"))["poses"]
nom = next(n for n in poses if n.startswith(args.pose.upper() + "_"))
q_deg = poses[nom]["target_deg"]
demo = np.load(Path(args.raw) / args.episode / "data.npz")
cube_pos, cube_quat_xyzw, target_pos = demo["cube_pos"][0], demo["cube_quat"][0], demo["target_pos"][0]
cube_yaw = 2 * math.atan2(float(cube_quat_xyzw[2]), float(cube_quat_xyzw[3]))


def milieu(lo, hi):
    return tuple((a + b) / 2 for a, b in zip(lo, hi))


cfg = p4.PickPlaceRealEnvCfg()
cfg.seed = 0
cfg.episode_length_s = 1000.0
ev = cfg.events
# positions exactes : position par défaut = celle de la démo, plage de tirage nulle
cfg.scene.cube.init_state.pos = (float(cube_pos[0]), float(cube_pos[1]), float(cube_pos[2]))
cfg.scene.target.init_state.pos = (float(target_pos[0]), float(target_pos[1]), float(target_pos[2]))
ev.cube.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (cube_yaw, cube_yaw)}
ev.target.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0)}
# caméras à leur pose nominale
ev.front_camera.params.update(pos_range=0.0, rot_range_deg=0.0)
ev.wrist_camera.params.update(pos_range=0.0, along_axis_range=0.0, rot_range_deg=0.0)
# apparence au milieu des plages de randomisation
ev.light.params.update(intensity_range=(1600.0, 1600.0), color_range=(0.9, 0.9))
ev.table_wood.params.update(brightness_range=(1.0, 1.0), tint_range=0.0)
for term in (ev.cube_color, ev.paper_color):
    m = milieu(term.params["color_low"], term.params["color_high"])
    term.params.update(color_low=m, color_high=m)
m = milieu(ev.robot_color.params["color_low"], ev.robot_color.params["color_high"])
ev.robot_color.params.update(color_low=m, color_high=m)

env = gym.make(p4.TASK_ID, cfg=cfg).unwrapped
robot = env.scene["robot"]
jids = [robot.joint_names.index(n) for n in SO101.joint_names]
obs, _ = env.reset()
target = torch.tensor([[math.radians(v) for v in q_deg]], device=env.device)
for _ in range(150):  # 5 s : pose atteinte et stable
    obs, *_ = env.step(target)
atteint = [round(math.degrees(v), 2) for v in robot.data.joint_pos.torch[0, jids].tolist()]
cube_fin = (env.scene["cube"].data.root_pos_w.torch[0] - env.scene.env_origins[0]).tolist()

out = chemins.ALIGNEMENT
out.mkdir(exist_ok=True)
for cam in ("front", "wrist"):
    img = obs["policy"][cam][0].cpu()
    if img.dtype != torch.uint8:
        img = img.float().clamp(0, 255).round().to(torch.uint8)
    Image.fromarray(img.numpy()).save(out / f"ref_{args.pose.upper()}_{cam}.png")
(out / f"ref_{args.pose.upper()}.json").write_text(json.dumps({
    "pose": nom, "cible_deg": q_deg, "atteint_deg": atteint, "episode": args.episode,
    "cube_env_m": [float(v) for v in cube_pos], "cube_yaw_deg": round(math.degrees(cube_yaw), 2),
    "cube_apres_pose_env_m": [round(v, 4) for v in cube_fin], "cible_env_m": [float(v) for v in target_pos],
}, indent=1), encoding="utf-8")
print(f"REF {nom} atteint {atteint} ; cube {np.round(cube_pos, 4).tolist()} -> {np.round(cube_fin, 4).tolist()}")
env.close()
app.close()
