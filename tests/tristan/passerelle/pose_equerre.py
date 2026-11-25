r"""Pose D « à l'équerre » : calcule en sim les angles qui donnent des alignements vérifiables au fil à plomb / équerre
sur le vrai bras, avec des repères visibles (centres des palonniers = axes des articulations, bout du doigt fixe) :
  * axe du coude exactement à la verticale de l'axe de l'épaule (shoulder_lift) ;
  * axe du poignet à la même hauteur que l'axe du coude (avant-bras « horizontal ») ;
  * bout du doigt fixe exactement à la verticale, sous l'axe du poignet (pince « vers le bas »).
Mesure aussi l'angle entre l'avant-bras et la pince quand wrist_flex = 0 (le zéro sim est-il « pince alignée » ?).
Ajoute la pose D à poses/poses.json et rend ses images.

Lancer depuis tests/alexis (Python d'Isaac) :
    $env:PYTHONPATH = "$PWD\src"; & C:\dev\env_isaac_lab\Scripts\python.exe ..\tristan\passerelle\pose_equerre.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--pose", choices=["D", "E"], default="D",
                    help="D : bras vertical, avant-bras horizontal, pince vers le bas ; "
                         "E : bras tendu vers le haut (épaule, coude, poignet et bout du doigt sur une même verticale)")
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
from isaaclab.utils import math as math_utils  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import cta_learn.sim.tasks  # noqa: E402,F401
from cta_learn.sim.robots import SO101  # noqa: E402
from cta_learn.sim.tasks.pick_place import CAM_H, CAM_W, FIXED_FINGER_INNER_X, FIXED_FINGER_TIP_Z, _pinhole  # noqa: E402
from cta_learn.sim.tasks.pick_place_real import TASK_ID, PickPlaceRealEnvCfg  # noqa: E402

OUT = Path(__file__).resolve().parent / "poses"
NAME = {"D": "D_equerre_bras_vertical_avantbras_horizontal_pince_bas",
        "E": "E_bras_tendu_vers_le_haut"}[args.pose]

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
print("corps du robot :", robot.body_names)
B = {n: robot.body_names.index(n) for n in ("upper_arm", "lower_arm", "wrist", "gripper")}
TIP = torch.tensor([[FIXED_FINGER_INNER_X, 0.0, FIXED_FINGER_TIP_Z]], device=dev)


def go(q_deg, steps=60):
    t = torch.tensor([[math.radians(v) for v in q_deg]], device=dev)
    for _ in range(steps):
        env.step(t)


def points():
    pos = robot.data.body_pos_w.torch[0]
    quat = robot.data.body_quat_w.torch[0]
    p = {n: pos[i] for n, i in B.items()}
    p["tip"] = pos[B["gripper"]] + math_utils.quat_apply(quat[B["gripper"]].unsqueeze(0), TIP)[0]
    return p


def angles():
    """(épaule→coude vs verticale, coude→poignet vs horizontale, poignet→bout du doigt vs verticale vers le bas), degrés,
    dans le plan vertical du bras (shoulder_pan = 0 : plan x-z)."""
    p = points()
    up = p["lower_arm"] - p["upper_arm"]
    fo = p["wrist"] - p["lower_arm"]
    gr = p["tip"] - p["wrist"]
    a_up = math.degrees(math.atan2(up[0].item(), up[2].item()))
    if args.pose == "E":  # avant-bras et pince verticaux vers le haut
        a_fo = math.degrees(math.atan2(fo[0].item(), fo[2].item()))
        a_gr = math.degrees(math.atan2(gr[0].item(), gr[2].item()))
    else:
        a_fo = math.degrees(math.atan2(fo[2].item(), fo[0].item()))
        a_gr = math.degrees(math.atan2(gr[0].item(), -gr[2].item()))
    return a_up, a_fo, a_gr, p


def reached():
    return [math.degrees(v) for v in robot.data.joint_pos.torch[0, jids].tolist()]


env.reset()
q = [0.0, -10.0, 20.0, 75.0, 0.0, 0.0] if args.pose == "D" else [0.0, -14.0, -74.0, -5.0, 0.0, 0.0]
go(q, 120)
# résolution articulation par articulation (chaque angle mesuré dépend des articulations en amont), 2 passes
for _ in range(3):
    for k, idx in ((1, 0), (2, 1), (3, 2)):  # (indice articulation, indice angle mesuré)
        for _ in range(4):
            a0 = angles()[idx]
            q[k] += 2.0
            go(q, 40)
            slope = (angles()[idx] - a0) / 2.0
            q[k] -= 2.0 + a0 / slope
            go(q, 60)
a_up, a_fo, a_gr, p = angles()
q_reached = reached()
print(f"pose {args.pose} : consignes {[round(v, 2) for v in q]}  atteint {[round(v, 2) for v in q_reached]}")
print(f"  alignements résiduels : bras {a_up:+.2f}°  avant-bras {a_fo:+.2f}°  pince {a_gr:+.2f}°")

grip_vs_fore_at0 = None
if args.pose == "D":
    # zéro du poignet : même bras/avant-bras, wrist_flex = 0 -> angle entre l'avant-bras et la pince
    q0 = list(q)
    q0[3] = 0.0
    go(q0, 90)
    _, a_fo0, a_gr0, _ = angles()
    # pince : angle sous l'horizontale = 90 - a_gr0 ; avant-bras : angle sous l'horizontale = -a_fo0
    grip_vs_fore_at0 = (90.0 - a_gr0) + a_fo0  # 0 = pince exactement dans le prolongement de l'avant-bras
    print(f"wrist_flex = 0 : pince vs prolongement de l'avant-bras = {grip_vs_fore_at0:+.2f}° "
          f"(+ = pince inclinée vers le bas)")
    go(q, 90)

# images
base = robot.data.root_pos_w.torch[0].clone()
views = ({"cote": ((0.10, -0.42, 0.20), (0.10, 0.0, 0.18)), "cote_proche_poignet": (None, None)} if args.pose == "D"
         else {"cote": ((0.02, -0.60, 0.25), (0.02, 0.0, 0.25))})
text = "  ".join(f"{n}={v:.1f}" for n, v in zip(names, q_reached))
OUT.mkdir(parents=True, exist_ok=True)
for view, (eye, look) in views.items():
    if eye is None:  # gros plan sur le poignet
        w = points()["wrist"]
        eye_w, look_w = w + torch.tensor([0.0, -0.22, 0.0], device=dev), w + torch.tensor([0.0, 0.0, -0.04], device=dev)
    else:
        eye_w, look_w = base + torch.tensor(eye, device=dev), base + torch.tensor(look, device=dev)
    cam.set_world_poses_from_view(eye_w.unsqueeze(0), look_w.unsqueeze(0))
    for _ in range(4):
        env.sim.render()
        cam.update(dt=0.0, force_recompute=True)
    img = Image.fromarray(cam.data.output["rgb"].torch[0].cpu().numpy())
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, img.width, 34), fill=(0, 0, 0))
    d.text((6, 3), f"{NAME} - vue {view}", fill=(255, 255, 255))
    d.text((6, 18), text + "  (deg, sim)", fill=(255, 255, 0))
    img.save(OUT / f"{NAME}_{view}.png")

rep_file = OUT / "poses.json"
rep = json.loads(rep_file.read_text(encoding="utf-8"))
rep["poses"][NAME] = {"target_deg": [round(v, 2) for v in q], "reached_deg": [round(v, 2) for v in q_reached],
                      "alignements_residuels_deg": {"bras": round(a_up, 2), "avant_bras": round(a_fo, 2),
                                                    "pince": round(a_gr, 2)}}
if grip_vs_fore_at0 is not None:
    rep["wrist_flex_0_pince_vs_avant_bras_deg"] = round(grip_vs_fore_at0, 2)
rep_file.write_text(json.dumps(rep, indent=1), encoding="utf-8")
env.close()
app.close()
