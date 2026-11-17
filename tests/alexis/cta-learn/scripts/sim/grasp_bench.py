"""Isolated grasp bench: pick the cube, hold it, release it. One parameter set per run.

Each trial runs the scripted expert up to the end of 'lift', holds 1 s, then opens the gripper
and checks the cube falls free. Outcomes:
    ok        held above 4 cm for 1 s, then released and fell back to the table
    pushed    cube moved > 6 mm during descend (finger landed on it)
    slipped   cube not lifted (still on the table after lift)
    wedged    cube lifted but more than 15 mm above the TCP (squeezed up into the palm)
    stuck     held fine but did not fall after the gripper opened
    ik_fail   expert timeout before lift ended
Results are appended to outputs/p0_3/grasp_bench.jsonl.

Example (PowerShell, repo root, PYTHONPATH=src):
    & ..\\isaacsim-env\\Scripts\\python.exe scripts\\sim\\grasp_bench.py --headless --trials 20 --grip_effort 0.2 --label leisaac_effort
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--label", type=str, default="baseline")
parser.add_argument("--trials", type=int, default=20)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--tcp_x", type=float, default=None, help="TCP x in gripper frame (m)")
parser.add_argument("--tcp_above_tip", type=float, default=0.010, help="TCP height above the fixed fingertip (m)")
parser.add_argument("--grasp_h", type=float, default=None, help="TCP height when closing (m)")
parser.add_argument("--grip_effort", type=float, default=None, help="gripper effort limit (N.m)")
parser.add_argument("--grip_stiffness", type=float, default=None)
parser.add_argument("--grip_close_deg", type=float, default=None)
parser.add_argument("--grip_open_deg", type=float, default=None)
parser.add_argument("--close_time", type=float, default=None)
parser.add_argument("--cube_mass", type=float, default=None)
parser.add_argument("--cube_friction", type=float, default=None)
parser.add_argument("--finger_friction", type=float, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaaclab.envs.mdp as base_mdp  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.managers import EventTermCfg, SceneEntityCfg  # noqa: E402

import cta_learn.sim.tasks  # noqa: E402,F401
from cta_learn import REPO_ROOT  # noqa: E402
from cta_learn.datagen.pick_place_expert import ExpertParams, PickPlaceExpert, yaw_from_quat_xyzw  # noqa: E402
from cta_learn.sim import mdp  # noqa: E402
from cta_learn.sim.ik import TcpIK  # noqa: E402
from cta_learn.sim.tasks.pick_place import ARM_JOINTS, FIXED_FINGER_TIP_Z, TASK_ID, TCP_OFFSET, PickPlaceEnvCfg  # noqa: E402

DT = 1 / 30
tcp = (args.tcp_x if args.tcp_x is not None else TCP_OFFSET[0], TCP_OFFSET[1], FIXED_FINGER_TIP_Z + args.tcp_above_tip)

cfg = PickPlaceEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.seed = args.seed
cfg.episode_length_s = 1e6
# no cameras: the bench only needs physics
cfg.scene.front = None
cfg.scene.wrist = None
cfg.observations.policy.front = None
cfg.observations.policy.wrist = None
cfg.events.front_camera = None
cfg.terminations.success = None
for fr in cfg.scene.ee_frame.target_frames:
    if fr.name == "tcp":
        fr.offset.pos = tcp
grip = cfg.scene.robot.actuators["sts3215-gripper"]
if args.grip_effort is not None:
    grip.effort_limit_sim = args.grip_effort
if args.grip_stiffness is not None:
    grip.stiffness = args.grip_stiffness
if args.cube_mass is not None:
    cfg.scene.cube.spawn.mass_props.mass = args.cube_mass
if args.cube_friction is not None:
    cfg.scene.cube.spawn.physics_material.static_friction = args.cube_friction
    cfg.scene.cube.spawn.physics_material.dynamic_friction = args.cube_friction
if args.finger_friction is not None:
    cfg.events.finger_friction = EventTermCfg(
        func=base_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["gripper", "jaw"]),
            "static_friction_range": (args.finger_friction,) * 2,
            "dynamic_friction_range": (args.finger_friction,) * 2,
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )

env = gym.make(TASK_ID, cfg=cfg).unwrapped
n = env.num_envs
ik = TcpIK(env.scene["robot"], ARM_JOINTS, tcp, dt=DT)
origins = env.scene.env_origins


def params() -> ExpertParams:
    p = ExpertParams()
    if args.grasp_h is not None:
        p.grasp_height = args.grasp_h
    if args.grip_close_deg is not None:
        p.gripper_closed = math.radians(args.grip_close_deg)
    if args.grip_open_deg is not None:
        p.gripper_open = math.radians(args.grip_open_deg)
    if args.close_time is not None:
        p.close_time = args.close_time
    return p


experts = [PickPlaceExpert() for _ in range(n)]
stage = ["run"] * n          # run -> hold -> release -> (done)
t_stage = [0.0] * n
cube_at_descend = [None] * n
grip_cmd = torch.zeros(n, device=env.device)
outcomes: Counter = Counter()
started = 0
details = []


def start(i, obs):
    global started
    e = obs["expert"]
    experts[i].reset(e["tcp_pos"][i].cpu().numpy(), e["cube_pos"][i].cpu().numpy(),
                     yaw_from_quat_xyzw(e["cube_quat"][i].cpu().numpy()), params(),
                     arm_q=obs["policy"]["joint_pos"][i, :5].cpu().numpy())
    ik.set_command([i], obs["policy"]["joint_pos"][i, :5].clone())
    grip_cmd[i] = obs["policy"]["joint_pos"][i, 5]
    stage[i], t_stage[i], cube_at_descend[i] = "run", 0.0, None
    started += 1


def finish(i, outcome, obs):
    outcomes[outcome] += 1
    details.append(outcome)
    if started < args.trials:
        o, _ = env.reset(env_ids=torch.tensor([i], device=env.device))
        start(i, o)
        return o
    stage[i] = "idle"
    return obs


obs, _ = env.reset()
for i in range(n):
    if started < args.trials:
        start(i, obs)
    else:
        stage[i] = "idle"
hold_cmd = [None] * n
t0 = time.time()
while app.is_running() and any(s != "idle" for s in stage):
    e = obs["expert"]
    goals = torch.zeros(n, 3, device=env.device)
    yaws = torch.zeros(n, device=env.device)
    tilts = torch.full((n,), 0.3, device=env.device)
    joint_goals = {}
    for i in range(n):
        if stage[i] == "idle":
            goals[i] = e["tcp_pos"][i] + origins[i]
            continue
        tcp_i, cube_i = e["tcp_pos"][i].cpu().numpy(), e["cube_pos"][i].cpu().numpy()
        if stage[i] == "run":
            cmd = experts[i].step(tcp_i, cube_i, e["target_pos"][i].cpu().numpy(), DT,
                                  arm_q=obs["policy"]["joint_pos"][i, :5].cpu().numpy())
            if cmd.phase == "descend" and cube_at_descend[i] is None:
                cube_at_descend[i] = cube_i.copy()
            if cmd.joint_goal is not None:
                joint_goals[i] = torch.as_tensor(cmd.joint_goal, device=env.device, dtype=torch.float32)
            goals[i] = torch.as_tensor(cmd.tcp_goal, device=env.device, dtype=torch.float32) + origins[i]
            yaws[i], tilts[i] = cmd.yaw, cmd.tilt_weight
            target_g = cmd.gripper
            if experts[i].phase == "transport":   # lift finished -> hold
                stage[i], t_stage[i], hold_cmd[i] = "hold", 0.0, (goals[i].clone(), cmd.yaw)
            if experts[i].failed:
                obs = finish(i, "ik_fail", obs)
                continue
        else:
            goals[i], yaws[i] = hold_cmd[i]
            target_g = experts[i].params.gripper_closed if stage[i] == "hold" else experts[i].params.gripper_release
            t_stage[i] += DT
        grip_cmd[i] = grip_cmd[i] + float(np.clip(target_g - grip_cmd[i].item(), -3.0 * DT, 3.0 * DT))
    arm_q = ik.compute(goals, yaws, tilts)
    for i, qg in joint_goals.items():
        arm_q[i] = qg
        ik.set_command([i], qg)
    obs, *_ = env.step(torch.cat([arm_q, grip_cmd.unsqueeze(1)], dim=1))
    e = obs["expert"]
    for i in range(n):
        if stage[i] in ("idle", "run"):
            continue
        cube_i, tcp_i = e["cube_pos"][i].cpu().numpy(), e["tcp_pos"][i].cpu().numpy()
        if stage[i] == "hold" and t_stage[i] >= 1.0:
            moved = np.linalg.norm(cube_i[:2] - cube_at_descend[i][:2]) if cube_at_descend[i] is not None else 0.0
            if cube_i[2] < 0.03:
                obs = finish(i, "pushed" if moved > 0.006 else "slipped", obs)
            elif cube_i[2] - tcp_i[2] > 0.015:
                obs = finish(i, "wedged", obs)
            else:
                stage[i], t_stage[i] = "release", 0.0
        elif stage[i] == "release" and t_stage[i] >= 1.0:
            obs = finish(i, "ok" if cube_i[2] < 0.025 else "stuck", obs)

res = {
    "label": args.label,
    "trials": sum(outcomes.values()),
    "ok_rate": outcomes["ok"] / max(1, sum(outcomes.values())),
    "outcomes": dict(outcomes),
    "config": {k: v for k, v in vars(args).items() if k in (
        "tcp_x", "tcp_above_tip", "grasp_h", "grip_effort", "grip_stiffness", "grip_close_deg", "grip_open_deg",
        "close_time", "cube_mass", "cube_friction", "finger_friction", "seed")},
    "tcp_offset": tcp,
    "elapsed_s": round(time.time() - t0, 1),
}
out = REPO_ROOT / "outputs" / "p0_3" / "grasp_bench.jsonl"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "a", encoding="utf-8") as f:
    f.write(json.dumps(res) + "\n")
print("BENCH", json.dumps(res))
env.close()
app.close()
