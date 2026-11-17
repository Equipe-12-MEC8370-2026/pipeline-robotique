"""D12 trial: record SEED demos for Isaac Lab Mimic in CTA-SO101-ProfileStack-Mimic-v0 (robot starts at READY).

The seeds come from the P3 scripted expert (the trial asks whether a FEW demos + automatic amplification can replace
the expert; later seeds will come from teleoperation or the human video). Each step records what Mimic needs
(Isaac Lab HDF5 format, obs/datagen_info): TCP pose, part poses (env frame), target TCP pose (the expert's
command: goal position + gripper closing yaw, vertical), subtask end signals (from the expert's step/phase),
and the joint-position action. Only successful demos are written.

    $env:PYTHONPATH = "$PWD\\src"; & ..\\isaacsim-env\\Scripts\\python.exe scripts\\sim\\record_mimic_seeds.py --headless --seeds 10
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=int, default=10, help="successful seed demos to record")
parser.add_argument("--max_attempts", type=int, default=20)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--hold_steps", type=int, default=60, help="steps recorded after success (margin for generation)")
parser.add_argument("--out", default="data/mimic/p3_seeds.hdf5")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils import math as PoseUtils  # noqa: E402
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler  # noqa: E402

import cta_learn.sim.tasks  # noqa: E402,F401
from cta_learn import REPO_ROOT  # noqa: E402
from cta_learn.datagen.pick_place_expert import yaw_from_quat_xyzw  # noqa: E402
from cta_learn.datagen.stack_expert import PHASES, StackExpert, StackParams, plan_from_layout  # noqa: E402
from cta_learn.sim.tasks.profile_stack import LAYOUT  # noqa: E402
from cta_learn.sim.tasks.profile_stack_mimic import EEF, TASK_ID, ProfileStackMimicEnvCfg  # noqa: E402

DT = 1 / 30
GRIPPER_SPEED = 3.0
KEY = {"profile_short": "short", "profile_long": "long"}
LIFT, RETREAT = PHASES.index("lift"), PHASES.index("retreat")


def up_axis(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


def parts_of(e) -> dict:
    out = {}
    for p, k in KEY.items():
        q = e[f"{k}_quat"][0].cpu().numpy()
        out[p] = (e[f"{k}_pos"][0].cpu().numpy(), yaw_from_quat_xyzw(q), up_axis(q))
    return out


def signals(ex: StackExpert) -> dict[str, bool]:
    """Subtask ends from the expert's progress: a part is grasped once 'lift' starts; the short part is placed
    once its retreat starts."""
    k, ph = ex.step_index, PHASES.index(ex.phase)
    return {"grasp_short": k > 0 or ph >= LIFT, "place_short": k > 0 or ph >= RETREAT,
            "grasp_long": k > 0 and ph >= LIFT}


cfg = ProfileStackMimicEnvCfg()
cfg.scene.num_envs = 1
cfg.seed = args.seed
success_cfg = cfg.terminations.success
cfg.terminations.success = None   # judged here (the term would auto-reset the env as soon as it fires)
env = gym.make(TASK_ID, cfg=cfg).unwrapped
dev = env.device
ik = env._ik
rng = np.random.default_rng(args.seed)
plans = plan_from_layout(LAYOUT)
out = REPO_ROOT / args.out
out.parent.mkdir(parents=True, exist_ok=True)
handler = HDF5DatasetFileHandler()
handler.create(str(out), env_name=TASK_ID)

kept = attempts = 0
while kept < args.seeds and attempts < args.max_attempts:
    attempts += 1
    obs, _ = env.reset()
    ex = StackExpert(plans)
    q = obs["policy"]["joint_pos"]
    ex.reset(obs["expert"]["tcp_pos"][0].cpu().numpy(), parts_of(obs["expert"]), StackParams.sample(rng),
             arm_q=q[0, :5].cpu().numpy())
    ik.set_command([0], q[0, :5].clone())
    grip = q[0, 5].clone()
    ep = EpisodeData()
    outcome, steps, hold = None, 0, 0
    # after success, keep recording --hold_steps more steps (the expert holds its 'done' pose): a generated demo
    # is cut at the end of its source segments, and without this margin it often ended a few steps before the
    # 1 s rest the success needs (15/24 generation 'failures' had a correct stack, D12 trial 2026-11-15)
    while hold < args.hold_steps and steps < 1500:
        e = obs["expert"]
        cmd = ex.step(e["tcp_pos"][0].cpu().numpy(), parts_of(e), DT, arm_q=obs["policy"]["joint_pos"][0, :5].cpu().numpy())
        goal_env = torch.as_tensor(cmd.tcp_goal, device=dev, dtype=torch.float32)
        arm_q = ik.compute((goal_env + env.scene.env_origins[0]).unsqueeze(0), torch.tensor([cmd.yaw], device=dev),
                           torch.tensor([cmd.tilt_weight], device=dev))
        if cmd.joint_goal is not None:
            qg = torch.as_tensor(cmd.joint_goal, device=dev, dtype=torch.float32).unsqueeze(0)
            arm_q = qg
            ik.set_command([0], qg[0])
        grip = grip + torch.clamp(torch.tensor(cmd.gripper, device=dev) - grip, -GRIPPER_SPEED * DT, GRIPPER_SPEED * DT)
        action = torch.cat([arm_q[0], grip.reshape(1)]).unsqueeze(0)

        eef = env.get_robot_eef_pose(EEF)[0]
        if cmd.joint_goal is not None:   # joint-space move: the target is where the TCP is
            target = eef.clone()
        else:
            target = PoseUtils.make_pose(goal_env, PoseUtils.matrix_from_quat(ik.desired_quat(torch.tensor([cmd.yaw], device=dev))[0]))
        objs = env.get_object_poses()
        ep.add("actions", action[0])
        ep.add(f"obs/datagen_info/eef_pose/{EEF}", eef)
        ep.add(f"obs/datagen_info/target_eef_pose/{EEF}", target)
        for k, v in objs.items():
            ep.add(f"obs/datagen_info/object_pose/{k}", v[0])
        for k, v in signals(ex).items():
            ep.add(f"obs/datagen_info/subtask_term_signals/{k}", torch.tensor([v], device=dev))

        obs, _, term, trunc, _ = env.step(action)
        steps += 1
        if outcome == "success":
            hold += 1
        elif bool(success_cfg.func(env, **success_cfg.params)[0]):
            outcome = "success"
        elif ex.failed:
            outcome = f"expert: {ex.failure_reason}"
            break
        elif term[0] or trunc[0]:
            outcome = "terminated"
            break
    if outcome == "success":
        ep.pre_export()
        handler.write_episode(ep)
        handler.flush()
        kept += 1
    print(f"[seed attempt {attempts}] {outcome or 'timeout'} after {steps} steps -> kept {kept}/{args.seeds}", flush=True)

handler.close()
print(f"SEEDS {kept} demos -> {out}")
env.close()
app.close()
