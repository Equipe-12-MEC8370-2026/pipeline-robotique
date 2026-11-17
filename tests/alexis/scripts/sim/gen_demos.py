"""Run the scripted expert in CTA-SO101-PickPlace-v0; optionally record successful episodes.

Gate (no recording, 20 attempts, success-rate report):
    python scripts\\sim\\gen_demos.py --headless --enable_cameras --attempts 20 --debug_videos 3
Generation (keep successes only, raw format -> data/raw/p0_pickplace):
    python scripts\\sim\\gen_demos.py --headless --enable_cameras --record --successes 150 --attempts 300

Recovery demos (P0.6): inject a missed first grasp, record only from its detection to the end:
    python scripts\\sim\\gen_demos.py --headless --enable_cameras --record --out data/raw/p0_recovery
        --inject_miss 0.015 0.025 --record_from_failure --successes 50 --attempts 120 --seed 3

Run with isaacsim-env and PYTHONPATH=<repo>\\src (see README.md).
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--attempts", type=int, default=20, help="max episodes to run")
parser.add_argument("--successes", type=int, default=0, help="stop after this many successes (0 = run all attempts)")
parser.add_argument("--record", action="store_true", help="write successful episodes to --out")
parser.add_argument("--out", type=str, default="data/raw/p0_pickplace")
parser.add_argument("--debug_videos", type=int, default=0, help="also save the first N attempts (any outcome)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--inject_miss", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                    help="offset the first grasp by MIN..MAX m in a random direction (missed grasp)")
parser.add_argument("--record_from_failure", action="store_true",
                    help="start recording when the expert detects a missed grasp; episodes without one are dropped")
parser.add_argument("--report", type=str, default="outputs/p0_3/gen_report.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from dataclasses import asdict  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import cta_learn.sim.tasks  # noqa: E402,F401
from cta_learn import REPO_ROOT  # noqa: E402
from cta_learn.datagen.episode_io import EpisodeWriter, list_episodes  # noqa: E402
from cta_learn.datagen.pick_place_expert import PHASES, ExpertParams, PickPlaceExpert, yaw_from_quat_xyzw  # noqa: E402
from cta_learn.sim.ik import TcpIK  # noqa: E402
from cta_learn.sim.tasks.pick_place import (  # noqa: E402
    ARM_JOINTS,
    CAM_H,
    CAM_W,
    TASK_DESCRIPTION,
    TASK_ID,
    TCP_OFFSET,
    PickPlaceEnvCfg,
)

DT = 1 / 30
GRIPPER_SPEED = 3.0  # rad/s, rate limit of the gripper command


def resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else REPO_ROOT / path


out_dir = resolve(args.out)
debug_dir = REPO_ROOT / "outputs" / "p0_3" / "debug_episodes"
report_path = resolve(args.report)
report_path.parent.mkdir(parents=True, exist_ok=True)

cfg = PickPlaceEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.seed = args.seed
env = gym.make(TASK_ID, cfg=cfg).unwrapped
robot = env.scene["robot"]
ik = TcpIK(robot, ARM_JOINTS, TCP_OFFSET, dt=DT)
rng = np.random.default_rng(args.seed)
n = env.num_envs
origins = env.scene.env_origins

next_index = len(list_episodes(out_dir)) if args.record else 0
experts = [PickPlaceExpert() for _ in range(n)]
writers: list[EpisodeWriter | None] = [None] * n
grip_cmd = torch.zeros(n, device=env.device)
attempt_of = [0] * n
attempts = successes = 0
outcomes: Counter = Counter()
lengths: list[int] = []
kept: list[str] = []


def start_episode(i: int, obs) -> None:
    global attempts
    e = obs["expert"]
    params = ExpertParams.sample(rng)
    if args.inject_miss:
        r, ang = rng.uniform(*args.inject_miss), rng.uniform(0, 2 * math.pi)
        params.first_grasp_offset_xy = (r * math.cos(ang), r * math.sin(ang))
    experts[i].reset(
        e["tcp_pos"][i].cpu().numpy(),
        e["cube_pos"][i].cpu().numpy(),
        yaw_from_quat_xyzw(e["cube_quat"][i].cpu().numpy()),
        params,
        arm_q=obs["policy"]["joint_pos"][i, :5].cpu().numpy(),
    )
    grip_cmd[i] = obs["policy"]["joint_pos"][i, 5]
    ik.set_command([i], obs["policy"]["joint_pos"][i, :5].clone())
    attempt_of[i] = attempts
    attempts += 1
    writers[i] = None
    if args.record_from_failure:
        return  # the writer is opened when the expert detects the missed grasp
    if args.record or attempt_of[i] < args.debug_videos:
        root = out_dir if args.record else debug_dir
        idx = next_index + successes if args.record else attempt_of[i]
        writers[i] = EpisodeWriter(root, idx if args.record else attempt_of[i], {"front": (CAM_W, CAM_H), "wrist": (CAM_W, CAM_H)})


def end_episode(i: int, outcome: str) -> None:
    global successes
    if not (args.record_from_failure and writers[i] is None):
        outcomes[outcome] += 1
    w = writers[i]
    meta = {
        "task": TASK_DESCRIPTION,
        "success": outcome == "success",
        "outcome": outcome,
        "attempt": attempt_of[i],
        "seed": args.seed,
        "expert_params": asdict(experts[i].params),
        "expert_failure": experts[i].failure_reason,
        "regrasps": experts[i].regrasps,
        "recorded_from_failure": bool(args.record_from_failure),
    }
    if args.record_from_failure and w is None:
        outcome = f"no_missed_grasp ({outcome})"  # nothing recorded: does not count
        outcomes[outcome] += 1
        print(f"[attempt {attempt_of[i]:4d}] {outcome:<28} successes={successes}/{attempts}", flush=True)
        return
    if outcome == "success":
        successes += 1
    if w is not None:
        if outcome == "success" or not args.record:
            kept.append(str(w.keep(meta)))
        else:
            w.discard()
    writers[i] = None
    print(f"[attempt {attempt_of[i]:4d}] {outcome:<28} successes={successes}/{attempts}", flush=True)


obs, _ = env.reset()
for i in range(n):
    start_episode(i, obs)
steps_in_ep = [0] * n
t0 = time.time()
while app.is_running():
    e = obs["expert"]
    goals = torch.zeros(n, 3, device=env.device)
    yaws = torch.zeros(n, device=env.device)
    tilts = torch.zeros(n, device=env.device)
    phase_idx = [0] * n
    joint_goals: dict[int, torch.Tensor] = {}
    for i in range(n):
        cmd = experts[i].step(
            e["tcp_pos"][i].cpu().numpy(), e["cube_pos"][i].cpu().numpy(), e["target_pos"][i].cpu().numpy(), DT,
            arm_q=obs["policy"]["joint_pos"][i, :5].cpu().numpy(),
            cube_yaw=yaw_from_quat_xyzw(e["cube_quat"][i].cpu().numpy()),
        )
        if args.record_from_failure and writers[i] is None and experts[i].regrasps > 0:
            writers[i] = EpisodeWriter(out_dir, next_index + successes, {"front": (CAM_W, CAM_H), "wrist": (CAM_W, CAM_H)})
        if cmd.joint_goal is not None:
            joint_goals[i] = torch.as_tensor(cmd.joint_goal, device=env.device, dtype=torch.float32)
        goals[i] = torch.as_tensor(cmd.tcp_goal, device=env.device, dtype=torch.float32) + origins[i]
        yaws[i] = cmd.yaw
        tilts[i] = cmd.tilt_weight
        target_g = torch.tensor(cmd.gripper, device=env.device)
        grip_cmd[i] = grip_cmd[i] + torch.clamp(target_g - grip_cmd[i], -GRIPPER_SPEED * DT, GRIPPER_SPEED * DT)
        phase_idx[i] = PHASES.index(cmd.phase)
    arm_q = ik.compute(goals, yaws, tilts)
    for i, qg in joint_goals.items():
        arm_q[i] = qg
        ik.set_command([i], qg)
    action = torch.cat([arm_q, grip_cmd.unsqueeze(1)], dim=1)

    for i in range(n):
        if writers[i] is not None:
            writers[i].add(
                {"front": obs["policy"]["front"][i].cpu().numpy(), "wrist": obs["policy"]["wrist"][i].cpu().numpy()},
                state=obs["policy"]["joint_pos"][i].cpu().numpy(),
                action=action[i].cpu().numpy(),
                cube_pos=e["cube_pos"][i].cpu().numpy(),
                target_pos=e["target_pos"][i].cpu().numpy(),
                tcp_pos=e["tcp_pos"][i].cpu().numpy(),
                phase=np.int16(phase_idx[i]),
                cube_quat=e["cube_quat"][i].cpu().numpy(),
                cmd_yaw=np.float32(yaws[i].item()),
                ik_pos_err=np.float32(ik.last_pos_err[i].item()),
                ik_yaw_err=np.float32(ik.last_yaw_err[i].item()),
                ik_tilt=np.float32(ik.last_tilt[i].item()),
            )

    obs, _, terminated, truncated, _ = env.step(action)
    succ = env.termination_manager.get_term("success")
    dropped = env.termination_manager.get_term("cube_dropped")
    for i in range(n):
        steps_in_ep[i] += 1
        outcome = None
        if terminated[i] or truncated[i]:
            if succ[i]:
                outcome = "success"
            elif dropped[i]:
                outcome = f"cube_dropped (phase {experts[i].phase})"
            else:
                outcome = f"timeout (phase {experts[i].phase})"
        elif experts[i].failed:
            outcome = f"expert_{experts[i].failure_reason}"
            obs, _ = env.reset(env_ids=torch.tensor([i], device=env.device))
        if outcome is not None:
            lengths.append(steps_in_ep[i])
            steps_in_ep[i] = 0
            end_episode(i, outcome)
            if attempts < args.attempts and not (args.successes and successes >= args.successes):
                start_episode(i, obs)
            else:
                attempt_of[i] = args.attempts  # this env is finished
    if all(attempt_of[i] >= args.attempts for i in range(n)) and all(w is None for w in writers):
        break
    if args.successes and successes >= args.successes:
        break

elapsed = time.time() - t0
done = sum(outcomes.values())
report = {
    "attempts": done,
    "successes": successes,
    "success_rate": successes / max(done, 1),
    "outcomes": dict(outcomes),
    "mean_episode_steps": float(np.mean(lengths)) if lengths else 0.0,
    "elapsed_s": round(elapsed, 1),
    "recorded_dir": str(out_dir) if args.record else None,
    "saved_episodes": len(kept),
    "seed": args.seed,
}
report_path.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
for w in writers:
    if w is not None:
        w.discard()
env.close()
app.close()
