"""Run the P3 scripted expert (TaskSpec -> two chained pick-and-places) in CTA-SO101-ProfileStack-v0.

Gate (P3.2, no recording, success-rate report + debug videos of the first attempts):
    python scripts\\sim\\gen_profile_demos.py --headless --enable_cameras --attempts 20 --debug_videos 3
Generation (P3.3, keep successes only, raw format -> data/raw/p3_profile_stack):
    python scripts\\sim\\gen_profile_demos.py --headless --enable_cameras --record --successes 500 --attempts 700 --seed 1

Run with isaacsim-env and PYTHONPATH=<repo>\\src (see docs/HANDOFF.md). num_envs=1 by default: the reset event
mdp.break_rigid_contacts adds one physics step to the envs not being reset (none with a single env).
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--attempts", type=int, default=20)
parser.add_argument("--successes", type=int, default=0, help="stop after this many successes (0 = run all attempts)")
parser.add_argument("--record", action="store_true")
parser.add_argument("--out", type=str, default="data/raw/p3_profile_stack")
parser.add_argument("--debug_videos", type=int, default=0, help="also save the first N attempts (any outcome)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--ood", action="store_true", help="out-of-range start poses (use_ood_ranges)")
parser.add_argument("--report", type=str, default="outputs/p3_2/gate.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import json  # noqa: E402
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
from cta_learn.datagen.pick_place_expert import yaw_from_quat_xyzw  # noqa: E402
from cta_learn.datagen.stack_expert import PHASES, StackExpert, StackParams, plan_from_layout  # noqa: E402
from cta_learn.sim.ik import TcpIK  # noqa: E402
from cta_learn.sim.tasks.pick_place import ARM_JOINTS, CAM_H, CAM_W  # noqa: E402
from cta_learn.retarget.profile_stack import stack_success  # noqa: E402
from cta_learn.sim.tasks.profile_stack import LAYOUT, TASK_DESCRIPTION, TASK_ID, TCP_OFFSET, ProfileStackEnvCfg  # noqa: E402

DT = 1 / 30
GRIPPER_SPEED = 3.0  # rad/s
PARTS = {"profile_short": "short", "profile_long": "long"}


def resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else REPO_ROOT / path


out_dir = resolve(args.out)
report_path = resolve(args.report)
debug_dir = report_path.parent / "debug_episodes"
report_path.parent.mkdir(parents=True, exist_ok=True)

cfg = ProfileStackEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.seed = args.seed
if args.ood:
    cfg.use_ood_ranges()
env = gym.make(TASK_ID, cfg=cfg).unwrapped
robot = env.scene["robot"]
ik = TcpIK(robot, ARM_JOINTS, TCP_OFFSET, dt=DT)
rng = np.random.default_rng(args.seed)
n, origins = env.num_envs, env.scene.env_origins
plans = plan_from_layout(LAYOUT)

next_index = len(list_episodes(out_dir)) if args.record else 0
experts = [StackExpert(plans) for _ in range(n)]
writers: list[EpisodeWriter | None] = [None] * n
grip_cmd = torch.zeros(n, device=env.device)
attempt_of = [0] * n
attempts = successes = 0
outcomes: Counter = Counter()
lengths: list[int] = []
per_attempt: list[dict] = []
start_pose: list[dict | None] = [None] * n
at_failure: list[dict | None] = [None] * n
last_goal = [None] * n
tilt_after_release = [0.0] * n   # max tilt (deg) of the long part once the expert opened the gripper on it
RELEASED = {PHASES.index(ph) for ph in ("release", "clear", "slide", "retreat", "done")}


def long_tilt_deg(q) -> float:
    x, y = float(q[0]), float(q[1])
    return float(np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1.0, 1.0))))


def up_axis(q) -> np.ndarray:
    """World direction of the part's local z axis, quaternion (x, y, z, w)."""
    x, y, z, w = (float(v) for v in q)
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


def parts_of(e, i) -> dict:
    out = {}
    for obj, k in PARTS.items():
        q = e[f"{k}_quat"][i].cpu().numpy()
        out[obj] = (e[f"{k}_pos"][i].cpu().numpy(), yaw_from_quat_xyzw(q), up_axis(q))
    return out


def start_episode(i: int, obs) -> None:
    global attempts
    e = obs["expert"]
    experts[i].reset(e["tcp_pos"][i].cpu().numpy(), parts_of(e, i), StackParams.sample(rng),
                     arm_q=obs["policy"]["joint_pos"][i, :5].cpu().numpy())
    grip_cmd[i] = obs["policy"]["joint_pos"][i, 5]
    ik.set_command([i], obs["policy"]["joint_pos"][i, :5].clone())
    attempt_of[i] = attempts
    attempts += 1
    writers[i] = None
    start_pose[i] = {obj: [round(float(v), 4) for v in (*pos[:2], yaw)] for obj, (pos, yaw, _) in parts_of(e, i).items()}
    if args.record or attempt_of[i] < args.debug_videos:
        root, idx = (out_dir, next_index + successes) if args.record else (debug_dir, attempt_of[i])
        writers[i] = EpisodeWriter(root, idx, {"front": (CAM_W, CAM_H), "wrist": (CAM_W, CAM_H)})


def end_episode(i: int, outcome: str, steps: int) -> None:
    global successes
    outcomes[outcome] += 1
    ex = experts[i]
    if outcome == "success":
        successes += 1
    per_attempt.append({"attempt": attempt_of[i], "outcome": outcome, "steps": steps, "regrasps": ex.regrasps,
                        "long_tilt_after_release_deg": round(tilt_after_release[i], 2), "after_done": after_done[i],
                        "start_xyyaw": start_pose[i], "at_failure": at_failure[i]})
    if at_failure[i]:
        print(f"    start {start_pose[i]} | at failure {at_failure[i]}", flush=True)
    at_failure[i] = None
    if after_done[i] and outcome != "success":
        print(f"    why no success 1.5 s after the expert finished: {after_done[i]}", flush=True)
    after_done[i], done_steps[i] = None, 0
    tilt_after_release[i] = 0.0
    w = writers[i]
    if w is not None:
        meta = {"task": TASK_DESCRIPTION, "success": outcome == "success", "outcome": outcome, "attempt": attempt_of[i],
                "seed": args.seed, "ood": args.ood, "expert_params": asdict(ex.params), "expert_failure": ex.failure_reason,
                "regrasps": ex.regrasps, "taskspec": LAYOUT["from_taskspec"], "layout": LAYOUT["task_id"]}
        if outcome == "success" or not args.record:
            w.keep(meta)
        else:
            w.discard()
    writers[i] = None
    print(f"[attempt {attempt_of[i]:4d}] {outcome:<42} steps={steps:4d} tilt_after_release={per_attempt[-1]['long_tilt_after_release_deg']:5.1f} deg"
          f" successes={successes}/{attempts}", flush=True)


obs, _ = env.reset()
for i in range(n):
    start_episode(i, obs)
steps_in_ep = [0] * n
done_steps = [0] * n
after_done: list[dict | None] = [None] * n


def diagnose(e, i) -> dict:
    """Which success checks fail (state 1.5 s after the expert finished), for the report."""
    o = lambda k: e[k][i:i + 1]  # noqa: E731
    goal = torch.tensor([LAYOUT["goals"]["profile_short"]["xyyaw"][:2]], device=env.device)
    _, checks = stack_success(o("short_pos"), o("short_quat"), o("long_pos"), o("long_quat"), o("tcp_pos"), goal, LAYOUT, torch)
    lp, sp = o("long_pos")[0], o("short_pos")[0]
    return {"failed_checks": [k for k, v in checks.items() if not bool(v[0])],
            "long_minus_short_mm": [round(float(v) * 1000, 1) for v in (lp - sp)],
            "short_to_goal_mm": round(float(torch.linalg.vector_norm(sp[:2] - goal[0])) * 1000, 1),
            "tcp_to_long_mm": round(float(torch.linalg.vector_norm(o("tcp_pos")[0] - lp)) * 1000, 1),
            "long_speed": round(float(torch.linalg.vector_norm(env.scene["profile_long"].data.root_lin_vel_w.torch[i])), 4),
            "long_ang_speed": round(float(torch.linalg.vector_norm(env.scene["profile_long"].data.root_ang_vel_w.torch[i])), 4)}
t0 = time.time()
while app.is_running():
    e = obs["expert"]
    goals = torch.zeros(n, 3, device=env.device)
    yaws = torch.zeros(n, device=env.device)
    tilts = torch.zeros(n, device=env.device)
    phase_idx, step_idx = [0] * n, [0] * n
    joint_goals: dict[int, torch.Tensor] = {}
    for i in range(n):
        cmd = experts[i].step(e["tcp_pos"][i].cpu().numpy(), parts_of(e, i), DT, arm_q=obs["policy"]["joint_pos"][i, :5].cpu().numpy())
        if cmd.joint_goal is not None:
            joint_goals[i] = torch.as_tensor(cmd.joint_goal, device=env.device, dtype=torch.float32)
        goals[i] = torch.as_tensor(cmd.tcp_goal, device=env.device, dtype=torch.float32) + origins[i]
        yaws[i], tilts[i] = cmd.yaw, cmd.tilt_weight
        target_g = torch.tensor(cmd.gripper, device=env.device)
        grip_cmd[i] = grip_cmd[i] + torch.clamp(target_g - grip_cmd[i], -GRIPPER_SPEED * DT, GRIPPER_SPEED * DT)
        phase_idx[i], step_idx[i] = PHASES.index(cmd.phase), experts[i].step_index
        g = experts[i].last_goal
        last_goal[i] = None if cmd.joint_goal is not None or g is None else [round(float(v), 4) for v in g]
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
                short_pos=e["short_pos"][i].cpu().numpy(), short_quat=e["short_quat"][i].cpu().numpy(),
                long_pos=e["long_pos"][i].cpu().numpy(), long_quat=e["long_quat"][i].cpu().numpy(),
                tcp_pos=e["tcp_pos"][i].cpu().numpy(),
                phase=np.int16(phase_idx[i]), step=np.int16(step_idx[i]),
                cmd_yaw=np.float32(yaws[i].item()),
                ik_pos_err=np.float32(ik.last_pos_err[i].item()), ik_tilt=np.float32(ik.last_tilt[i].item()),
            )

    for i in range(n):
        if step_idx[i] == 1 and phase_idx[i] in RELEASED:
            tilt_after_release[i] = max(tilt_after_release[i], long_tilt_deg(e["long_quat"][i].cpu().numpy()))
    obs, _, terminated, truncated, _ = env.step(action)
    tm = env.termination_manager
    succ, sd, ld = tm.get_term("success"), tm.get_term("short_dropped"), tm.get_term("long_dropped")
    for i in range(n):
        steps_in_ep[i] += 1
        if experts[i].done:
            done_steps[i] += 1
            if done_steps[i] == 45:
                after_done[i] = diagnose(obs["expert"], i)
        ex, outcome = experts[i], None
        where = f"step {ex.step_index} {ex.phase}"
        if terminated[i] or truncated[i]:
            if succ[i]:
                outcome = "success"
            elif sd[i] or ld[i]:
                outcome = f"{'short' if sd[i] else 'long'}_dropped ({where})"
            else:
                outcome = f"timeout ({where})"
        elif ex.failed:
            outcome = f"expert: {ex.failure_reason}"
            try:  # diagnostics only: must never stop a generation run
                at_failure[i] = {"tcp": np.round(obs["expert"]["tcp_pos"][i].cpu().numpy(), 4).tolist(), "goal": last_goal[i],
                                 "ik_pos_err_mm": round(float(ik.last_pos_err[i].item()) * 1000, 1),
                                 "ik_tilt_deg": round(float(np.degrees(ik.last_tilt[i].item())), 1),
                                 "arm_deg": np.round(np.degrees(obs["policy"]["joint_pos"][i, :5].cpu().numpy()), 1).tolist()}
            except Exception as exc:  # noqa: BLE001
                at_failure[i] = {"diagnostic_error": repr(exc)}
            obs, _ = env.reset(env_ids=torch.tensor([i], device=env.device))
        if outcome is not None:
            lengths.append(steps_in_ep[i])
            end_episode(i, outcome, steps_in_ep[i])
            steps_in_ep[i] = 0
            if attempts < args.attempts and not (args.successes and successes >= args.successes):
                start_episode(i, obs)
            else:
                attempt_of[i] = args.attempts
    if all(attempt_of[i] >= args.attempts for i in range(n)) and all(w is None for w in writers):
        break
    if args.successes and successes >= args.successes:
        break

done = sum(outcomes.values())
tilts = [a["long_tilt_after_release_deg"] for a in per_attempt if a["outcome"] == "success"]
report = {"attempts": done, "long_tilt_after_release_deg_successes": {"max": max(tilts, default=0.0),
          "mean": float(np.mean(tilts)) if tilts else 0.0}, "successes": successes, "success_rate": successes / max(done, 1), "outcomes": dict(outcomes),
          "mean_episode_steps": float(np.mean(lengths)) if lengths else 0.0, "elapsed_s": round(time.time() - t0, 1),
          "seed": args.seed, "ood": args.ood, "recorded_dir": str(out_dir) if args.record else None, "per_attempt": per_attempt}
report_path.write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items() if k != "per_attempt"}, indent=2))
for w in writers:
    if w is not None:
        w.discard()
env.close()
app.close()
