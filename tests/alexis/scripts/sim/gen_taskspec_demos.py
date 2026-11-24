r"""Run the gesture library (datagen.gestures, chained from the layout) in CTA-SO101-TaskSpec-v0, for ANY task.

Generic version of gen_profile_demos.py. The task = the layout in CTA_LAYOUT (see sim/tasks/taskspec_task.py).
PowerShell, from the repo root:
    $env:PYTHONPATH = "$PWD\src"; $env:CTA_LAYOUT = "configs/tasks/so101_cta_angle_block.yaml"
Success rate (no recording):
    & ..\isaacsim-env\Scripts\python.exe scripts\sim\gen_taskspec_demos.py --headless --enable_cameras --attempts 50
Generation (keep successes only, raw format):
    ... --record --successes 500 --attempts 700 --seed 1 --out data/raw/<name>
Report: outputs/taskspec_demos/<task_id>/gate[_<tag>].json. num_envs=1 (see mdp.break_rigid_contacts).
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--attempts", type=int, default=20)
parser.add_argument("--successes", type=int, default=0, help="stop after this many successes (0 = run all attempts)")
parser.add_argument("--record", action="store_true")
parser.add_argument("--out", type=str, default=None, help="raw episodes dir (default data/raw/<task_id>)")
parser.add_argument("--debug_videos", type=int, default=0, help="also save the first N attempts (any outcome)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--ood", action="store_true", help="out-of-range start poses (use_ood_ranges)")
parser.add_argument("--tag", type=str, default="", help="suffix of the report name")
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
from cta_learn.datagen.gestures import plan_task  # noqa: E402
from cta_learn.datagen.pick_place_expert import yaw_from_quat_xyzw  # noqa: E402
from cta_learn.datagen.stack_expert import PHASES, StackExpert, StackParams  # noqa: E402
from cta_learn.retarget.layout import task_success  # noqa: E402
from cta_learn.sim.ik import TcpIK  # noqa: E402
from cta_learn.sim.tasks import pick_place as p0  # noqa: E402
from cta_learn.sim.tasks.pick_place import ARM_JOINTS, CAM_H, CAM_W  # noqa: E402
from cta_learn.sim.tasks.taskspec_task import TASK_ID, TaskSpecEnvCfg, layout_path  # noqa: E402

DT = 1 / 30
GRIPPER_SPEED = 3.0  # rad/s

cfg = TaskSpecEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.seed = args.seed
if args.ood:
    cfg.use_ood_ranges()
LAYOUT = cfg.layout
NAMES = list(LAYOUT["order"])
TCP_OFFSET = (p0.FIXED_FINGER_INNER_X + 0.005 + LAYOUT["grasp_width_m"] / 2, 0.0, p0.FIXED_FINGER_TIP_Z + 0.010)


def resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else REPO_ROOT / path


out_dir = resolve(args.out or f"data/raw/{LAYOUT['task_id']}")
report_path = REPO_ROOT / "outputs" / "taskspec_demos" / LAYOUT["task_id"] / f"gate{'_' + args.tag if args.tag else ''}.json"
debug_dir = report_path.parent / "debug_episodes"
report_path.parent.mkdir(parents=True, exist_ok=True)

env = gym.make(TASK_ID, cfg=cfg).unwrapped
robot = env.scene["robot"]
ik = TcpIK(robot, ARM_JOINTS, TCP_OFFSET, dt=DT)
rng = np.random.default_rng(args.seed)
n, origins = env.num_envs, env.scene.env_origins
plans = plan_task(LAYOUT)
print(f"[gestures] {layout_path().name}: " + "; ".join(f"{p.obj} -> {p.support or 'table'}" for p in plans), flush=True)

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
after_done: list[dict | None] = [None] * n


def up_axis(q) -> np.ndarray:
    """World direction of the part's local z axis, quaternion (x, y, z, w)."""
    x, y, z, w = (float(v) for v in q)
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


def parts_of(e, i) -> dict:
    out = {}
    for k in NAMES:
        q = e[f"{k}_quat"][i].cpu().numpy()
        out[k] = (e[f"{k}_pos"][i].cpu().numpy(), yaw_from_quat_xyzw(q), up_axis(q))
    return out


def diagnose(e, i) -> dict:
    """Which success checks fail (state 1.5 s after the expert finished)."""
    pos = {k: e[f"{k}_pos"][i:i + 1] for k in NAMES}
    quat = {k: e[f"{k}_quat"][i:i + 1] for k in NAMES}
    zone = {k: e[f"{k}_goal_xy"][i:i + 1] for k in NAMES if f"{k}_goal_xy" in e}
    _, checks = task_success(pos, quat, e["tcp_pos"][i:i + 1], zone, LAYOUT, torch)
    return {"failed_checks": [k for k, v in checks.items() if not bool(v[0])],
            "positions_mm": {k: [round(float(v) * 1000, 1) for v in pos[k][0]] for k in NAMES}}


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
    start_pose[i] = {k: [round(float(v), 4) for v in (*pos[:2], yaw)] for k, (pos, yaw, _) in parts_of(e, i).items()}
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
                        "after_done": after_done[i], "start_xyyaw": start_pose[i]})
    if after_done[i] and outcome != "success":
        print(f"    why no success 1.5 s after the expert finished: {after_done[i]}", flush=True)
    after_done[i], done_steps[i] = None, 0
    w = writers[i]
    if w is not None:
        meta = {"task": LAYOUT["instruction"]["en"], "success": outcome == "success", "outcome": outcome,
                "attempt": attempt_of[i], "seed": args.seed, "ood": args.ood, "expert_params": asdict(ex.params),
                "expert_failure": ex.failure_reason, "regrasps": ex.regrasps, "taskspec": LAYOUT["from_taskspec"],
                "layout": LAYOUT["task_id"], "gestures": ["pick_place_box"] * len(plans)}
        if outcome == "success" or not args.record:
            w.keep(meta)
        else:
            w.discard()
    writers[i] = None
    print(f"[attempt {attempt_of[i]:4d}] {outcome:<42} steps={steps:4d} successes={successes}/{attempts}", flush=True)


obs, _ = env.reset()
for i in range(n):
    start_episode(i, obs)
steps_in_ep = [0] * n
done_steps = [0] * n
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
    arm_q = ik.compute(goals, yaws, tilts)
    for i, qg in joint_goals.items():
        arm_q[i] = qg
        ik.set_command([i], qg)
    action = torch.cat([arm_q, grip_cmd.unsqueeze(1)], dim=1)
    for i in range(n):
        if writers[i] is not None:
            extras = {f"{k}_{f}": e[f"{k}_{f}"][i].cpu().numpy() for k in NAMES for f in ("pos", "quat")}
            writers[i].add(
                {"front": obs["policy"]["front"][i].cpu().numpy(), "wrist": obs["policy"]["wrist"][i].cpu().numpy()},
                state=obs["policy"]["joint_pos"][i].cpu().numpy(), action=action[i].cpu().numpy(),
                tcp_pos=e["tcp_pos"][i].cpu().numpy(), phase=np.int16(phase_idx[i]), step=np.int16(step_idx[i]),
                cmd_yaw=np.float32(yaws[i].item()), ik_pos_err=np.float32(ik.last_pos_err[i].item()),
                ik_tilt=np.float32(ik.last_tilt[i].item()), **extras)
    obs, _, terminated, truncated, _ = env.step(action)
    tm = env.termination_manager
    succ = tm.get_term("success")
    dropped = {k: tm.get_term(f"{k}_dropped") for k in NAMES}
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
            elif any(bool(d[i]) for d in dropped.values()):
                outcome = f"{next(k for k, d in dropped.items() if d[i])}_dropped ({where})"
            else:
                outcome = f"timeout ({where})"
        elif ex.failed:
            outcome = f"expert: {ex.failure_reason}"
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
report = {"layout": str(layout_path()), "task_id": LAYOUT["task_id"], "attempts": done, "successes": successes,
          "success_rate": successes / max(done, 1), "outcomes": dict(outcomes),
          "mean_episode_steps": float(np.mean(lengths)) if lengths else 0.0, "elapsed_s": round(time.time() - t0, 1),
          "seed": args.seed, "ood": args.ood, "recorded_dir": str(out_dir) if args.record else None, "per_attempt": per_attempt}
report_path.write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items() if k != "per_attempt"}, indent=2))
for w in writers:
    if w is not None:
        w.discard()
env.close()
app.close()
