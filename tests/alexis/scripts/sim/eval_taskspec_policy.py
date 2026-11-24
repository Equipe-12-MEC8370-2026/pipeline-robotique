r"""Closed-loop evaluation of a trained policy (or the gesture library) in CTA-SO101-TaskSpec-v0, for ANY task.

Generic version of eval_profile_policy.py (same protocol: 40 trials in the training ranges + 10 outside, seed 7,
paired with the scripted agent on identical start conditions, light, table colour and camera jitter).
PowerShell, from the repo root, with $env:PYTHONPATH = "$PWD\src" and $env:CTA_LAYOUT = "configs/tasks/so101_<task>.yaml":
Policy (start scripts\eval\policy_server.py --ckpt <checkpoint> in .venv-learn first):
    & ..\isaacsim-env\Scripts\python.exe scripts\sim\eval_taskspec_policy.py --headless --enable_cameras --agent policy --tag act
Gesture library on the same trials (paired reference):
    ... --agent expert --tag expert
Output: outputs/taskspec_eval/<task_id>/<tag>/report.json + videos/ (front | wrist side by side).
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--agent", choices=["policy", "expert"], required=True)
parser.add_argument("--tag", type=str, default=None)
parser.add_argument("--n_train", type=int, default=40)
parser.add_argument("--n_ood", type=int, default=10)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--max_trials", type=int, default=0)
parser.add_argument("--port", type=int, default=50765)
parser.add_argument("--keep_success_videos", type=int, default=5)
parser.add_argument("--shutdown_server", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import json  # noqa: E402
import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import cta_learn.sim.tasks  # noqa: E402,F401
from cta_learn import REPO_ROOT  # noqa: E402
from cta_learn.core import load_embodiment  # noqa: E402
from cta_learn.datagen.episode_io import _VideoWriter  # noqa: E402
from cta_learn.datagen.gestures import plan_task  # noqa: E402
from cta_learn.datagen.pick_place_expert import yaw_from_quat_xyzw  # noqa: E402
from cta_learn.datagen.stack_expert import StackExpert, StackParams  # noqa: E402
from cta_learn.eval import metrics as M  # noqa: E402
from cta_learn.eval.protocol import PolicyClient  # noqa: E402
from cta_learn.eval.stack_metrics import LIFT_THRESHOLD, classify_task_outcome, make_stack_trials  # noqa: E402
from cta_learn.retarget.layout import task_success  # noqa: E402
from cta_learn.sim.ik import TcpIK  # noqa: E402
from cta_learn.sim.tasks import pick_place as p0  # noqa: E402
from cta_learn.sim.tasks.pick_place import ARM_JOINTS, CAM_H, CAM_W  # noqa: E402
from cta_learn.sim.tasks.taskspec_task import TASK_ID, TaskSpecEnvCfg  # noqa: E402

DT = 1 / 30
GRIPPER_SPEED = 3.0

emb = load_embodiment("so101")
cfg = TaskSpecEnvCfg()
cfg.scene.num_envs = 1
cfg.seed = args.seed
LAYOUT = cfg.layout
NAMES = list(LAYOUT["order"])
TCP_OFFSET = (p0.FIXED_FINGER_INNER_X + 0.005 + LAYOUT["grasp_width_m"] / 2, 0.0, p0.FIXED_FINGER_TIP_Z + 0.010)
tag = args.tag or args.agent
base = REPO_ROOT / "outputs" / "taskspec_eval" / LAYOUT["task_id"]
out_dir = base / tag
if out_dir.exists():
    raise SystemExit(f"{out_dir} exists: pick another --tag")
(out_dir / "videos").mkdir(parents=True)
env = gym.make(TASK_ID, cfg=cfg).unwrapped
robot = env.scene["robot"]
origins = env.scene.env_origins
dev = env.device


def up_axis(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


def parts_of(e) -> dict:
    out = {}
    for k in NAMES:
        q = e[f"{k}_quat"][0].cpu().numpy()
        out[k] = (e[f"{k}_pos"][0].cpu().numpy(), yaw_from_quat_xyzw(q), up_axis(q))
    return out


def _uint8(img: torch.Tensor) -> np.ndarray:
    a = img.cpu()
    if a.dtype != torch.uint8:
        a = a.float().clamp(0, 255).round().to(torch.uint8)
    return a.numpy()


class ExpertAgent:
    def __init__(self):
        self.ik = TcpIK(robot, ARM_JOINTS, TCP_OFFSET, dt=DT)
        self.expert = StackExpert(plan_task(LAYOUT))
        self.grip = torch.zeros(1, device=dev)

    def info(self):
        return {"agent": "expert (gesture library)"}

    def reset(self, obs, trial):
        e, q = obs["expert"], obs["policy"]["joint_pos"]
        self.expert.reset(e["tcp_pos"][0].cpu().numpy(), parts_of(e),
                          StackParams.sample(np.random.default_rng([args.seed, trial["trial"]])), arm_q=q[0, :5].cpu().numpy())
        self.grip[0] = q[0, 5]
        self.ik.set_command([0], q[0, :5].clone())

    def act(self, obs):
        e, q = obs["expert"], obs["policy"]["joint_pos"]
        cmd = self.expert.step(e["tcp_pos"][0].cpu().numpy(), parts_of(e), DT, arm_q=q[0, :5].cpu().numpy())
        goal = torch.as_tensor(cmd.tcp_goal, device=dev, dtype=torch.float32).unsqueeze(0) + origins
        arm_q = self.ik.compute(goal, torch.tensor([cmd.yaw], device=dev), torch.tensor([cmd.tilt_weight], device=dev))
        if cmd.joint_goal is not None:
            qg = torch.as_tensor(cmd.joint_goal, device=dev, dtype=torch.float32)
            arm_q[0] = qg
            self.ik.set_command([0], qg)
        g = torch.tensor(cmd.gripper, device=dev)
        self.grip[0] = self.grip[0] + torch.clamp(g - self.grip[0], -GRIPPER_SPEED * DT, GRIPPER_SPEED * DT)
        return torch.cat([arm_q, self.grip.unsqueeze(1)], dim=1), False

    @property
    def failed(self):
        return self.expert.failure_reason if self.expert.failed else None


class PolicyAgent:
    """A remote policy (cta_learn.eval.protocol) seeing only joint positions + the two cameras."""

    failed = None

    def __init__(self):
        self.client = PolicyClient(("127.0.0.1", args.port))
        self.lower, self.upper = emb.lower, emb.upper

    def info(self):
        return {"agent": "policy", **self.client.info()}

    def reset(self, obs, trial):
        self.client.reset()

    def act(self, obs):
        p = obs["policy"]
        a = self.client.act({"observation.state": p["joint_pos"][0].cpu().numpy().astype(np.float32),
                             "observation.images.front": _uint8(p["front"][0]),
                             "observation.images.wrist": _uint8(p["wrist"][0])})
        a, clipped = M.clip_to_limits(a, self.lower, self.upper)
        return torch.as_tensor(a, device=dev).unsqueeze(0), clipped


def set_trial(trial) -> dict:
    em = env.event_manager
    for k in NAMES:
        o = trial["offsets"][k]
        c = em.get_term_cfg(f"place_{k}")
        c.params["pose_range"] = {a: (o[a], o[a]) for a in ("x", "y", "yaw")}
        em.set_term_cfg(f"place_{k}", c)
    torch.manual_seed(args.seed * 1000 + trial["trial"])
    obs, _ = env.reset()
    return obs


agent = ExpertAgent() if args.agent == "expert" else PolicyAgent()
trials = make_stack_trials(LAYOUT, args.n_train, args.n_ood, args.seed, parts=NAMES)
if args.max_trials:
    trials = trials[: args.max_trials]
info = agent.info()
print(f"[eval] {LAYOUT['task_id']} / {tag}: {len(trials)} trials, {info}", flush=True)
zone = {k: torch.tensor([g["xyyaw"][:2]], device=dev) for k, g in LAYOUT["goals"].items() if g["relation"] == "on_table"}
records, kept_successes, t_start = [], 0, time.time()
for n_done, trial in enumerate(trials):
    if not app.is_running():
        break
    obs = set_trial(trial)
    agent.reset(obs, trial)
    vid_path = out_dir / "videos" / f"trial_{trial['trial']:03d}.mp4"
    vid = _VideoWriter(vid_path, 2 * CAM_W, CAM_H, 30)
    z0 = {k: float(obs["expert"][f"{k}_pos"][0, 2]) for k in NAMES}
    max_rise = {k: 0.0 for k in NAMES}
    n_clip, steps, t0, outcome, last = 0, 0, time.time(), None, None
    while outcome is None:
        e, p_obs = obs["expert"], obs["policy"]
        vid.write(np.concatenate([_uint8(p_obs["front"][0]), _uint8(p_obs["wrist"][0])], axis=1))
        for k in NAMES:
            max_rise[k] = max(max_rise[k], float(e[f"{k}_pos"][0, 2]) - z0[k])
        last = {k: v[0:1].clone() for k, v in e.items()}
        action, clipped = agent.act(obs)
        n_clip += int(clipped)
        obs, _, terminated, truncated, _ = env.step(action)
        steps += 1
        if terminated[0] or truncated[0] or agent.failed:
            tm = env.termination_manager
            succ = bool(tm.get_term("success")[0]) if (terminated[0] or truncated[0]) else False
            dropped = {k: bool(tm.get_term(f"{k}_dropped")[0]) for k in NAMES}
            _, checks = task_success({k: last[f"{k}_pos"] for k in NAMES}, {k: last[f"{k}_quat"] for k in NAMES},
                                     last["tcp_pos"], zone, LAYOUT, torch)
            checks = {k: bool(v[0]) for k, v in checks.items()}
            lifted = {k: max_rise[k] > LIFT_THRESHOLD for k in NAMES}
            outcome = classify_task_outcome(succ, NAMES, dropped, lifted, checks, None if succ else agent.failed)
    vid.close()
    rec = {**trial, "agent": tag, "outcome": outcome, "steps": steps, "sim_s": round(steps * DT, 2),
           "wall_s": round(time.time() - t0, 1), "clipped_steps": n_clip, "max_rise_m": {k: round(v, 4) for k, v in max_rise.items()}}
    records.append(rec)
    keep = outcome != "success" or kept_successes < args.keep_success_videos
    kept_successes += outcome == "success"
    if not keep:
        vid_path.unlink()
    n_succ = sum(r["outcome"] == "success" for r in records)
    print(f"[trial {trial['trial']:3d} {trial['split']:<12}] {outcome:<28} {steps:4d} steps ({rec['wall_s']} s) "
          f"successes={n_succ}/{n_done + 1} of {len(trials)}", flush=True)

summary = M.summarize(records)
report = {"task_id": LAYOUT["task_id"], "tag": tag, "info": info, "seed": args.seed, "n_train": args.n_train,
          "n_ood": args.n_ood, "elapsed_s": round(time.time() - t_start, 1), "summary": summary, "records": records}
expert_report = base / "expert" / "report.json"
if args.agent == "policy" and expert_report.exists():
    exp = json.loads(expert_report.read_text())
    if exp["seed"] == args.seed:
        report["paired_with_expert"] = M.paired_table(records, exp["records"])
(out_dir / "report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(summary, indent=2))
if isinstance(agent, PolicyAgent):
    agent.client.close(shutdown=args.shutdown_server)
env.close()
app.close()
