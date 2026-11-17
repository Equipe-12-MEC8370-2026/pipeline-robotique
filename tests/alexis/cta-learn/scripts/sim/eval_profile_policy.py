"""P3.5: closed-loop evaluation in CTA-SO101-ProfileStack-v0 (run with isaacsim-env, PYTHONPATH=<repo>\\src).

Policy (start scripts\\eval\\policy_server.py --ckpt <P3 checkpoint> in .venv-learn first):
    python scripts\\sim\\eval_profile_policy.py --headless --enable_cameras --agent policy --tag act_p3
Scripted expert on the SAME start conditions (paired reference):
    python scripts\\sim\\eval_profile_policy.py --headless --enable_cameras --agent expert --tag expert
Smoke test: add --max_trials 2

Trials: cta_learn.eval.stack_metrics.make_stack_trials (seed 7: 40 in the training ranges, 10 outside; the demos
used seed 1). Each part's reset range is collapsed to the trial's offsets and torch is re-seeded before each
reset, so light, table colour and camera jitter are identical between agents. Episode limit 40 s; success =
the scene's term (stack complete and at rest for 1 s). Output: outputs/p3_5/<tag>/report.json + videos/.
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
parser.add_argument("--trials", type=int, nargs="+", default=None)
parser.add_argument("--port", type=int, default=50765)
parser.add_argument("--keep_success_videos", type=int, default=5)
parser.add_argument("--shutdown_server", action="store_true")
parser.add_argument("--out", type=str, default="outputs/p3_5")
parser.add_argument("--save_states", action="store_true", help="also save per-step part / TCP / joint states (npz next to each kept video)")
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
from cta_learn.datagen.pick_place_expert import yaw_from_quat_xyzw  # noqa: E402
from cta_learn.datagen.stack_expert import StackExpert, StackParams, plan_from_layout  # noqa: E402
from cta_learn.eval import metrics as M  # noqa: E402
from cta_learn.eval.protocol import PolicyClient  # noqa: E402
from cta_learn.eval.stack_metrics import LIFT_THRESHOLD, PARTS, classify_stack_outcome, make_stack_trials  # noqa: E402
from cta_learn.retarget.profile_stack import stack_success  # noqa: E402
from cta_learn.sim.ik import TcpIK  # noqa: E402
from cta_learn.sim.tasks.pick_place import ARM_JOINTS, CAM_H, CAM_W  # noqa: E402
from cta_learn.sim.tasks.profile_stack import GOAL_SHORT_XY, LAYOUT, TASK_ID, TCP_OFFSET, ProfileStackEnvCfg  # noqa: E402

DT = 1 / 30
GRIPPER_SPEED = 3.0
KEY = {"profile_short": "short", "profile_long": "long"}

tag = args.tag or args.agent
out_dir = REPO_ROOT / args.out / tag
if out_dir.exists():
    raise SystemExit(f"{out_dir} exists: pick another --tag")
(out_dir / "videos").mkdir(parents=True)

emb = load_embodiment("so101")
cfg = ProfileStackEnvCfg()
cfg.scene.num_envs = 1
cfg.seed = args.seed
env = gym.make(TASK_ID, cfg=cfg).unwrapped
robot = env.scene["robot"]
origins = env.scene.env_origins
dev = env.device
half_h = {p: LAYOUT["objects"][p]["size_m"][2] / 2 for p in PARTS}


def up_axis(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


def parts_of(e) -> dict:
    out = {}
    for p, k in KEY.items():
        q = e[f"{k}_quat"][0].cpu().numpy()
        out[p] = (e[f"{k}_pos"][0].cpu().numpy(), yaw_from_quat_xyzw(q), up_axis(q))
    return out


class ExpertAgent:
    name = "expert"

    def __init__(self):
        self.ik = TcpIK(robot, ARM_JOINTS, TCP_OFFSET, dt=DT)
        self.expert = StackExpert(plan_from_layout(LAYOUT))
        self.grip = torch.zeros(1, device=dev)

    def info(self):
        return {"agent": "expert"}

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


def _uint8(img: torch.Tensor) -> np.ndarray:
    a = img.cpu()
    if a.dtype != torch.uint8:
        a = a.float().clamp(0, 255).round().to(torch.uint8)
    return a.numpy()


class PolicyAgent:
    """A remote policy (cta_learn.eval.protocol) seeing only joint positions + the two cameras."""

    name = "policy"
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
    for p in PARTS:
        o = trial["offsets"][p]
        c = em.get_term_cfg(p)
        c.params["pose_range"] = {k: (o[k], o[k]) for k in ("x", "y", "yaw")}
        em.set_term_cfg(p, c)
    torch.manual_seed(args.seed * 1000 + trial["trial"])
    obs, _ = env.reset()
    return obs


agent = ExpertAgent() if args.agent == "expert" else PolicyAgent()
trials = make_stack_trials(LAYOUT, args.n_train, args.n_ood, args.seed)
if args.trials:
    trials = [t for t in trials if t["trial"] in set(args.trials)]
if args.max_trials:
    trials = trials[: args.max_trials]
info = agent.info()
print(f"[eval] {tag}: {len(trials)} trials, {info}", flush=True)
goal = torch.tensor([GOAL_SHORT_XY], device=dev)
records, kept_successes, t_start = [], 0, time.time()
for trial in trials:
    if not app.is_running():
        break
    obs = set_trial(trial)
    agent.reset(obs, trial)
    vid_path = out_dir / "videos" / f"trial_{trial['trial']:03d}.mp4"
    vid = _VideoWriter(vid_path, 2 * CAM_W, CAM_H, 30)
    e0 = obs["expert"]
    start_err = {p: round(1000 * float(np.linalg.norm(e0[f"{KEY[p]}_pos"][0, :2].cpu().numpy()
                                                       - (np.array(LAYOUT["objects"][p]["start_xyyaw"][:2])
                                                          + [trial["offsets"][p]["x"], trial["offsets"][p]["y"]]))), 2)
                 for p in PARTS}
    z0 = {p: float(e0[f"{KEY[p]}_pos"][0, 2]) for p in PARTS}
    max_rise = {p: 0.0 for p in PARTS}
    n_clip, steps, t0, outcome, last = 0, 0, time.time(), None, None
    states = {k: [] for k in ("short_pos", "short_quat", "long_pos", "long_quat", "tcp_pos", "joint_pos")}
    while outcome is None:
        e, p_obs = obs["expert"], obs["policy"]
        vid.write(np.concatenate([_uint8(p_obs["front"][0]), _uint8(p_obs["wrist"][0])], axis=1))
        for p in PARTS:
            max_rise[p] = max(max_rise[p], float(e[f"{KEY[p]}_pos"][0, 2]) - z0[p])
        last = {k: v[0:1].clone() for k, v in e.items()}
        if args.save_states:
            for k in states:
                states[k].append((e[k][0] if k in e else p_obs[k][0]).cpu().numpy())
        action, clipped = agent.act(obs)
        n_clip += int(clipped)
        obs, _, terminated, truncated, _ = env.step(action)
        steps += 1
        if terminated[0] or truncated[0]:
            # auto-reset on termination: judge the last pre-step observation (33 ms earlier)
            tm = env.termination_manager
            succ = bool(tm.get_term("success")[0])
            dropped = {"profile_short": bool(tm.get_term("short_dropped")[0]), "profile_long": bool(tm.get_term("long_dropped")[0])}
            _, checks = stack_success(last["short_pos"], last["short_quat"], last["long_pos"], last["long_quat"],
                                      last["tcp_pos"], goal, LAYOUT, torch)
            checks = {k: bool(v[0]) for k, v in checks.items()}
            lifted = {p: max_rise[p] > LIFT_THRESHOLD for p in PARTS}
            holding = float(last["long_pos"][0, 2]) > LAYOUT["objects"]["profile_short"]["section_m"] + half_h["profile_long"] + 0.01
            outcome = classify_stack_outcome(succ, dropped, lifted, checks, holding)
        elif agent.failed:
            outcome = f"expert_{agent.failed}"
    vid.close()
    rec = {**trial, "agent": tag, "outcome": outcome, "steps": steps, "sim_s": round(steps * DT, 2),
           "wall_s": round(time.time() - t0, 1), "clipped_steps": n_clip,
           "max_rise_m": {p: round(v, 4) for p, v in max_rise.items()}, "start_pose_err_mm": start_err}
    if args.agent == "expert":
        rec["expert_step_phase"] = [agent.expert.step_index, agent.expert.phase]
    records.append(rec)
    keep = outcome != "success" or kept_successes < args.keep_success_videos
    kept_successes += outcome == "success"
    if not keep:
        vid_path.unlink()
    elif args.save_states:
        np.savez_compressed(vid_path.with_suffix(".npz"), **{k: np.stack(v) for k, v in states.items()})
    print(f"[trial {trial['trial']:3d} {trial['split']:<12}] {outcome:<22} {steps:4d} steps rise short "
          f"{max_rise['profile_short']:.3f} long {max_rise['profile_long']:.3f} start err {start_err} mm clip {n_clip:3d} "
          f"({rec['wall_s']} s)", flush=True)

summary = M.summarize(records)
report = {"tag": tag, "info": info, "seed": args.seed, "n_train": args.n_train, "n_ood": args.n_ood,
          "elapsed_s": round(time.time() - t_start, 1), "summary": summary, "records": records}
expert_report = REPO_ROOT / args.out / "expert" / "report.json"
if args.agent == "policy" and expert_report.exists():
    exp = json.loads(expert_report.read_text())
    if exp["seed"] == args.seed:
        report["paired_with_expert"] = M.paired_table(records, exp["records"])
(out_dir / "report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(summary, indent=2))
if report.get("paired_with_expert"):
    print("paired:", report["paired_with_expert"])
if isinstance(agent, PolicyAgent):
    agent.client.close(shutdown=args.shutdown_server)
env.close()
app.close()
