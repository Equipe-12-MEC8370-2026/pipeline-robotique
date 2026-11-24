"""P0.5: closed-loop evaluation in CTA-SO101-PickPlace-v0 (run with isaacsim-env, PYTHONPATH=<repo>\\src).

Policy (start scripts\\eval\\policy_server.py in .venv-learn first):
    python scripts\\sim\\eval_policy.py --headless --enable_cameras --agent policy --tag act_p0
Scripted expert on the SAME initial conditions (paired reference):
    python scripts\\sim\\eval_policy.py --headless --enable_cameras --agent expert --tag expert
Smoke test: add --max_trials 2

Trials come from cta_learn.eval.metrics.make_trials(n_train, n_ood, seed): cube/target poses are forced
per trial (degenerate reset ranges) and torch is re-seeded before each reset, so light, table colour and
camera jitter are identical between agents. Output: outputs/p0_5/<tag>/report.json + videos/.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--agent", choices=["policy", "expert"], required=True)
parser.add_argument("--tag", type=str, default=None, help="run name (default: agent)")
parser.add_argument("--n_train", type=int, default=40)
parser.add_argument("--n_ood", type=int, default=10)
parser.add_argument("--seed", type=int, default=7, help="trial seed (demos were generated with seed 1)")
parser.add_argument("--max_trials", type=int, default=0, help="run only the first N trials (smoke test)")
parser.add_argument("--trials", type=int, nargs="+", default=None, help="run only these trial ids (e.g. live demo)")
parser.add_argument("--port", type=int, default=50765)
parser.add_argument("--keep_success_videos", type=int, default=5, help="all failures are kept; plus N successes")
parser.add_argument("--shutdown_server", action="store_true", help="stop the policy server at the end")
parser.add_argument("--out", type=str, default="outputs/p0_5")
parser.add_argument("--variant", choices=["p0", "real"], default="p0",
                    help="p0 = CTA-SO101-PickPlace-v0; real = CTA-SO101-PickPlaceReal-v0 (P4 look-alike scene)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import json  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import cta_learn.sim.tasks  # noqa: E402,F401
from cta_learn import REPO_ROOT  # noqa: E402
from cta_learn.core import load_embodiment  # noqa: E402
from cta_learn.datagen.episode_io import _VideoWriter  # noqa: E402
from cta_learn.datagen.pick_place_expert import ExpertParams, PickPlaceExpert, yaw_from_quat_xyzw  # noqa: E402
from cta_learn.eval import metrics as M  # noqa: E402
from cta_learn.eval.protocol import PolicyClient  # noqa: E402
from cta_learn.sim.ik import TcpIK  # noqa: E402
from cta_learn.sim.tasks.pick_place import (  # noqa: E402
    ARM_JOINTS,
    CAM_H,
    CAM_W,
    CUBE_NOMINAL,
    CUBE_RANGE,
    TARGET_RANGE,
    TASK_ID,
    TCP_OFFSET,
    PickPlaceEnvCfg,
)

DT = 1 / 30
GRIPPER_SPEED = 3.0  # rad/s, same rate limit as the demo generator

# the eval's notion of "training range" must be the scene's
for k, v in CUBE_RANGE.items():
    assert np.allclose(v, M.TRAIN_CUBE[k]), f"metrics.TRAIN_CUBE[{k}] != scene CUBE_RANGE"
for k, v in TARGET_RANGE.items():
    assert np.allclose(v, M.TRAIN_TARGET[k]), f"metrics.TRAIN_TARGET[{k}] != scene TARGET_RANGE"

tag = args.tag or args.agent
out_dir = REPO_ROOT / args.out / tag
if out_dir.exists():
    raise SystemExit(f"{out_dir} exists: pick another --tag")
(out_dir / "videos").mkdir(parents=True)

emb = load_embodiment("so101")
if args.variant == "real":
    from cta_learn.sim.tasks.pick_place_real import TASK_ID  # noqa: F811
    from cta_learn.sim.tasks.pick_place_real import PickPlaceRealEnvCfg as PickPlaceEnvCfg  # noqa: F811
cfg = PickPlaceEnvCfg()
cfg.scene.num_envs = 1
cfg.seed = args.seed
env = gym.make(TASK_ID, cfg=cfg).unwrapped
robot = env.scene["robot"]
origins = env.scene.env_origins
dev = env.device


class ExpertAgent:
    """The P0.3 scripted expert (privileged state + IK), as in scripts/sim/gen_demos.py, for one env."""

    name = "expert"

    def __init__(self):
        self.ik = TcpIK(robot, ARM_JOINTS, TCP_OFFSET, dt=DT)
        self.expert = PickPlaceExpert()
        self.grip = torch.zeros(1, device=dev)

    def info(self):
        return {"agent": "expert"}

    def reset(self, obs, trial):
        e, q = obs["expert"], obs["policy"]["joint_pos"]
        self.expert.reset(e["tcp_pos"][0].cpu().numpy(), e["cube_pos"][0].cpu().numpy(),
                          yaw_from_quat_xyzw(e["cube_quat"][0].cpu().numpy()),
                          ExpertParams.sample(np.random.default_rng([args.seed, trial["trial"]])),
                          arm_q=q[0, :5].cpu().numpy())
        self.grip[0] = q[0, 5]
        self.ik.set_command([0], q[0, :5].clone())

    def act(self, obs):
        e, q = obs["expert"], obs["policy"]["joint_pos"]
        cmd = self.expert.step(e["tcp_pos"][0].cpu().numpy(), e["cube_pos"][0].cpu().numpy(),
                               e["target_pos"][0].cpu().numpy(), DT, arm_q=q[0, :5].cpu().numpy(),
                               cube_yaw=yaw_from_quat_xyzw(e["cube_quat"][0].cpu().numpy()))
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

    def __init__(self):
        self.client = PolicyClient(("127.0.0.1", args.port))
        self.lower, self.upper = emb.lower, emb.upper
        self.failed = None

    def info(self):
        return {"agent": "policy", **self.client.info()}

    def reset(self, obs, trial):
        self.client.reset()

    def act(self, obs):
        p = obs["policy"]
        a = self.client.act({
            "observation.state": p["joint_pos"][0].cpu().numpy().astype(np.float32),
            "observation.images.front": _uint8(p["front"][0]),
            "observation.images.wrist": _uint8(p["wrist"][0]),
        })
        a, clipped = M.clip_to_limits(a, self.lower, self.upper)
        return torch.as_tensor(a, device=dev).unsqueeze(0), clipped


def set_trial(trial) -> dict:
    """Force this trial's cube/target offsets, re-seed the other randomizations, reset."""
    c, t = trial["cube"], trial["target"]
    em = env.event_manager
    cube_cfg = em.get_term_cfg("cube")
    cube_cfg.params["pose_range"] = {"x": (c["x"], c["x"]), "y": (c["y"], c["y"]), "yaw": (c["yaw"], c["yaw"])}
    em.set_term_cfg("cube", cube_cfg)
    tgt_cfg = em.get_term_cfg("target")
    tgt_cfg.params["pose_range"] = {"x": (t["x"], t["x"]), "y": (t["y"], t["y"])}
    em.set_term_cfg("target", tgt_cfg)
    torch.manual_seed(args.seed * 1000 + trial["trial"])
    obs, _ = env.reset()
    return obs


agent = ExpertAgent() if args.agent == "expert" else PolicyAgent()
trials = M.make_trials(args.n_train, args.n_ood, args.seed)
if args.trials:
    unknown = set(args.trials) - {t["trial"] for t in trials}
    if unknown:
        raise SystemExit(f"--trials {sorted(unknown)} not in 0..{len(trials) - 1}")
    trials = [t for t in trials if t["trial"] in set(args.trials)]
if args.max_trials:
    trials = trials[: args.max_trials]
info = agent.info()
print(f"[eval] {tag}: {len(trials)} trials, {info}", flush=True)

records, kept_successes = [], 0
t_start = time.time()
for trial in trials:
    if not app.is_running():
        break
    obs = set_trial(trial)
    agent.reset(obs, trial)
    vid_path = out_dir / "videos" / f"trial_{trial['trial']:03d}.mp4"
    vid = _VideoWriter(vid_path, 2 * CAM_W, CAM_H, 30)
    cube_z, n_clip, steps, t0 = [], 0, 0, time.time()
    cube0 = obs["expert"]["cube_pos"][0].cpu().numpy()
    target_xy = obs["expert"]["target_pos"][0, :2].cpu().numpy()
    outcome = None
    while outcome is None:
        p = obs["policy"]
        vid.write(np.concatenate([_uint8(p["front"][0]), _uint8(p["wrist"][0])], axis=1))
        cube_z.append(float(obs["expert"]["cube_pos"][0, 2]))
        last_cube_xy = obs["expert"]["cube_pos"][0, :2].cpu().numpy()
        last_tcp_cube = float(torch.linalg.vector_norm(obs["expert"]["tcp_pos"][0] - obs["expert"]["cube_pos"][0]))
        action, clipped = agent.act(obs)
        n_clip += int(clipped)
        obs, _, terminated, truncated, _ = env.step(action)
        steps += 1
        if terminated[0] or truncated[0]:
            # Isaac Lab auto-resets on termination: the returned obs is already the next episode's, so the
            # cube's final state is the last pre-step observation (1 step = 33 ms earlier)
            succ = bool(env.termination_manager.get_term("success")[0])
            off = bool(env.termination_manager.get_term("cube_dropped")[0])
            cz = np.array(cube_z)
            outcome = M.classify_outcome(succ, off, cz, last_cube_xy, target_xy,
                                         holding_final=bool(cz[-1] - cz[0] > 0.01))
        elif agent.failed:
            outcome = f"expert_{agent.failed}"
    vid.close()
    rec = {**trial, "agent": tag, "outcome": outcome, "steps": steps, "sim_s": round(steps * DT, 2),
           "wall_s": round(time.time() - t0, 1), "clipped_steps": n_clip,
           "max_lift_m": round(float(max(cube_z) - cube0[2]), 4),
           "final_cube_z_m": round(cube_z[-1], 4), "final_tcp_cube_m": round(last_tcp_cube, 4),
           "final_cube_target_xy_m": round(float(np.linalg.norm(last_cube_xy - target_xy)), 4),
           "cube0_offset_err_mm": round(1000 * float(np.linalg.norm(
               cube0[:2] - (np.array(CUBE_NOMINAL) + [trial["cube"]["x"], trial["cube"]["y"]]))), 2)}
    if args.agent == "expert":
        rec["expert_phase"] = agent.expert.phase
        rec["expert_regrasps"] = agent.expert.regrasps
    records.append(rec)
    keep = outcome != "success" or kept_successes < args.keep_success_videos
    kept_successes += outcome == "success"
    if not keep:
        vid_path.unlink()
    print(f"[trial {trial['trial']:3d} {trial['split']:<12}] {outcome:<24} {steps:4d} steps "
          f"tcp-cube {rec['final_tcp_cube_m']:.3f} z {rec['final_cube_z_m']:.3f} pose err {rec['cube0_offset_err_mm']} mm "
          f"lift {rec['max_lift_m']:.3f} m clip {n_clip:3d}  ({rec['wall_s']} s)", flush=True)

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
