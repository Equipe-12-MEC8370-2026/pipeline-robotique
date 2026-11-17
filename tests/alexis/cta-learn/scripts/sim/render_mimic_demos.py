"""D12 trial: render Mimic-generated demos with cameras into the raw episode format (same as gen_profile_demos.py).

Mimic generates from the READY pose without cameras (scripts/sim/mimic_generate.py). Here, in the normal P3 scene
(folded start, front + wrist cameras): restore the demo's initial part poses, move folded -> READY in joint space
exactly like the scripted expert (StackExpert's ready phase, rate-limited, gripper opening), let the arm settle on
READY, then replay the generated joint-position actions open loop (holding the last one up to --extra_steps).
Only demos that still succeed (stack complete and at rest 1 s) are written.

    $env:PYTHONPATH = "$PWD\\src"; & ..\\isaacsim-env\\Scripts\\python.exe scripts\\sim\\render_mimic_demos.py --headless --enable_cameras `
        --input data\\mimic\\p3_generated.hdf5 --out data\\raw\\p3_mimic
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--max", type=int, default=0, help="render at most N demos (0 = all)")
parser.add_argument("--extra_steps", type=int, default=60)
parser.add_argument("--settle_tol_deg", type=float, default=0.5)
parser.add_argument("--report", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import cta_learn.sim.tasks  # noqa: E402,F401
from cta_learn import REPO_ROOT  # noqa: E402
from cta_learn.datagen.episode_io import EpisodeWriter, list_episodes  # noqa: E402
from cta_learn.datagen.pick_place_expert import READY_POSE_DEG  # noqa: E402
from cta_learn.datagen.stack_expert import PHASES, StackExpert, plan_from_layout  # noqa: E402
from cta_learn.sim import mdp  # noqa: E402
from cta_learn.sim.tasks.pick_place import CAM_H, CAM_W  # noqa: E402
from cta_learn.sim.tasks.profile_stack import LAYOUT, TASK_DESCRIPTION, TASK_ID, ProfileStackEnvCfg  # noqa: E402
from cta_learn.sim.tasks.profile_stack_mimic import GRIPPER_OPEN_DEG  # noqa: E402

DT = 1 / 30
GRIPPER_SPEED = 3.0
READY = torch.tensor([math.radians(v) for v in (*READY_POSE_DEG, GRIPPER_OPEN_DEG)])


def resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else REPO_ROOT / path


cfg = ProfileStackEnvCfg()
cfg.scene.num_envs = 1
success_cfg = cfg.terminations.success
cfg.terminations.success = None      # judged here
cfg.terminations.time_out = None     # demo length is set by the generated actions
env = gym.make(TASK_ID, cfg=cfg).unwrapped
dev = env.device
ids = torch.tensor([0], device=dev, dtype=torch.int32)
origin = env.scene.env_origins[0]
ready = READY.to(dev)
out_dir = resolve(args.out)
next_index = len(list_episodes(out_dir))
src = h5py.File(resolve(args.input), "r")
names = sorted(src["data"].keys(), key=lambda s: int(s.split("_")[1]))
if args.max:
    names = names[: args.max]
records, t0 = [], time.time()
for name in names:
    g = src["data"][name]
    actions = torch.tensor(np.array(g["actions"]), device=dev, dtype=torch.float32)
    obs, _ = env.reset()
    for part in ("profile_short", "profile_long"):
        pose = torch.tensor(np.array(g[f"initial_state/rigid_object/{part}/root_pose"]), device=dev, dtype=torch.float32)
        pose[:, :3] += origin
        env.scene[part].write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
        env.scene[part].write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=dev), env_ids=ids)
    mdp.break_rigid_contacts(env, ids, ("profile_short", "profile_long"))
    env.scene.write_data_to_sim()
    env.sim.forward()
    obs = env.observation_manager.compute()

    w = EpisodeWriter(out_dir, next_index, {"front": (CAM_W, CAM_H), "wrist": (CAM_W, CAM_H)})

    def record(o, a, phase: int) -> None:
        e = o["expert"]
        w.add({"front": o["policy"]["front"][0].cpu().numpy(), "wrist": o["policy"]["wrist"][0].cpu().numpy()},
              state=o["policy"]["joint_pos"][0].cpu().numpy(), action=a[0].cpu().numpy(),
              short_pos=e["short_pos"][0].cpu().numpy(), short_quat=e["short_quat"][0].cpu().numpy(),
              long_pos=e["long_pos"][0].cpu().numpy(), long_quat=e["long_quat"][0].cpu().numpy(),
              tcp_pos=e["tcp_pos"][0].cpu().numpy(), phase=np.int16(phase), step=np.int16(-1),
              cmd_yaw=np.float32(np.nan), ik_pos_err=np.float32(np.nan), ik_tilt=np.float32(np.nan))

    # 1) folded -> READY, as the scripted expert does (joint space, 0.5 rad/s, gripper opening at 3 rad/s)
    ex = StackExpert(plan_from_layout(LAYOUT))
    q = obs["policy"]["joint_pos"]
    ex.phase, ex._q_cmd = "ready", q[0, :5].cpu().numpy().astype(np.float64)   # only its ready phase is used
    grip = q[0, 5].clone()
    n_ready = 0
    while True:
        qn = obs["policy"]["joint_pos"][0]
        cmd = ex._step_ready(obs["expert"]["tcp_pos"][0].cpu().numpy(), qn[:5].cpu().numpy(), DT)
        grip = grip + torch.clamp(ready[5] - grip, -GRIPPER_SPEED * DT, GRIPPER_SPEED * DT)
        a = torch.cat([torch.as_tensor(cmd.joint_goal, device=dev, dtype=torch.float32), grip.reshape(1)]).unsqueeze(0)
        record(obs, a, PHASES.index("ready"))
        obs, *_ = env.step(a)
        n_ready += 1
        err = (obs["policy"]["joint_pos"][0] - ready).abs().max().item()
        if (ex.phase != "ready" and err < math.radians(args.settle_tol_deg)) or n_ready > 300:
            break
        if ex.phase != "ready":           # reached within 3 deg: hold READY until settled
            ex.phase = "ready"
            ex._q_cmd = ready[:5].cpu().numpy().astype(np.float64)
    # 2) the generated actions, open loop, then hold the last one
    ok, steps = False, 0
    seq = list(actions) + [actions[-1]] * args.extra_steps
    for a in seq:
        a = a.unsqueeze(0)
        record(obs, a, -1)
        obs, *_ = env.step(a)
        steps += 1
        if bool(success_cfg.func(env, **success_cfg.params)[0]):
            ok = True
            break
    meta = {"task": TASK_DESCRIPTION, "success": ok, "outcome": "success" if ok else "failed_on_render",
            "source": "isaaclab_mimic", "mimic_demo": name, "mimic_file": str(resolve(args.input)),
            "ready_steps": n_ready, "mimic_steps": steps, "layout": LAYOUT["task_id"]}
    if ok:
        w.keep(meta)
        next_index += 1
    else:
        w.discard()
    records.append({"demo": name, "success": ok, "ready_steps": n_ready, "mimic_steps": steps,
                    "generated_steps": int(len(actions))})
    print(f"[render {name}] {'success' if ok else 'FAIL'} ready {n_ready} + mimic {steps}/{len(actions)} steps", flush=True)

src.close()
summary = {"input": str(resolve(args.input)), "rendered": len(records), "successes": sum(r["success"] for r in records),
           "elapsed_s": round(time.time() - t0), "records": records}
print("RENDER_SUMMARY", json.dumps({k: v for k, v in summary.items() if k != "records"}))
if args.report:
    resolve(args.report).parent.mkdir(parents=True, exist_ok=True)
    resolve(args.report).write_text(json.dumps(summary, indent=2))
env.close()
app.close()
