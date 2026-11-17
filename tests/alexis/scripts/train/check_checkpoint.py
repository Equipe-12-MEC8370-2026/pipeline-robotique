"""P0.4 step 4: check a trained checkpoint offline (ACT, SmolVLA, ...; run with .venv-learn).

    .\\.venv-learn\\Scripts\\python.exe scripts\\train\\check_checkpoint.py
    .\\.venv-learn\\Scripts\\python.exe scripts\\train\\check_checkpoint.py --ckpt outputs\\train\\act_p0\\checkpoints\\050000\\pretrained_model

Checks: the checkpoint (policy + pre/post-processors) reloads; on real dataset observations the
predicted action chunk is finite, in radians within the SO-101 joint limits, and close to the
expert's chunk (open-loop error, degrees). These are TRAINING episodes: this validates the
pipeline, not generalisation — that is P0.5 (closed loop in sim, unseen positions).
"""

import argparse
import json
import sys
import time

import numpy as np
import torch

from cta_learn import REPO_ROOT
from cta_learn.core import UnitRangeError, load_embodiment, validate

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="outputs/train/act_p0/checkpoints/last/pretrained_model")
p.add_argument("--root", default="data/datasets/p0_pickplace")
p.add_argument("--episodes", type=int, nargs="*", default=[0, 37, 75, 112, 149])
p.add_argument("--frames", type=int, nargs="*", default=[0, 60, 120, 180, 240])
p.add_argument("--max_mean_err_deg", type=float, default=5.0)
p.add_argument("--out", default="outputs/p0_4/checkpoint_check.json")
a = p.parse_args()

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies import make_pre_post_processors  # noqa: E402
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.policies.factory import get_policy_class  # noqa: E402

ckpt = REPO_ROOT / a.ckpt
emb = load_embodiment("so101")
lo, hi = emb.lower, emb.upper
checks = {}


def check(name, ok, detail):
    checks[name] = {"ok": bool(ok), "detail": detail}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


device = "cuda" if torch.cuda.is_available() else "cpu"
ptype = PreTrainedConfig.from_pretrained(ckpt).type  # act, smolvla, ...
policy = get_policy_class(ptype).from_pretrained(ckpt).to(device).eval()
pre, post = make_pre_post_processors(policy.config, pretrained_path=str(ckpt),
                                     preprocessor_overrides={"device_processor": {"device": device}})
chunk = policy.config.chunk_size
check("reload", True, f"{ptype} {ckpt.relative_to(REPO_ROOT)} ({sum(x.numel() for x in policy.parameters()) / 1e6:.1f} M params, chunk {chunk}, {device})")

ds = LeRobotDataset("cta/p0_pickplace", root=REPO_ROOT / a.root, episodes=a.episodes, video_backend="pyav",
                    delta_timestamps={"action": [i / 30 for i in range(chunk)]})
ep_idx = np.asarray(ds.hf_dataset["episode_index"])
errs, first_errs, preds, lat = [], [], [], []
for ep in a.episodes:
    rows = np.flatnonzero(ep_idx == ep)
    for f in a.frames:
        if f >= len(rows):
            continue
        item = ds[int(rows[f])]
        obs = {k: v.unsqueeze(0) for k, v in item.items() if k.startswith("observation.")}
        obs["task"] = [item["task"]]  # language-conditioned policies (SmolVLA); ignored by ACT
        with torch.no_grad():
            t0 = time.perf_counter()
            pred = post(policy.predict_action_chunk(pre(obs)))
            if device == "cuda":
                torch.cuda.synchronize()
            lat.append(time.perf_counter() - t0)
        pred = pred.squeeze(0).float().cpu().numpy()  # (chunk, 6) rad
        gt = item["action"].numpy()
        valid = ~item["action_is_pad"].numpy()
        e = np.rad2deg(np.abs(pred[valid] - gt[valid]))
        errs.append(e.mean(0))
        first_errs.append(e[0])
        preds.append(pred)

P = np.concatenate(preds)
check("finite", np.isfinite(P).all(), f"{len(preds)} chunks x {chunk} steps")
try:  # the project's single unit layer: radians within limits +/- DEFAULT_LIMIT_TOL (5 % of each range)
    validate(P, "rad", emb)
    in_lim = True
except UnitRangeError:
    in_lim = False
over = np.rad2deg(np.maximum(P.max(0) - hi, 0) + np.maximum(lo - P.min(0), 0))
check("within_joint_limits", in_lim,
      f"pred min {np.round(np.rad2deg(P.min(0)), 1).tolist()} max {np.round(np.rad2deg(P.max(0)), 1).tolist()} deg; "
      f"limits {np.round(np.rad2deg(lo), 1).tolist()} .. {np.round(np.rad2deg(hi), 1).tolist()}; "
      f"overshoot past the hard limit {np.round(over, 1).tolist()} deg (to clip at runtime)")
mean_err = np.mean(errs, 0)
check("open_loop_error", mean_err.mean() < a.max_mean_err_deg,
      f"mean |pred - expert| over the chunk per joint {np.round(mean_err, 2).tolist()} deg "
      f"(avg {mean_err.mean():.2f}, first step {np.mean(first_errs):.2f}; threshold {a.max_mean_err_deg})")
lat_ms = 1000 * np.median(lat[1:] if len(lat) > 1 else lat)
check("latency", lat_ms < 1000 / 30 * chunk, f"median {lat_ms:.1f} ms per chunk (budget: 1 chunk per {chunk} steps at 30 Hz)")

out = REPO_ROOT / a.out
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(checks, indent=2))
ok = all(c["ok"] for c in checks.values())
print("CHECKPOINT OK" if ok else "CHECKPOINT CHECK FAILED")
sys.exit(0 if ok else 1)
