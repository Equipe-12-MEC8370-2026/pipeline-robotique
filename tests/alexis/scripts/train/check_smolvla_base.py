r"""P1.2: load the verified SmolVLA base with the network off, then run one inference (run with .venv-learn).

    .\.venv-learn\Scripts\python.exe scripts\train\check_smolvla_base.py

Checks: SOURCE.json present (from fetch_smolvla_base.py); policy + pre/post-processors load with
HF_HUB_OFFLINE=1; parameter counts (total / trainable with the default freezing); one action chunk
from a dummy observation (3 cameras, state, task text) is finite with the right shape; latency and
peak VRAM. Dummy inputs: this checks the plumbing, not behaviour.
"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"  # must be set before huggingface_hub is imported
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import json
import sys
import time
from pathlib import Path

import torch

TASK = "pick up the red cube and place it on the green target"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="data/pretrained/smolvla_base")
    p.add_argument("--out", default="outputs/p1_2/smolvla_base_check.json")
    a = p.parse_args()
    base = Path(a.base)
    res = {}

    def check(name, ok, detail):
        res[name] = {"ok": bool(ok), "detail": detail}
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    src = base / "SOURCE.json"
    check("source", src.is_file(), json.loads(src.read_text())["revision"] if src.is_file() else "missing SOURCE.json")

    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    policy = SmolVLAPolicy.from_pretrained(str(base)).to(dev).eval()
    n_all = sum(q.numel() for q in policy.parameters())
    n_train = sum(q.numel() for q in policy.parameters() if q.requires_grad)
    cfg = policy.config
    check("load_offline", True, f"{n_all / 1e6:.1f} M params, {n_train / 1e6:.1f} M trainable (expert only), "
          f"chunk {cfg.chunk_size}, {time.time() - t0:.0f} s")
    check("features", list(cfg.image_features) == [f"observation.images.camera{i}" for i in (1, 2, 3)]
          and cfg.robot_state_feature.shape == (6,) and cfg.action_feature.shape == (6,),
          f"images {list(cfg.image_features)}, state {cfg.robot_state_feature.shape}, action {cfg.action_feature.shape}")

    pre, post = make_pre_post_processors(cfg, pretrained_path=str(base),
                                         preprocessor_overrides={"device_processor": {"device": dev}})
    check("processors_offline", True, f"pre {[type(s).__name__ for s in pre.steps]}; post {[type(s).__name__ for s in post.steps]}")

    g = torch.Generator().manual_seed(0)
    obs = {k: torch.rand(3, 256, 256, generator=g) for k in cfg.image_features}
    obs["observation.state"] = torch.zeros(6)
    obs["task"] = TASK
    torch.cuda.reset_peak_memory_stats() if dev == "cuda" else None
    times = []
    with torch.inference_mode():
        for _ in range(6):
            t = time.perf_counter()
            batch = pre(dict(obs))
            chunk = post(policy.predict_action_chunk(batch))
            torch.cuda.synchronize() if dev == "cuda" else None
            times.append((time.perf_counter() - t) * 1000)
    shape = tuple(chunk.shape)
    check("inference", torch.isfinite(chunk).all().item() and shape == (1, cfg.chunk_size, 6),
          f"chunk {shape}, finite {torch.isfinite(chunk).all().item()}")
    med = sorted(times[1:])[len(times[1:]) // 2]
    vram = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0.0
    check("latency", med < 1000, f"median {med:.0f} ms per chunk (first {times[0]:.0f} ms), peak VRAM {vram:.2f} GB, {dev}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    ok = all(r["ok"] for r in res.values())
    print("SMOLVLA BASE OK" if ok else "SMOLVLA BASE FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
