"""P0.4: train ACT with LeRobot's official trainer (run with .venv-learn).

    # smoke test first: 100 steps in outputs/train/smoke/<job>_<time>/ + verdict JSON
    .\\.venv-learn\\Scripts\\python.exe scripts\\train\\train_act.py --smoke
    # full run (any LeRobot override can follow, e.g. --steps=60000)
    .\\.venv-learn\\Scripts\\python.exe scripts\\train\\train_act.py --steps=60000

Before training, the dataset is checked against the SO-101 embodiment (radians within limits,
joint names, cameras). The console log is also written to <output_dir>.log; with --smoke the
loss trend, step time split (GPU update vs data loading) and peak VRAM go to <output_dir>.json.
"""

import argparse
import json
import sys

from cta_learn.core import load_embodiment
from cta_learn.learn.train_act import (
    build_config,
    build_resume_config,
    parse_train_log,
    patch_lerobot_for_windows,
    preflight,
    smoke_verdict,
)


class Tee:
    """Duplicate a stream into the log file (LeRobot logs to stderr, tqdm too)."""

    def __init__(self, stream, f):
        self.stream, self.f = stream, f

    def write(self, s):
        self.stream.write(s)
        self.f.write(s)
        return len(s)

    def flush(self):
        self.stream.flush()
        self.f.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/pipelines/train_act_p0.yaml")
    p.add_argument("--embodiment", default="so101")
    p.add_argument("--smoke", action="store_true", help="100 steps in a fresh dir, then a pass/fail verdict")
    p.add_argument("--resume", default=None, metavar="CHECKPOINT_DIR",
                   help="resume from checkpoints/<step> (its saved train_config.json is used, --config ignored)")
    a, overrides = p.parse_known_args()

    if a.resume:
        if overrides or a.smoke:
            sys.exit("--resume takes no other overrides")
        cfg, flag = build_resume_config(a.resume)
        sys.argv = [sys.argv[0], flag]  # LeRobot's validate() reads --config_path from sys.argv
    else:
        cfg = build_config(a.config, overrides, smoke=a.smoke)
    info = preflight(cfg.dataset.root, load_embodiment(a.embodiment))
    print(f"[preflight] dataset OK: {info}", flush=True)
    if cfg.output_dir.exists() and not a.resume:
        sys.exit(f"output dir {cfg.output_dir} already exists (pass --output_dir=... or --resume <ckpt>)")
    cfg.output_dir.parent.mkdir(parents=True, exist_ok=True)
    log_path = cfg.output_dir.with_suffix(".log")

    import torch
    from lerobot.scripts.lerobot_train import train

    patch_lerobot_for_windows()

    with open(log_path, "a" if a.resume else "w", encoding="utf-8") as f:
        if a.resume:
            f.write(f"\n===== RESUME from {a.resume} =====\n")
        sys.stdout, sys.stderr = Tee(sys.stdout, f), Tee(sys.stderr, f)
        try:
            train(cfg)
        finally:
            sys.stdout, sys.stderr = sys.stdout.stream, sys.stderr.stream

    if a.smoke:
        verdict = smoke_verdict(parse_train_log(log_path.read_text(encoding="utf-8")))
        verdict["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None
        verdict["checkpoint"] = str(cfg.output_dir / "checkpoints" / "last" / "pretrained_model")
        verdict["checkpoint_exists"] = (cfg.output_dir / "checkpoints" / "last" / "pretrained_model").exists()
        verdict["ok"] = verdict["ok"] and verdict["checkpoint_exists"]
        cfg.output_dir.with_suffix(".json").write_text(json.dumps(verdict, indent=2))
        print(json.dumps(verdict, indent=2))
        print("SMOKE OK" if verdict["ok"] else "SMOKE FAILED")
        sys.exit(0 if verdict["ok"] else 1)


if __name__ == "__main__":  # DataLoader workers use "spawn" and re-import this file
    main()
