"""ACT training helpers around LeRobot 0.6.1's official trainer (no re-implementation).

Run with .venv-learn. The script `scripts/train/train_act.py` uses these to:
  1. build LeRobot's ``TrainPipelineConfig`` from our YAML (+ CLI overrides), paths resolved
     against the repo root;
  2. refuse to train on a dataset whose state/action stats are not radians within the
     embodiment limits, or whose feature names / cameras do not match the embodiment;
  3. parse LeRobot's console metrics (``step:… loss:… updt_s:… data_s:…``) for the smoke verdict.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np

from cta_learn import REPO_ROOT
from cta_learn.core import Embodiment, check_dataset_stats

SMOKE_STEPS = 100


def _abs(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def _extract_pretrained_path(yaml_path: Path) -> str:
    """Fine-tuning configs give ``policy.path`` (e.g. SmolVLA base) instead of ``policy.type``.

    draccus rejects ``path``; LeRobot's own CLI strips it into a registry that ``validate()`` reads
    (the other ``policy.*`` keys become overrides of the pretrained config). We do the same, with the
    path resolved against the repo root. Returns the (possibly cleaned, temporary) config file to parse.
    """
    from lerobot.configs import parser
    from lerobot.configs.train import TrainPipelineConfig

    parser._config_path_args.pop("policy", None)
    parser._config_yaml_overrides.pop("policy", None)
    cleaned = parser.extract_path_fields_from_config(str(yaml_path), TrainPipelineConfig.__get_path_fields__())
    if "policy" in parser._config_path_args:
        path = Path(parser._config_path_args["policy"])
        parser._config_path_args["policy"] = str(_abs(path)) if not path.is_absolute() and _abs(path).exists() else str(path)
    return cleaned


def build_config(yaml_path: str | Path, overrides: list[str] | None = None, smoke: bool = False):
    """Parse the YAML into a LeRobot ``TrainPipelineConfig``; ``smoke`` = 100 steps in a fresh dir."""
    import draccus
    import lerobot.policies  # noqa: F401  (registers the policy config types, e.g. "act")
    from lerobot.configs.train import TrainPipelineConfig

    config_path = _extract_pretrained_path(_abs(yaml_path))
    cfg = draccus.parse(TrainPipelineConfig, config_path=config_path, args=list(overrides or []))
    cfg.dataset.root = str(_abs(cfg.dataset.root))
    if smoke:
        cfg.steps = SMOKE_STEPS
        cfg.log_freq = 10
        cfg.save_freq = SMOKE_STEPS
        cfg.output_dir = Path("outputs/train/smoke") / f"{cfg.job_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    cfg.output_dir = _abs(cfg.output_dir)
    return cfg


def build_resume_config(checkpoint_dir: str | Path):
    """Config to resume a run from ``checkpoints/<step>`` (weights, optimizer, RNG, data order).

    Returns ``(cfg, argv_flag)``: LeRobot's ``validate()`` reads ``--config_path`` from ``sys.argv``,
    so the caller must append ``argv_flag`` to ``sys.argv`` before calling ``train``.
    """
    import draccus
    import lerobot.policies  # noqa: F401
    from lerobot.configs.train import TrainPipelineConfig

    train_cfg = _abs(checkpoint_dir) / "pretrained_model" / "train_config.json"
    if not train_cfg.is_file():
        raise FileNotFoundError(f"no train_config.json in {checkpoint_dir}")
    if not (train_cfg.parent.parent / "training_state").is_dir():
        raise FileNotFoundError(f"no training_state in {checkpoint_dir}")
    cfg = draccus.parse(TrainPipelineConfig, config_path=str(train_cfg), args=["--resume=true"])
    return cfg, f"--config_path={train_cfg}"


def preflight(dataset_root: str | Path, emb: Embodiment) -> dict:
    """Check the dataset against the embodiment before any GPU time is spent. Raises on mismatch."""
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata("local/preflight", root=_abs(dataset_root))
    names = [f"{n}.pos" for n in emb.joint_names]
    for key in ("observation.state", "action"):
        got = meta.features[key]["names"]
        if got != names:
            raise ValueError(f"{key} names {got} != embodiment {names}")
        check_dataset_stats(np.asarray(meta.stats[key]["min"]), np.asarray(meta.stats[key]["max"]), "rad", emb)
    cams = sorted(k.removeprefix("observation.images.") for k in meta.camera_keys)
    if cams != sorted(emb.camera_names):
        raise ValueError(f"dataset cameras {cams} != embodiment cameras {sorted(emb.camera_names)}")
    return {"episodes": meta.total_episodes, "frames": meta.total_frames, "fps": meta.fps, "cameras": cams}


def update_last_checkpoint(checkpoint_dir: Path) -> Path:
    """Point ``checkpoints/last`` at ``checkpoint_dir``; Windows-safe replacement for LeRobot's version.

    LeRobot uses a symlink, which Windows refuses without Developer Mode (WinError 1314). We try the
    symlink first and fall back to a directory junction (no privilege needed, transparent to readers).
    """
    import os

    checkpoint_dir = Path(checkpoint_dir)
    last = checkpoint_dir.parent / "last"
    if last.is_symlink() or (hasattr(last, "is_junction") and last.is_junction()):
        os.unlink(last) if last.is_symlink() else os.rmdir(last)
    try:
        last.symlink_to(checkpoint_dir.relative_to(checkpoint_dir.parent), target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        import _winapi

        _winapi.CreateJunction(str(checkpoint_dir.resolve()), str(last))
    return last


def patch_lerobot_for_windows() -> None:
    """Swap LeRobot's symlink-only ``update_last_checkpoint`` (runtime patch, installed files untouched)."""
    import lerobot.scripts.lerobot_train as lt

    lt.update_last_checkpoint = update_last_checkpoint


_NUM = r"([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)([KMBTQ]?)"
_SUFFIX = {"": 1, "K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12, "Q": 1e15}
_FIELDS = {"step": "step", "loss": "loss", "grdn": "grad_norm", "lr": "lr",
           "updt_s": "update_s", "data_s": "data_s", "smp/s": "samples_per_s", "mem_gb": "gpu_mem_gb"}


def parse_train_log(text: str) -> list[dict]:
    """Extract LeRobot's periodic metric lines (``step:10 smpl:80 … loss:1.234 …``) as dicts."""
    records = []
    for line in text.splitlines():
        if "step:" not in line or "loss:" not in line:
            continue
        rec = {}
        for m in re.finditer(r"(\S+?):" + _NUM + r"(?=\s|$)", line, flags=re.IGNORECASE):
            key, val, suf = m.group(1), float(m.group(2)), m.group(3).upper()
            if key in _FIELDS:
                rec[_FIELDS[key]] = val * _SUFFIX[suf]
        if "step" in rec and "loss" in rec:
            records.append(rec)
    return records


def smoke_verdict(records: list[dict], min_drop: float = 0.2) -> dict:
    """Smoke test passes if the loss is finite and the last logged loss is ≥ ``min_drop`` below the first."""
    if len(records) < 2:
        return {"ok": False, "reason": f"only {len(records)} metric lines logged"}
    losses = [r["loss"] for r in records]
    first, last = losses[0], losses[-1]
    finite = all(np.isfinite(losses))
    ok = finite and last <= (1 - min_drop) * first
    upd = float(np.mean([r.get("update_s", np.nan) for r in records[1:]]))
    data = float(np.mean([r.get("data_s", np.nan) for r in records[1:]]))
    return {
        "ok": bool(ok), "first_loss": first, "last_loss": last, "finite": bool(finite),
        "mean_update_s": upd, "mean_data_s": data,
        "steps_per_s": 1.0 / (upd + data) if upd + data > 0 else None,
        "bottleneck": "data loading" if data > upd else "gpu update",
    }
