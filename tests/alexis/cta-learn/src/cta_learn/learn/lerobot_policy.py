"""Any trained LeRobot policy checkpoint (ACT, SmolVLA, ...) behind the policy-server protocol (run in .venv-learn).

Observations arrive as the simulator / robot produces them:
    "observation.state"            float32 (6,)     joint positions, radians
    "observation.images.<camera>"  uint8 (H, W, 3)  RGB, with the SIMULATOR's camera names (front, wrist)
and are converted to the dataset convention LeRobot was trained on (float CHW in [0, 1], batch 1).
Camera renaming (e.g. front -> camera1 for SmolVLA) is done by the checkpoint's own saved preprocessor;
cameras the policy pads as empty ("empty_camera_*") are not requested.
Language-conditioned policies (a tokenizer step in the preprocessor) also get the task text: given
explicitly, or read from the training dataset (train_config.json -> dataset root -> meta/tasks.parquet).
The returned action is float32 (6,) in radians, NOT clipped: clipping belongs to the robot side.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def _training_dataset_root(ckpt: str | Path) -> Path | None:
    tc = Path(ckpt) / "train_config.json"
    if not tc.is_file():
        return None
    root = json.loads(tc.read_text(encoding="utf-8")).get("dataset", {}).get("root")
    return Path(root) if root else None


def training_cameras(ckpt: str | Path) -> list[str] | None:
    """Camera keys of the dataset the checkpoint was trained on (what the policy actually saw), or None."""
    root = _training_dataset_root(ckpt)
    info = root / "meta" / "info.json" if root else None
    if not info or not info.is_file():
        return None
    feats = json.loads(info.read_text(encoding="utf-8"))["features"]
    return [k for k, v in feats.items() if v.get("dtype") in ("video", "image")]


def training_task(ckpt: str | Path) -> str | None:
    """The single task string of the dataset the checkpoint was trained on, or None if unknown."""
    root = _training_dataset_root(ckpt)
    tasks = root / "meta" / "tasks.parquet" if root else None
    if not tasks or not tasks.is_file():
        return None
    import pandas as pd

    names = list(pd.read_parquet(tasks).index)
    if len(names) != 1:
        raise ValueError(f"{tasks} has {len(names)} tasks; pass the task text explicitly")
    return str(names[0])


def client_image_keys(image_features, rename_map: dict[str, str], trained_on: list[str] | None = None) -> list[str]:
    """Camera keys the client must send.

    With the training dataset's cameras known (``trained_on``), exactly those: a fine-tuned base can keep
    feature slots the dataset never filled (SmolVLA base: camera3), padded at run time. Each must reach a
    policy feature after renaming. Otherwise (e.g. a base model): every feature, inverse-renamed, minus
    the padded ``empty_camera_*`` slots.
    """
    if trained_on is not None:
        bad = [k for k in trained_on if rename_map.get(k, k) not in image_features]
        if bad:
            raise ValueError(f"training cameras {bad} do not map to policy image features {list(image_features)}")
        return list(trained_on)
    inverse = {v: k for k, v in rename_map.items()}
    return [inverse.get(k, k) for k in image_features if ".empty_camera_" not in k]


class LeRobotPolicy:
    def __init__(self, ckpt: str | Path, device: str | None = None, n_action_steps: int | None = None,
                 temporal_ensemble_coeff: float | None = None, task: str | None = None):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.factory import get_policy_class
        from lerobot.processor.rename_processor import RenameObservationsProcessorStep
        from lerobot.processor.tokenizer_processor import TokenizerProcessorStep

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cfg = PreTrainedConfig.from_pretrained(ckpt)
        cfg.device = self.device
        if temporal_ensemble_coeff is not None:
            if not hasattr(cfg, "temporal_ensemble_coeff"):
                raise ValueError(f"temporal ensembling is ACT-only, not available for '{cfg.type}'")
            cfg.temporal_ensemble_coeff, cfg.n_action_steps = temporal_ensemble_coeff, 1
        elif n_action_steps is not None:
            if not 1 <= n_action_steps <= cfg.chunk_size:
                raise ValueError(f"n_action_steps {n_action_steps} not in [1, chunk_size={cfg.chunk_size}]")
            cfg.n_action_steps = n_action_steps
        self.policy = get_policy_class(cfg.type).from_pretrained(ckpt, config=cfg).to(self.device).eval()
        self.pre, self.post = make_pre_post_processors(
            self.policy.config, pretrained_path=str(ckpt),
            preprocessor_overrides={"device_processor": {"device": self.device}})
        rename = next((s.rename_map for s in self.pre.steps if isinstance(s, RenameObservationsProcessorStep)), {})
        self.image_keys = client_image_keys(self.policy.config.image_features, rename, training_cameras(ckpt))
        self.needs_task = any(isinstance(s, TokenizerProcessorStep) for s in self.pre.steps)
        self.task = (task or training_task(ckpt)) if self.needs_task else None
        if self.needs_task and not self.task:
            raise ValueError(f"'{cfg.type}' needs a task text and none was found in {ckpt}: pass task=...")
        self.ckpt = str(ckpt)
        self.calls = 0
        self.reset()

    def info(self) -> dict:
        c = self.policy.config
        return {"policy": c.type, "ckpt": self.ckpt, "device": self.device, "chunk_size": c.chunk_size,
                "n_action_steps": c.n_action_steps,
                "temporal_ensemble_coeff": getattr(c, "temporal_ensemble_coeff", None),
                "image_keys": self.image_keys, "task": self.task}

    def reset(self) -> None:
        self.policy.reset()

    def to_batch(self, obs: dict[str, np.ndarray]) -> dict:
        missing = [k for k in ["observation.state", *self.image_keys] if k not in obs]
        if missing:
            raise KeyError(f"missing observation keys {missing}")
        state = np.asarray(obs["observation.state"], np.float32)
        if state.shape != (6,):
            raise ValueError(f"observation.state shape {state.shape}, expected (6,)")
        batch = {"observation.state": torch.from_numpy(state).unsqueeze(0)}
        for k in self.image_keys:
            img = obs[k]
            if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
                raise ValueError(f"{k}: expected uint8 (H, W, 3), got {img.dtype} {img.shape}")
            batch[k] = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
        if self.needs_task:
            batch["task"] = self.task
        return batch

    @torch.no_grad()
    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        self.calls += 1
        action = self.post(self.policy.select_action(self.pre(self.to_batch(obs))))
        return action.squeeze(0).float().cpu().numpy()

    def handler(self, kind: str, meta: dict, arrays: dict) -> tuple[dict, dict]:
        if kind == "info":
            return self.info(), {}
        if kind == "reset":
            self.reset()
            return {}, {}
        if kind == "act":
            return {}, {"action": self.act(arrays)}
        raise ValueError(f"unknown request '{kind}'")
