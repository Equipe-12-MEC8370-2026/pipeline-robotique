"""P1.3: generic LeRobot policy wrapper (camera keys, task text, ACT + SmolVLA checkpoints)."""

import json

import numpy as np
import pytest

from cta_learn import REPO_ROOT
from cta_learn.learn.lerobot_policy import client_image_keys, training_task

ACT_CKPT = REPO_ROOT / "outputs" / "train" / "act_p0" / "checkpoints" / "last" / "pretrained_model"
SMOLVLA_BASE = REPO_ROOT / "data" / "pretrained" / "smolvla_base"
TASK = "Pick up the red cube and place it on the green square."


def test_client_image_keys_inverts_rename_and_skips_empty():
    feats = ["observation.images.camera1", "observation.images.camera2", "observation.images.empty_camera_0"]
    rename = {"observation.images.front": "observation.images.camera1",
              "observation.images.wrist": "observation.images.camera2"}
    assert client_image_keys(feats, rename) == ["observation.images.front", "observation.images.wrist"]
    assert client_image_keys(feats[:2], {}) == feats[:2]
    # fine-tuned SmolVLA: base slot camera3 kept in the features but never in the dataset
    ft = ["observation.images.camera1", "observation.images.camera2", "observation.images.camera3",
          "observation.images.empty_camera_0"]
    sim = ["observation.images.front", "observation.images.wrist"]
    assert client_image_keys(ft, rename, trained_on=sim) == sim
    with pytest.raises(ValueError, match="do not map"):
        client_image_keys(ft, {}, trained_on=sim)


def test_training_task(tmp_path):
    import pandas as pd

    assert training_task(tmp_path) is None  # no train_config.json (e.g. a base model)
    ds = tmp_path / "ds"
    (ds / "meta").mkdir(parents=True)
    ck = tmp_path / "ck"
    ck.mkdir()
    (ck / "train_config.json").write_text(json.dumps({"dataset": {"root": str(ds)}}))
    assert training_task(ck) is None  # dataset without tasks.parquet
    pd.DataFrame({"task_index": [0]}, index=pd.Index([TASK], name="task")).to_parquet(ds / "meta" / "tasks.parquet")
    assert training_task(ck) == TASK
    pd.DataFrame({"task_index": [0, 1]}, index=pd.Index(["a", "b"], name="task")).to_parquet(ds / "meta" / "tasks.parquet")
    with pytest.raises(ValueError, match="2 tasks"):
        training_task(ck)


def _obs(keys, rng):
    from cta_learn.core import load_embodiment

    emb = load_embodiment("so101")
    return {"observation.state": ((emb.lower + emb.upper) / 2).astype(np.float32),
            **{k: rng.integers(0, 255, (480, 640, 3), dtype=np.uint8) for k in keys}}


@pytest.mark.skipif(not ACT_CKPT.exists(), reason="trained ACT checkpoint not present")
def test_act_through_generic_wrapper():
    from cta_learn.learn.lerobot_policy import LeRobotPolicy

    pol = LeRobotPolicy(ACT_CKPT, device="cpu", n_action_steps=5)
    info = pol.info()
    assert info["policy"] == "act" and info["task"] is None and not pol.needs_task
    assert info["image_keys"] == ["observation.images.front", "observation.images.wrist"]
    assert "task" not in pol.to_batch(_obs(info["image_keys"], np.random.default_rng(0)))


# any fine-tuned SmolVLA checkpoint: the P1.4 smoke runs (deleted 2026-11-16 to free disk space), else the final P1 one
SMOKE = sorted((REPO_ROOT / "outputs" / "train" / "smoke").glob("smolvla_p1_*/checkpoints/last/pretrained_model")) or [
    p for p in [REPO_ROOT / "outputs" / "train" / "smolvla_p1" / "checkpoints" / "last" / "pretrained_model"] if p.exists()]


@pytest.mark.skipif(not SMOKE, reason="no fine-tuned SmolVLA checkpoint")
def test_smolvla_finetuned_checkpoint_speaks_sim_names():
    import torch

    from cta_learn.learn.lerobot_policy import LeRobotPolicy

    pol = LeRobotPolicy(SMOKE[-1], device="cuda" if torch.cuda.is_available() else "cpu", n_action_steps=25)
    info = pol.info()
    assert info["policy"] == "smolvla" and info["n_action_steps"] == 25
    assert info["image_keys"] == ["observation.images.front", "observation.images.wrist"]
    assert info["task"] == TASK  # read from the training dataset
    obs = _obs(info["image_keys"], np.random.default_rng(1))
    a = pol.act(obs)
    assert a.shape == (6,) and np.isfinite(a).all()


@pytest.mark.skipif(not SMOLVLA_BASE.joinpath("model.safetensors").exists(), reason="SmolVLA base not fetched (P1.2)")
def test_smolvla_base_through_generic_wrapper():
    import torch

    from cta_learn.learn.lerobot_policy import LeRobotPolicy

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pol = LeRobotPolicy(SMOLVLA_BASE, device=dev, n_action_steps=10, task=TASK)
    info = pol.info()
    assert info["policy"] == "smolvla" and info["chunk_size"] == 50 and info["n_action_steps"] == 10
    assert info["task"] == TASK and pol.needs_task
    assert info["image_keys"] == [f"observation.images.camera{i}" for i in (1, 2, 3)]
    obs = _obs(info["image_keys"], np.random.default_rng(0))
    assert pol.to_batch(obs)["task"] == TASK
    acts = [pol.handler("act", {}, obs)[1]["action"] for _ in range(12)]  # 2 chunks (replan after 10)
    assert all(a.shape == (6,) and a.dtype == np.float32 and np.isfinite(a).all() for a in acts)
    assert pol.calls == 12
    with pytest.raises(ValueError, match="ACT-only"):
        LeRobotPolicy(SMOLVLA_BASE, device=dev, temporal_ensemble_coeff=0.01, task=TASK)
