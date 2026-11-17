"""P0.4 tests: config building, dataset preflight, log parsing, and one ACT step on a real batch (CPU)."""

import math

import numpy as np
import pytest

from cta_learn import REPO_ROOT
from cta_learn.core import UnitRangeError, embodiment_from_dict, load_embodiment

lerobot = pytest.importorskip("lerobot")

from cta_learn.learn.train_act import build_config, parse_train_log, preflight, smoke_verdict  # noqa: E402

DATASET = REPO_ROOT / "data" / "datasets" / "p0_pickplace"
needs_dataset = pytest.mark.skipif(not (DATASET / "meta" / "info.json").exists(), reason="P0 dataset not generated")
YAML = "configs/pipelines/train_act_p0.yaml"


def test_build_config_defaults():
    cfg = build_config(YAML)
    assert cfg.policy.type == "act"
    assert cfg.policy.chunk_size == cfg.policy.n_action_steps == 100
    assert cfg.policy.use_vae and not cfg.policy.push_to_hub
    assert cfg.dataset.video_backend == "pyav"
    assert cfg.output_dir == REPO_ROOT / "outputs" / "train" / "act_p0"
    assert str(REPO_ROOT) in cfg.dataset.root and cfg.save_freq == 10_000 and cfg.env_eval_freq == 0


def test_build_config_smoke_and_overrides():
    cfg = build_config(YAML, ["--batch_size=4", "--policy.chunk_size=50", "--policy.n_action_steps=50"], smoke=True)
    assert cfg.steps == 100 and cfg.save_freq == 100 and cfg.log_freq == 10
    assert cfg.batch_size == 4 and cfg.policy.chunk_size == 50
    assert "smoke" in cfg.output_dir.parts and cfg.output_dir.is_absolute()


def test_build_config_smolvla_pretrained_path():
    from lerobot.configs import parser

    cfg = build_config("configs/pipelines/train_smolvla_p1.yaml", ["--batch_size=4"], smoke=True)
    assert cfg.policy is None  # filled by LeRobot's validate() from the pretrained path at train time
    assert parser.get_path_arg("policy") == str(REPO_ROOT / "data" / "pretrained" / "smolvla_base")
    assert "--empty_cameras=1" in parser.get_yaml_overrides("policy")
    assert cfg.rename_map == {"observation.images.front": "observation.images.camera1",
                              "observation.images.wrist": "observation.images.camera2"}
    assert cfg.batch_size == 4 and cfg.steps == 100 and "smoke" in cfg.output_dir.parts
    build_config(YAML)  # a later type-based config must not inherit the pretrained path
    assert parser.get_path_arg("policy") is None


@needs_dataset
def test_preflight_p0_dataset():
    info = preflight(DATASET, load_embodiment("so101"))
    assert info["episodes"] == 150 and info["fps"] == 30 and info["cameras"] == ["front", "wrist"]


@needs_dataset
def test_preflight_refuses_wrong_units():
    # same joint names as SO-101 but tiny limits: the radian stats fall outside -> refused
    so = load_embodiment("so101")
    tight = embodiment_from_dict({
        "name": "tight", "control_hz": 30,
        "arms": [{"name": "main", "ee_body": "g", "joints": [
            {"name": n, "kind": "gripper" if n == "gripper" else "revolute", "limits_deg": [-1, 1],
             "lerobot_norm": "0_100" if n == "gripper" else "m100_100"} for n in so.joint_names]}],
        "cameras": [{"name": c, "width": 640, "height": 480, "fps": 30, "mount": "world"} for c in so.camera_names],
    })
    with pytest.raises(UnitRangeError):
        preflight(DATASET, tight)


LOG = """INFO 2026-11-11 21:00:01 n_train.py:641 step:10 smpl:80 ep:0 epch:0.00 loss:12.500 grdn:150.2 lr:1.0e-05 updt_s:0.120 data_s:0.300 smp/s:19 mem_gb:3.10
Training:  10%|#         | 10/100
INFO 2026-11-11 21:00:05 n_train.py:641 step:100 smpl:800 ep:3 epch:0.02 loss:4.200 grdn:80.1 lr:1.0e-05 updt_s:0.110 data_s:0.050 smp/s:50 mem_gb:3.20
INFO 2026-11-11 21:10:05 n_train.py:641 step:10K smpl:80K ep:257 epch:1.72 loss:0.150 grdn:9.0 lr:1.0e-05 updt_s:0.100 data_s:0.040 smp/s:57 mem_gb:3.20
"""


def test_parse_train_log():
    recs = parse_train_log(LOG)
    assert [r["step"] for r in recs] == [10, 100, 10_000]
    assert recs[0]["loss"] == 12.5 and recs[0]["data_s"] == 0.3 and recs[1]["gpu_mem_gb"] == 3.2
    assert math.isclose(recs[0]["lr"], 1e-5)


def test_smoke_verdict():
    v = smoke_verdict(parse_train_log(LOG))
    assert v["ok"] and v["finite"] and v["bottleneck"] == "gpu update"
    assert math.isclose(v["steps_per_s"], 1 / (0.105 + 0.045))
    assert not smoke_verdict(parse_train_log(LOG)[:1])["ok"]
    flat = [{"step": 10, "loss": 5.0}, {"step": 100, "loss": 4.9}]
    assert not smoke_verdict(flat)["ok"]
    assert not smoke_verdict([{"step": 10, "loss": 5.0}, {"step": 20, "loss": float("nan")}])["ok"]


@needs_dataset
def test_act_one_step_on_real_batch(tmp_path):
    """Dataset batch -> preprocessor -> ACT forward/backward -> select_action -> postprocessor; save/reload."""
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.policies.act.modeling_act import ACTPolicy

    chunk = 10
    cfg = build_config(YAML, [f"--policy.chunk_size={chunk}", f"--policy.n_action_steps={chunk}", "--policy.device=cpu"])
    cfg.policy.pretrained_backbone_weights = None  # no download in unit tests
    meta = LeRobotDatasetMetadata(cfg.dataset.repo_id, root=cfg.dataset.root)
    ds = LeRobotDataset(cfg.dataset.repo_id, root=cfg.dataset.root, episodes=[0],
                        delta_timestamps={"action": [i / meta.fps for i in range(chunk)]}, video_backend="pyav")
    batch = torch.utils.data.default_collate([ds[0], ds[50]])
    assert batch["action"].shape == (2, chunk, 6)

    torch.manual_seed(0)
    policy = make_policy(cfg.policy, ds_meta=meta)
    pre, post = make_pre_post_processors(cfg.policy, dataset_stats=meta.stats)
    loss, out = policy.forward(pre(batch))
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in policy.parameters())

    policy.eval()
    obs = {k: v[:1] for k, v in batch.items() if k.startswith("observation.")}
    with torch.no_grad():
        act = post(policy.select_action(pre(obs)))
    assert act.shape == (1, 6) and torch.isfinite(act).all()

    policy.save_pretrained(tmp_path)
    reloaded = ACTPolicy.from_pretrained(tmp_path).eval()
    policy.reset()
    reloaded.reset()
    with torch.no_grad():
        a1 = policy.select_action(pre(obs))
        a2 = reloaded.select_action(pre(obs))
    np.testing.assert_allclose(a1.numpy(), a2.numpy(), atol=1e-5)


def test_update_last_checkpoint_moves(tmp_path):
    from cta_learn.learn.train_act import update_last_checkpoint

    ckpts = tmp_path / "checkpoints"
    for step in ("000100", "000200"):
        (ckpts / step / "pretrained_model").mkdir(parents=True)
        (ckpts / step / "pretrained_model" / "step.txt").write_text(step)
        last = update_last_checkpoint(ckpts / step)
        assert (last / "pretrained_model" / "step.txt").read_text() == step
    assert sorted(p.name for p in ckpts.iterdir()) == ["000100", "000200", "last"]
    assert (ckpts / "000100" / "pretrained_model" / "step.txt").exists()  # old checkpoint kept


def test_build_resume_config(tmp_path):
    from cta_learn.learn.train_act import build_resume_config

    cfg = build_config(YAML)
    ckpt = tmp_path / "080000"
    (ckpt / "pretrained_model").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        build_resume_config(ckpt)
    cfg._save_pretrained(ckpt / "pretrained_model")
    with pytest.raises(FileNotFoundError):  # training_state missing
        build_resume_config(ckpt)
    (ckpt / "training_state").mkdir()
    rcfg, flag = build_resume_config(ckpt)
    assert rcfg.resume and rcfg.steps == cfg.steps and rcfg.output_dir == cfg.output_dir
    assert flag == f"--config_path={ckpt / 'pretrained_model' / 'train_config.json'}"
