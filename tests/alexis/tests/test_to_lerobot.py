import shutil

import numpy as np
import pytest

from cta_learn.core import UnitRangeError, embodiment_from_dict
from cta_learn.datagen.episode_io import EpisodeWriter

lerobot = pytest.importorskip("lerobot")
pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")

# tiny embodiment: 2 joints, one 32x24 camera -> fast test
EMB = embodiment_from_dict({
    "name": "tiny", "control_hz": 30,
    "arms": [{"name": "main", "ee_body": "g", "joints": [
        {"name": "j1", "kind": "revolute", "limits_deg": [-90, 90], "lerobot_norm": "m100_100"},
        {"name": "grip", "kind": "gripper", "limits_deg": [0, 90], "lerobot_norm": "0_100"}]}],
    "cameras": [{"name": "front", "width": 32, "height": 24, "fps": 30, "mount": "world"}],
})


def _write(root, idx, n, scale=0.5):
    w = EpisodeWriter(root, idx, {"front": (32, 24)})
    for t in range(n):
        img = np.full((24, 32, 3), (t * 9) % 255, np.uint8)
        w.add({"front": img}, state=np.array([scale * np.sin(t / 5), 0.3]), action=np.array([scale * np.sin((t + 1) / 5), 0.3]))
    w.keep({"task": "tiny task", "success": True})


def test_convert_round_trip(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from cta_learn.datagen.to_lerobot import convert

    raw = tmp_path / "raw"
    _write(raw, 0, 10)
    _write(raw, 1, 7)
    res = convert(raw, tmp_path / "ds", "local/tiny", EMB)
    assert res == {"episodes": 2, "frames": 17, "root": str(tmp_path / "ds"), "fps": 30}
    ds = LeRobotDataset("local/tiny", root=tmp_path / "ds")
    assert ds.num_episodes == 2 and ds.num_frames == 17
    item = ds[3]
    assert tuple(item["observation.state"].shape) == (2,)
    assert tuple(item["observation.images.front"].shape) == (3, 24, 32)
    np.testing.assert_allclose(item["action"].numpy(), [0.5 * np.sin(4 / 5), 0.3], atol=1e-6)
    assert ds.meta.features["observation.state"]["names"] == ["j1.pos", "grip.pos"]


def test_convert_refuses_degrees(tmp_path):
    from cta_learn.datagen.to_lerobot import convert

    raw = tmp_path / "raw"
    _write(raw, 0, 5, scale=60.0)  # 60 "rad" = clearly degrees -> must be refused
    with pytest.raises(UnitRangeError):
        convert(raw, tmp_path / "ds", "local/tiny", EMB)
    assert not (tmp_path / "ds").exists()


def test_convert_concatenates_raw_roots(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from cta_learn.datagen.to_lerobot import convert

    _write(tmp_path / "a", 0, 6)
    _write(tmp_path / "b", 0, 4, scale=0.2)
    res = convert([tmp_path / "a", tmp_path / "b"], tmp_path / "ds", "local/tiny", EMB)
    assert res["episodes"] == 2 and res["frames"] == 10
    ds = LeRobotDataset("local/tiny", root=tmp_path / "ds")
    np.testing.assert_allclose(ds[7]["action"].numpy(), [0.2 * np.sin(2 / 5), 0.3], atol=1e-6)
