import shutil

import numpy as np
import pytest

from cta_learn.datagen.episode_io import EpisodeWriter, list_episodes, load_episode, read_video_frames

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")


def _frame(t, h=48, w=64):
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :, 0] = (t * 20) % 255
    img[h // 4 : h // 2, w // 4 : w // 2] = 255
    return img


def test_kept_episode_round_trips(tmp_path):
    w = EpisodeWriter(tmp_path, 3, {"front": (64, 48), "wrist": (64, 48)}, fps=30)
    for t in range(12):
        w.add({"front": _frame(t), "wrist": _frame(t + 1)}, state=np.full(6, t, float), action=np.full(6, t + 0.5))
    out = w.keep({"task": "test", "success": True})
    assert out.name == "episode_000003" and list_episodes(tmp_path) == [out]
    arrays, meta = load_episode(out)
    assert arrays["state"].shape == (12, 6) and arrays["action"][4, 0] == 4.5
    np.testing.assert_allclose(arrays["timestamp"][:3], [0, 1 / 30, 2 / 30])
    assert meta["num_frames"] == 12 and meta["units"]["state"] == "rad"
    frames = read_video_frames(out / "front.mp4")
    assert frames.shape == (12, 48, 64, 3)
    # near-lossless: the white square survives encoding
    assert frames[5, 15:20, 20:28].mean() > 230


def test_discarded_episode_leaves_nothing(tmp_path):
    w = EpisodeWriter(tmp_path, 0, {"front": (64, 48)})
    w.add({"front": _frame(0)}, state=np.zeros(6), action=np.zeros(6))
    w.discard()
    assert list(tmp_path.iterdir()) == []
