"""Convert raw episodes (``episode_io`` format) into a LeRobot v3 dataset. Runs in ``.venv-learn``.

Feature names follow the embodiment so every consumer sees the same joint order and unit::

    observation.state        float32 (6,)  rad, names = so101 joint names
    action                   float32 (6,)  rad, absolute joint targets
    observation.images.front video  480x640x3
    observation.images.wrist video  480x640x3
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from cta_learn.core import Embodiment, check_dataset_stats
from cta_learn.datagen.episode_io import list_episodes, load_episode, read_video_frames


def lerobot_features(emb: Embodiment) -> dict:
    names = [f"{n}.pos" for n in emb.joint_names]
    feats = {
        "observation.state": {"dtype": "float32", "shape": (emb.num_joints,), "names": names},
        "action": {"dtype": "float32", "shape": (emb.num_joints,), "names": names},
    }
    for cam in emb.cameras:
        feats[f"observation.images.{cam.name}"] = {
            "dtype": "video",
            "shape": (cam.height, cam.width, 3),
            "names": ["height", "width", "channels"],
        }
    return feats


def convert(raw_dir: Path | list[Path], out_dir: Path, repo_id: str, emb: Embodiment, limit: int | None = None,
            overwrite: bool = False) -> dict:
    """``raw_dir`` may be a list of raw roots: their episodes are concatenated in that order."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    raw_dirs = [Path(raw_dir)] if isinstance(raw_dir, (str, Path)) else [Path(r) for r in raw_dir]
    episodes = [e for r in raw_dirs for e in list_episodes(r)]
    episodes = episodes[:limit] if limit else episodes
    if not episodes:
        raise FileNotFoundError(f"no episodes in {raw_dirs}")
    out_dir = Path(out_dir)
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{out_dir} exists (use overwrite=True)")
        shutil.rmtree(out_dir)

    # unit guard on ALL raw data before writing anything
    states = np.concatenate([load_episode(e)[0]["state"] for e in episodes])
    actions = np.concatenate([load_episode(e)[0]["action"] for e in episodes])
    check_dataset_stats(states.min(0), states.max(0), "rad", emb)
    check_dataset_stats(actions.min(0), actions.max(0), "rad", emb)

    fps = load_episode(episodes[0])[1]["fps"]
    ds = LeRobotDataset.create(repo_id=repo_id, fps=fps, features=lerobot_features(emb), root=out_dir,
                               robot_type=emb.name, use_videos=True)
    frames_total = 0
    for ep in episodes:
        arrays, meta = load_episode(ep)
        videos = {c.name: read_video_frames(ep / f"{c.name}.mp4") for c in emb.cameras}
        n = len(arrays["state"])
        for name, v in videos.items():
            if len(v) != n:
                raise ValueError(f"{ep.name}: {name}.mp4 has {len(v)} frames, data has {n}")
        for t in range(n):
            frame = {
                "observation.state": arrays["state"][t].astype(np.float32),
                "action": arrays["action"][t].astype(np.float32),
                "task": meta["task"],
            }
            for name, v in videos.items():
                frame[f"observation.images.{name}"] = v[t]
            ds.add_frame(frame)
        # parallel encoding spawns worker processes that died (BrokenProcessPool) with 2x 640x480 cameras
        # on 16 GB RAM (2026-11-11): encode in-process.
        ds.save_episode(parallel_encoding=False)
        frames_total += n
    ds.finalize()
    return {"episodes": len(episodes), "frames": frames_total, "root": str(out_dir), "fps": fps}
