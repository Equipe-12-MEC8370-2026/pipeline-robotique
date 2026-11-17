"""Raw episode format written by the simulator runtime and read by the LeRobot converter.

One directory per episode::

    episode_000042/
        data.npz        state (T,6) rad, action (T,6) rad, timestamp (T,), plus extras (cube_pos, phase, ...)
        front.mp4       (T frames, H x W, 30 fps)  one video per camera
        wrist.mp4
        meta.json       task, fps, success, seed, expert params, units

Videos are encoded on the fly by an ffmpeg subprocess (near-lossless H.264), so 150
episodes take a few hundred MB instead of ~80 GB of raw frames. Episodes are written to
``<name>.tmp`` and renamed only when kept, so a crash never leaves a half episode.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

UNITS = {"state": "rad", "action": "rad", "positions": "m (env frame)"}


class _VideoWriter:
    def __init__(self, path: Path, width: int, height: int, fps: int, crf: int = 16):
        self.path = path
        self.proc = subprocess.Popen(
            [
                "ffmpeg", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p", str(path),
            ],
            stdin=subprocess.PIPE,
        )
        self.frames = 0

    def write(self, frame: np.ndarray) -> None:
        self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        self.frames += 1

    def close(self) -> None:
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed while writing {self.path}")

    def kill(self) -> None:
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        self.proc.kill()
        self.proc.wait()


class EpisodeWriter:
    def __init__(self, root: Path, index: int, cameras: dict[str, tuple[int, int]], fps: int = 30):
        self.final_dir = Path(root) / f"episode_{index:06d}"
        self.tmp_dir = self.final_dir.with_name(self.final_dir.name + ".tmp")
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir)
        self.tmp_dir.mkdir(parents=True)
        self.fps = fps
        self.videos = {name: _VideoWriter(self.tmp_dir / f"{name}.mp4", w, h, fps) for name, (w, h) in cameras.items()}
        self.columns: dict[str, list] = {}

    def add(self, images: dict[str, np.ndarray], **arrays) -> None:
        for name, img in images.items():
            self.videos[name].write(img)
        for key, val in arrays.items():
            self.columns.setdefault(key, []).append(np.asarray(val))

    def __len__(self) -> int:
        return len(self.columns.get("state", []))

    def keep(self, meta: dict) -> Path:
        for v in self.videos.values():
            v.close()
        data = {k: np.stack(v) for k, v in self.columns.items()}
        n = len(self)
        data["timestamp"] = np.arange(n, dtype=np.float64) / self.fps
        np.savez_compressed(self.tmp_dir / "data.npz", **data)
        meta = {**meta, "fps": self.fps, "num_frames": n, "cameras": list(self.videos), "units": UNITS}
        (self.tmp_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=float))
        if self.final_dir.exists():
            shutil.rmtree(self.final_dir)
        self.tmp_dir.rename(self.final_dir)
        return self.final_dir

    def discard(self) -> None:
        for v in self.videos.values():
            v.kill()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)


def list_episodes(root: Path) -> list[Path]:
    return sorted(p for p in Path(root).glob("episode_*") if p.is_dir() and not p.name.endswith(".tmp"))


def load_episode(ep_dir: Path) -> tuple[dict, dict]:
    """Return (arrays, meta). Videos are not decoded here (see read_video_frames)."""
    with np.load(Path(ep_dir) / "data.npz") as z:
        arrays = {k: z[k] for k in z.files}
    meta = json.loads((Path(ep_dir) / "meta.json").read_text())
    return arrays, meta


def read_video_frames(path: Path) -> np.ndarray:
    """Decode an episode video to (T, H, W, 3) uint8 with the ffmpeg CLI (no Python video dependency)."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
         "-of", "csv=p=0:s=x", str(path)],
        capture_output=True, text=True, check=True,
    )
    w, h = (int(v) for v in probe.stdout.strip().split("x"))
    raw = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
