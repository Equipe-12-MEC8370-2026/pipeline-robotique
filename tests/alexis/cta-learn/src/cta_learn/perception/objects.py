"""M2 — object masks over a clip with SAM 2 (video predictor), tracked in chunks to bound RAM.

SAM 2 keeps every frame of a session as a 1024x1024 float tensor (~12.6 MB on CPU): a 36 s clip at 15 fps
would need ~7 GB. The clip is therefore split into chunks; tracking starts at the prompt frame (box), runs
forward chunk by chunk and backward chunk by chunk, each new chunk being seeded with the last mask of the
previous one (overlap of one frame). sam2/torch are imported lazily: the helpers work in any environment.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def plan_chunks(n: int, start: int, chunk: int, lo: int = 0, hi: int | None = None) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Forward and backward chunk ranges [a, b] (inclusive) covering lo..hi (default 0..n-1) from ``start``.

    Forward chunks begin at start, then each begins on the previous chunk's last frame (seed). Backward chunks
    end at start, then each ends on the previous chunk's first frame. chunk >= 2.
    """
    hi = n - 1 if hi is None else hi
    if not 0 <= lo <= start <= hi < n or chunk < 2:
        raise ValueError("bad start, range or chunk")
    fwd, a = [], start
    while a < hi:
        b = min(a + chunk - 1, hi)
        fwd.append((a, b))
        a = b
    bwd, b = [], start
    while b > lo:
        a = max(b - chunk + 1, lo)
        bwd.append((a, b))
        b = a
    return fwd, bwd


def merge_tracks(tracks: list[np.ndarray], prompt_idx: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Merge per-prompt tracks of ONE object: per frame, a non-empty mask beats an empty one; among non-empty
    masks the track whose prompt is nearest in time wins. Returns (masks, source track index or -1 if none)."""
    T = len(tracks[0])
    out = np.zeros_like(tracks[0])
    src = np.full(T, -1)
    nonempty = np.stack([t.reshape(T, -1).any(1) for t in tracks])  # (K, T)
    for i in range(T):
        cand = np.flatnonzero(nonempty[:, i])
        if cand.size:
            k = cand[np.argmin([abs(prompt_idx[c] - i) for c in cand])]
            out[i], src[i] = tracks[k][i], k
    return out, src


def mask_bbox(m: np.ndarray) -> np.ndarray | None:
    ys, xs = np.nonzero(m)
    if xs.size == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], float)


def box_iou(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def jump_frames(masks: np.ndarray, max_jump_px: float, max_area_ratio: float = 2.5) -> list[int]:
    """Frames where the mask centroid moves more than ``max_jump_px`` from the previous frame, or the area
    changes by more than ``max_area_ratio`` (either way), or the mask appears/disappears. Possible identity swaps."""
    out, prev_c, prev_a = [], None, None
    for i, m in enumerate(masks):
        a = int(m.sum())
        c = np.argwhere(m).mean(0) if a else None
        if i > 0:
            if (a == 0) != (prev_a == 0):
                out.append(i)
            elif a and prev_a:
                if np.linalg.norm(c - prev_c) > max_jump_px or max(a, prev_a) / min(a, prev_a) > max_area_ratio:
                    out.append(i)
        prev_c, prev_a = c, a
    return out


def extract_frames(video: Path, out_dir: Path, fps: int, height: int) -> list[Path]:
    """JPEG frames at a constant ``fps`` and ``height`` (the format SAM 2's loader reads), named 00000.jpg..."""
    from cta_learn.video.ingest import ffbin
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("*.jpg"):
        f.unlink()
    subprocess.run([ffbin("ffmpeg"), "-v", "error", "-y", "-i", str(video), "-vf", f"fps={fps},scale=-2:{height}",
                    "-q:v", "2", "-start_number", "0", str(out_dir / "%05d.jpg")], check=True)
    return sorted(out_dir.glob("*.jpg"))


def track_object(predictor, frames: list[Path], prompt_idx: int, box, chunk: int = 150) -> np.ndarray:
    """Masks (N, H, W) bool for one object over all ``frames``; ``box`` in frame pixels at ``prompt_idx``.

    A single prompt cannot carry an object through a full occlusion: use several prompts (one per visible
    stretch) and ``merge_tracks``.
    """
    import cv2
    import torch

    h, w = cv2.imread(str(frames[0])).shape[:2]
    masks = np.zeros((len(frames), h, w), bool)
    fwd, bwd = plan_chunks(len(frames), prompt_idx, chunk)

    def run(a: int, b: int, seed_idx: int, reverse: bool, box=None, seed_mask=None):
        with tempfile.TemporaryDirectory() as tmp:
            for k, f in enumerate(frames[a:b + 1]):
                shutil.copy(f, Path(tmp) / f"{k:05d}.jpg")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                st = predictor.init_state(video_path=tmp, offload_video_to_cpu=True, offload_state_to_cpu=True)
                if box is not None:
                    predictor.add_new_points_or_box(st, frame_idx=seed_idx - a, obj_id=1, box=np.asarray(box, np.float32))
                else:
                    predictor.add_new_mask(st, frame_idx=seed_idx - a, obj_id=1, mask=seed_mask)
                for fi, _, logits in predictor.propagate_in_video(st, start_frame_idx=seed_idx - a, reverse=reverse):
                    masks[a + fi] = (logits[0, 0] > 0).cpu().numpy()
            predictor.reset_state(st)
            del st
            torch.cuda.empty_cache()

    for k, (a, b) in enumerate(fwd):
        run(a, b, a, False, box=box if k == 0 else None, seed_mask=None if k == 0 else masks[a])
    for k, (a, b) in enumerate(bwd):
        run(a, b, b, True, box=box if k == 0 else None, seed_mask=None if k == 0 else masks[b])
    return masks


def save_masks(path: Path, masks: np.ndarray, fps: int) -> None:
    """Bool masks (T, H, W) bit-packed + shape + fps, compressed."""
    np.savez_compressed(path, packed=np.packbits(masks, axis=-1), shape=np.array(masks.shape), fps=fps)


def load_masks(path: Path) -> tuple[np.ndarray, int]:
    d = np.load(path)
    T, H, W = (int(v) for v in d["shape"])
    return np.unpackbits(d["packed"], axis=-1, count=W).astype(bool).reshape(T, H, W), int(d["fps"])
