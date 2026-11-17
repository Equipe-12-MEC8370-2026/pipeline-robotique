"""M2 — hand keypoints per frame (MediaPipe Hand Landmarker, VIDEO mode) on a synced clip.

Output arrays (T frames, 2 slots, 21 keypoints):
  kp_px    (T, 2, 21, 2) float32  pixel coordinates in the clip (NaN when no hand in the slot)
  kp_z     (T, 2, 21)    float32  MediaPipe relative depth (wrist-relative, image-width units)
  score    (T, 2)        float32  handedness score (0 when empty)
  t_s      (T,)          float32  clip time of each frame
Slot 0 = hand labelled "Left" by MediaPipe, slot 1 = "Right". The label is MediaPipe's guess (its convention
depends on whether the image is mirrored, and it can flip between frames): kept as is; P2.5 decides which hand
is which with geometry (arm side, continuity), not with this label.
mediapipe is imported lazily: the pure helpers below work in any environment.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

N_KP = 21
# MediaPipe hand skeleton (for drawing)
EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
         (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


@dataclass
class HandTrack:
    kp_px: np.ndarray
    kp_z: np.ndarray
    score: np.ndarray
    t_s: np.ndarray

    def n_hands(self) -> np.ndarray:
        return (~np.isnan(self.kp_px[:, :, 0, 0])).sum(1)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, kp_px=self.kp_px, kp_z=self.kp_z, score=self.score, t_s=self.t_s)

    @classmethod
    def load(cls, path: Path) -> "HandTrack":
        d = np.load(path)
        return cls(d["kp_px"], d["kp_z"], d["score"], d["t_s"])


def assign_slots(hands: list[tuple[str, float, np.ndarray]]) -> list[tuple[int, float, np.ndarray]]:
    """Map detections (label, score, (21,3) normalised kp) to slots 0 (Left) / 1 (Right), one per slot.

    If both detections carry the same label, the better-scored one keeps it and the other takes the free slot.
    """
    out, used = [], set()
    for label, score, kp in sorted(hands, key=lambda h: -h[1]):
        slot = 0 if label == "Left" else 1
        if slot in used:
            slot = 1 - slot
        if slot in used:
            continue  # more than 2 hands: drop the weakest
        used.add(slot)
        out.append((slot, score, kp))
    return out


def detection_rate(n_hands: np.ndarray, t_s: np.ndarray, intervals: list[tuple[float, float]], min_hands: int = 1) -> float:
    """Fraction of frames inside the given time intervals with at least ``min_hands`` hands."""
    m = np.zeros(len(t_s), bool)
    for a, b in intervals:
        m |= (t_s >= a) & (t_s < b)
    return float((n_hands[m] >= min_hands).mean()) if m.any() else float("nan")


def track_hands(video: Path, model: Path, num_hands: int = 2, min_detection: float = 0.5,
                min_presence: float = 0.5, min_tracking: float = 0.5) -> HandTrack:
    """Run the Hand Landmarker over every frame of ``video`` (VIDEO mode: temporal tracking between frames)."""
    import av
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

    opts = HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(model)), running_mode=RunningMode.VIDEO,
                                 num_hands=num_hands, min_hand_detection_confidence=min_detection,
                                 min_hand_presence_confidence=min_presence, min_tracking_confidence=min_tracking)
    kp_px, kp_z, score, ts = [], [], [], []
    with av.open(str(video)) as c, HandLandmarker.create_from_options(opts) as lm:
        s = c.streams.video[0]
        last_ms = -1
        for fr in c.decode(s):
            t = float(fr.pts * s.time_base)
            img = fr.to_ndarray(format="rgb24")
            h, w = img.shape[:2]
            ms = max(int(round(t * 1000)), last_ms + 1)  # strictly increasing timestamps
            last_ms = ms
            r = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img)), ms)
            dets = [(hd[0].category_name, hd[0].score, np.array([[p.x, p.y, p.z] for p in lmk], np.float32))
                    for hd, lmk in zip(r.handedness, r.hand_landmarks)]
            P = np.full((2, N_KP, 2), np.nan, np.float32)
            Z = np.full((2, N_KP), np.nan, np.float32)
            S = np.zeros(2, np.float32)
            for slot, sc, kp in assign_slots(dets):
                P[slot] = kp[:, :2] * [w, h]
                Z[slot] = kp[:, 2]
                S[slot] = sc
            kp_px.append(P), kp_z.append(Z), score.append(S), ts.append(t)
    return HandTrack(np.stack(kp_px), np.stack(kp_z), np.stack(score), np.asarray(ts, np.float32))
