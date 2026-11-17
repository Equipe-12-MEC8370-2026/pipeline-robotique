"""P2.4c — SAM 2 masks of the target parts over the TPV clip, chunked, + checks + overlay video.

    .venv-perception\\Scripts\\python.exe scripts\\perception\\track_objects.py [--config configs/perception/objects_113656.yaml]
-> data/perception/<pair>/masks_<object>.npz (packed bool masks at --height, 15 fps), outputs/p2_4/objects_report.json,
   outputs/p2_4/objects_<pair>.mp4
Checks: tracked backward, each mask must land on its own part in the rack (box IoU with check_start); no mask
jumps; the two parts' masks must not overlap (identity swap).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from cta_learn import REPO_ROOT
from cta_learn.perception.objects import (box_iou, extract_frames, jump_frames, mask_bbox, merge_tracks, save_masks,
                                         track_object)
from cta_learn.video.ingest import ffbin

MODELS = REPO_ROOT / "data" / "pretrained" / "perception"
OUT = REPO_ROOT / "outputs" / "p2_4"
COLORS = {"profile_short": (255, 140, 0), "profile_long": (0, 200, 255)}  # RGB


def overlay(frames: list[Path], masks: dict[str, np.ndarray], fps: int, dst: Path) -> None:
    h, w = cv2.imread(str(frames[0])).shape[:2]
    ff = subprocess.Popen([ffbin("ffmpeg"), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "22", "-pix_fmt", "yuv420p", str(dst)],
                          stdin=subprocess.PIPE)
    for i, f in enumerate(frames):
        img = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
        for name, m in masks.items():
            mk = m[i]
            col = np.array(COLORS.get(name, (255, 0, 255)))
            img[mk] = (0.45 * img[mk] + 0.55 * col).astype(np.uint8)
            cnts, _ = cv2.findContours(mk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, tuple(int(c) for c in col), 2)
        y = 28
        for name in masks:
            cv2.putText(img, name, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, tuple(int(c) for c in COLORS.get(name, (255, 0, 255))), 2)
            y += 26
        cv2.putText(img, f"{i / fps:5.1f}s", (w - 110, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        ff.stdin.write(img.tobytes())
    ff.stdin.close()
    ff.wait()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "perception" / "objects_113656.yaml")
    ap.add_argument("--max_jump_px", type=float, default=60.0, help="at the working height, per frame")
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    pair, view, fps, H = cfg["pair"], cfg["view"], cfg["fps"], cfg["height"]
    s = H / 1080.0
    d = REPO_ROOT / "data" / "perception" / pair
    frames = extract_frames(REPO_ROOT / "data" / "clips" / pair / f"{view}.mp4", d / f"frames_{view}_{fps}fps", fps, H)
    print(f"[objects] {pair} {view}: {len(frames)} frames at {fps} fps, {H}p")

    from sam2.build_sam import build_sam2_video_predictor
    pred = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_s.yaml", str(MODELS / "sam2.1_hiera_small.pt"), device="cuda")
    masks, report = {}, {"pair": pair, "view": view, "fps": fps, "height": H, "objects": {}}
    for name, o in cfg["objects"].items():
        t0 = time.time()
        pis = [int(round(p["t"] * fps)) for p in o["prompts"]]
        tracks = [track_object(pred, frames, pi, np.array(p["box"], float) * s, chunk=cfg["chunk"]) for pi, p in zip(pis, o["prompts"])]
        m, src = merge_tracks(tracks, pis)
        masks[name] = m
        save_masks(d / f"masks_{name}.npz", m, fps)
        ci = int(round(o["check_start"]["t"] * fps))
        bb = mask_bbox(m[ci])
        area = m.reshape(len(m), -1).sum(1)
        check_box = np.array(o["check_start"]["box"], float)
        at_start = [box_iou(b / s, check_box) > 0.3 if (b := mask_bbox(mk)) is not None else False for mk in m]
        gaps = np.flatnonzero(area == 0)
        r = {"prompt_frames": pis, "seconds": round(time.time() - t0, 1),
             "frames_with_mask": int((area > 0).sum()), "frames": len(m),
             "frames_per_track": {str(k): int((src == k).sum()) for k in range(len(tracks))},
             "hidden_s": [round(g / fps, 2) for g in gaps],
             "check_start_is_a_prompt": any(abs(p["t"] - o["check_start"]["t"]) < 1e-6 for p in o["prompts"]),
             "start_mask_bbox_px": None if bb is None else (bb / s).round(1).tolist(),
             "start_iou_with_expected": 0.0 if bb is None else box_iou(bb / s, check_box),
             "last_time_at_start_place_s": round(max(np.flatnonzero(at_start)) / fps, 2) if any(at_start) else None,
             "jump_frames": jump_frames(m, a.max_jump_px),
             "area_px_min_median_max": [int(area[area > 0].min()), int(np.median(area[area > 0])), int(area.max())] if (area > 0).any() else None}
        report["objects"][name] = r
        hid = f"{r['hidden_s'][0]}-{r['hidden_s'][-1]} s" if gaps.size else "none"
        print(f"[objects] {name}: {r['frames_with_mask']}/{r['frames']} frames with a mask (hidden: {len(gaps)} frames, {hid}); "
              f"start IoU {r['start_iou_with_expected']:.2f}{' (prompt)' if r['check_start_is_a_prompt'] else ''}; "
              f"last seen at its start place {r['last_time_at_start_place_s']} s; jumps at "
              f"{[round(j / fps, 1) for j in r['jump_frames']]} s ({r['seconds']} s)")
    names = list(masks)
    if len(names) == 2:
        inter = (masks[names[0]] & masks[names[1]]).reshape(len(frames), -1).sum(1)
        small = np.minimum(masks[names[0]].reshape(len(frames), -1).sum(1), masks[names[1]].reshape(len(frames), -1).sum(1))
        frac = np.where(small > 0, inter / np.maximum(small, 1), 0.0)
        report["overlap_max_fraction"] = float(frac.max())
        report["frames_overlap_gt_20pct"] = [round(i / fps, 2) for i in np.flatnonzero(frac > 0.2)]
        print(f"[objects] overlap between the two masks: max {frac.max():.1%} of the smaller one; "
              f"frames > 20 %: {len(report['frames_overlap_gt_20pct'])}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "objects_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    overlay(frames, masks, fps, OUT / f"objects_{pair}.mp4")
    print(f"[objects] overlay -> {OUT / f'objects_{pair}.mp4'}")


if __name__ == "__main__":
    main()
