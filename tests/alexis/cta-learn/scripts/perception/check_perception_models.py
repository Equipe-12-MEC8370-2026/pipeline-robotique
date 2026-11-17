"""P2.4a — smoke test of the perception stack in .venv-perception (offline, local models only).

    .venv-perception\\Scripts\\python.exe scripts\\perception\\check_perception_models.py
1. MediaPipe Hand Landmarker (VIDEO mode, CPU) on 2 s of pair 114518 (hands flat on the table), TPV and FPV.
2. SAM 2.1 small, image predictor on GPU: box prompt around the near aluminium plate of set-up B.
3. SAM 2.1 small, video predictor: same box, propagated over 30 frames at 540p.
Writes outputs/p2_4/models_check.json and an overlay image.
"""
from __future__ import annotations

import json
import time

import av
import cv2
import numpy as np
import torch

from cta_learn import REPO_ROOT

MODELS = REPO_ROOT / "data" / "pretrained" / "perception"
CLIP = REPO_ROOT / "data" / "clips" / "pair_20260831_114518"
OUT = REPO_ROOT / "outputs" / "p2_4"


def read_frames(path, t0: float, n: int, height: int | None = None) -> list[np.ndarray]:
    out = []
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        c.seek(int(t0 / s.time_base), stream=s)
        for fr in c.decode(s):
            if float(fr.pts * s.time_base) < t0 - 1e-3:
                continue
            img = fr.to_ndarray(format="rgb24")
            if height:
                img = cv2.resize(img, (int(img.shape[1] * height / img.shape[0]), height), interpolation=cv2.INTER_AREA)
            out.append(img)
            if len(out) == n:
                break
    return out


def check_hands() -> dict:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

    res = {"mediapipe": mp.__version__}
    for view in ("tpv", "fpv"):
        opts = HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(MODELS / "hand_landmarker.task")),
                                     running_mode=RunningMode.VIDEO, num_hands=2)
        frames = read_frames(CLIP / f"{view}.mp4", 20.0, 60)
        counts, t = [], time.time()
        with HandLandmarker.create_from_options(opts) as lm:
            for i, f in enumerate(frames):
                r = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(f)), int(i * 1000 / 30))
                counts.append(len(r.hand_landmarks))
        res[view] = {"frames": len(frames), "ms_per_frame": round((time.time() - t) * 1000 / len(frames), 1),
                     "hands_per_frame_mean": float(np.mean(counts)), "frames_with_2_hands": int(np.sum(np.array(counts) == 2))}
    return res


def check_sam2() -> dict:
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    cfg, ckpt = "configs/sam2.1/sam2.1_hiera_s.yaml", str(MODELS / "sam2.1_hiera_small.pt")
    img = read_frames(CLIP / "tpv.mp4", 0.5, 1)[0]
    box = np.array([380, 755, 495, 850], float)  # near aluminium plate in set-up B (full-res px)
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    pred = SAM2ImagePredictor(build_sam2(cfg, ckpt, device="cuda"))
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred.set_image(img)
        masks, scores, _ = pred.predict(box=box, multimask_output=False)
    m = masks[0].astype(bool)
    res = {"image": {"seconds": round(time.time() - t, 1), "score": float(scores[0]), "mask_px": int(m.sum()),
                     "mask_bbox": [int(v) for v in cv2.boundingRect(m.astype(np.uint8))],
                     "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}}
    ov = img.copy()
    ov[m] = (0.5 * ov[m] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
    cv2.rectangle(ov, tuple(box[:2].astype(int)), tuple(box[2:].astype(int)), (0, 255, 0), 2)
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / "sam2_check.jpg"), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
    del pred
    torch.cuda.empty_cache()

    # video predictor: 30 frames at 540p written as JPEGs (the format SAM 2's loader expects)
    frames_dir = OUT / "sam2_check_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(read_frames(CLIP / "tpv.mp4", 0.5, 30, height=540)):
        cv2.imwrite(str(frames_dir / f"{i:05d}.jpg"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    vp = build_sam2_video_predictor(cfg, ckpt, device="cuda")
    areas = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        st = vp.init_state(video_path=str(frames_dir), offload_video_to_cpu=True, offload_state_to_cpu=True)
        vp.add_new_points_or_box(st, frame_idx=0, obj_id=1, box=box / 2)
        for _, _, logits in vp.propagate_in_video(st):
            areas.append(int((logits[0, 0] > 0).sum()))
    res["video"] = {"frames": len(areas), "seconds": round(time.time() - t, 1), "mask_px_first_last": [areas[0], areas[-1]],
                    "mask_px_rel_std": round(float(np.std(areas) / np.mean(areas)), 3),
                    "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    return res


def main() -> None:
    res = {"torch": torch.__version__, "cuda": torch.cuda.is_available(), "hands": check_hands(), "sam2": check_sam2()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "models_check.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
