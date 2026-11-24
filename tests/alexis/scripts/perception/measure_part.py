"""Measure a part's bounding-box sizes from the TPV masks when no CAD is given (2nd task, 2026-11-18).

    .venv-learn\\Scripts\\python.exe scripts\\perception\\measure_part.py --objects_cfg configs/perception/objects_114110.yaml
        --object angle_block --times 1.0 2.0 13.5 29.5 30.5
-> outputs/p2_4/size_<pair>_<object>.json
The part must rest, without hands on it, at the given times (ideally in two places / orientations: the views then
constrain the three sizes better). Sizes (a, b, c) are found by coordinate descent: each size is moved by
+-10, +-5, +-2 mm and kept when the mean box-fit IoU over the frames rises (each frame fits its own pose and
posture, see perception.poses.fit_box). A box only approximates a part with chamfers or pockets: expect a few mm.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from cta_learn import REPO_ROOT
from cta_learn.perception.objects import load_masks
from cta_learn.perception.poses import fit_box
from cta_learn.video.calib import intrinsics, pose_from_homography, ray_to_height

OUT = REPO_ROOT / "outputs" / "p2_4"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects_cfg", type=Path, required=True)
    ap.add_argument("--object", required=True)
    ap.add_argument("--times", type=float, nargs="+", required=True, help="s, part at rest without hands")
    ap.add_argument("--start", type=float, nargs=3, default=(0.06, 0.06, 0.06), help="initial sizes, m")
    ap.add_argument("--max_iter", type=int, default=12)
    a = ap.parse_args()
    ocfg = yaml.safe_load(a.objects_cfg.read_text(encoding="utf-8"))
    pair, H = ocfg["pair"], ocfg["height"]
    cal = json.loads((REPO_ROOT / "data" / "calib" / "tpv.json").read_text(encoding="utf-8"))
    setup = cal["pair_to_setup"][pair]
    K_full = np.array(cal["setups"][setup]["K"])
    R, t = pose_from_homography(K_full, np.array(cal["setups"][setup]["H_table_to_img"]))
    sc = H / 1080.0
    K = intrinsics(K_full[0, 0] * sc, int(1920 * sc), H)
    masks, fps = load_masks(REPO_ROOT / "data" / "perception" / pair / f"masks_{a.object}.npz")
    frames = [masks[int(round(tt * fps))] for tt in a.times]
    inits = []
    for m in frames:
        ys, xs = np.nonzero(m)
        low = np.argmax(ys)
        inits.append(ray_to_height(K, R, t, (xs[low], ys[low]), 0.0)[:2])

    def score(size):
        fits = [fit_box(m, K, R, t, size, ini, xy_window=0.06) for m, ini in zip(frames, inits)]
        return float(np.mean([f.iou for f in fits])), fits

    size = np.array(a.start, float)
    best, fits = score(size)
    print(f"[size] start {np.round(size * 1000).tolist()} mm -> mean IoU {best:.3f}")
    for it in range(a.max_iter):
        improved = False
        for k in range(3):
            for d in (0.010, -0.010, 0.005, -0.005, 0.002, -0.002):
                cand = size.copy()
                cand[k] += d
                if cand[k] < 0.01:
                    continue
                s, f = score(cand)
                if s > best + 1e-4:
                    best, fits, size, improved = s, f, cand, True
        print(f"[size] iteration {it + 1}: {np.round(np.sort(size)[::-1] * 1000).tolist()} mm -> mean IoU {best:.3f}")
        if not improved:
            break
    out = {"pair": pair, "object": a.object, "times_s": a.times,
           "size_m_longest_first": np.round(np.sort(size)[::-1], 4).tolist(), "mean_iou": round(best, 4),
           "fits": [{"t": tt, "x": round(f.x, 4), "y": round(f.y, 4), "yaw_deg": round(f.yaw_deg, 1), "up_axis": f.up_axis,
                     "iou": round(f.iou, 3)} for tt, f in zip(a.times, fits)],
           "method": "coordinate descent on box sizes, mean box-fit IoU over the frames (scripts/perception/measure_part.py)"}
    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / f"size_{pair}_{a.object}.json"
    dst.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[size] -> {dst}: {out['size_m_longest_first']} m, IoU {best:.3f}; fits {out['fits']}")


if __name__ == "__main__":
    main()
