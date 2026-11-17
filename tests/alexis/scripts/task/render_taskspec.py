"""P2.5 — draw a TaskSpec back onto the demo video (visual check that it tells the demo's story).

    .venv-learn\\Scripts\\python.exe scripts\\task\\render_taskspec.py
-> outputs/p2_5/taskspec_<pair>.mp4: TPV frames (15 fps, 540p) with, for each part, its INITIAL box (thin) until
its grasp and its GOAL box (thick) from its release, plus a timeline of the gestures and phases with a cursor.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from cta_learn import CONFIGS_DIR, REPO_ROOT
from cta_learn.core.taskspec import TaskSpec
from cta_learn.perception.poses import box_corners, box_extents
from cta_learn.video.calib import intrinsics, pose_from_homography, project
from cta_learn.video.ingest import ffbin

EDGES = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
COL = {"profile_short": (0, 140, 255), "profile_long": (255, 200, 0)}  # BGR
PH_COL = {"reach": (180, 180, 180), "grasp": (0, 200, 255), "transport": (0, 160, 0), "place": (200, 120, 0),
          "release": (200, 0, 200), "retreat": (120, 120, 120)}


def draw_box(img, K, R, t, size, pose, col, thick):
    sec, length = min(size), max(size)
    c = project(K, R, t, box_corners(box_extents(sec, length, pose.posture == "upright"), pose.x, pose.y, pose.yaw_deg, pose.z_base))
    for i, j in EDGES:
        cv2.line(img, tuple(c[i].astype(int)), tuple(c[j].astype(int)), col, thick, cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, default=CONFIGS_DIR / "taskspecs" / "cta_profile_upright_113656.json")
    a = ap.parse_args()
    spec = TaskSpec.load(a.spec)
    pair = spec.source["pair"]
    frames = sorted((REPO_ROOT / "data" / "perception" / pair / "frames_tpv_15fps").glob("*.jpg"))
    fps, H = 15, cv2.imread(str(frames[0])).shape[0]
    cal = json.loads((REPO_ROOT / "data" / "calib" / "tpv.json").read_text(encoding="utf-8"))
    st = cal["setups"][cal["pair_to_setup"][pair]]
    K_full = np.array(st["K"])
    R, t = pose_from_homography(K_full, np.array(st["H_table_to_img"]))
    K = intrinsics(K_full[0, 0] * H / 1080, int(1920 * H / 1080), H)
    W = K[0, 2] * 2
    objs = {o.id: o for o in spec.objects}
    steps = {s.object: s for s in spec.steps}
    dur = len(frames) / fps
    bar_h = 70
    dst = REPO_ROOT / "outputs" / "p2_5" / f"taskspec_{pair}.mp4"
    dst.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen([ffbin("ffmpeg"), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{int(W)}x{H + bar_h}",
                           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "22", "-pix_fmt", "yuv420p", str(dst)],
                          stdin=subprocess.PIPE)
    for i, f in enumerate(frames):
        now = i / fps
        img = cv2.imread(str(f))
        for oid, o in objs.items():
            s = steps[oid]
            if now <= s.t_grasp:
                draw_box(img, K, R, t, o.size_m, o.initial, COL[oid], 1)
            if now >= s.t_release:
                draw_box(img, K, R, t, o.size_m, s.goal.pose, COL[oid], 3)
        cv2.putText(img, spec.instruction["en"], (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        bar = np.full((bar_h, int(W), 3), 30, np.uint8)
        X = lambda tt: int(10 + (W - 20) * tt / dur)
        for name, iv in spec.signals.items():
            if iv:
                cv2.rectangle(bar, (X(iv[0]), 6), (X(iv[1]), 20), (90, 90, 255), -1)
                cv2.putText(bar, f"signal {name}", (X(iv[0]) + 2, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        for k, s in enumerate(spec.steps):
            y0 = 26 + 20 * k
            for ph in s.phases:
                cv2.rectangle(bar, (X(ph.t[0]), y0), (max(X(ph.t[1]), X(ph.t[0]) + 1), y0 + 14), PH_COL[ph.name], -1)
            cv2.putText(bar, f"{s.index}: {s.object} ({s.hand} hand) -> {s.goal.pose.posture} {s.goal.relation.type}",
                        (X(s.phases[0].t[0]) + 2, y0 + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.line(bar, (X(now), 0), (X(now), bar_h), (0, 255, 255), 2)
        ff.stdin.write(np.vstack([img, bar]).tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"[render] -> {dst}")


if __name__ == "__main__":
    main()
