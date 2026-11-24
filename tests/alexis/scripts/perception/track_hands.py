"""P2.4b — hand keypoints on the synced CTA clips (TPV and FPV) + gate report + overlay video.

    .venv-perception\\Scripts\\python.exe scripts\\perception\\track_hands.py [--pairs pair_... ] [--overlay pair_...]
-> data/perception/<pair>/hands_{tpv,fpv}.npz, outputs/p2_4/hands_report.json, outputs/p2_4/hands_<pair>.mp4
Gate: >= 90 % of frames with at least one hand during the labelled phases (configs/tasks/cta_demos_labels.yaml).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import av
import cv2
import numpy as np
import yaml

from cta_learn import CONFIGS_DIR, REPO_ROOT
from cta_learn.perception.hands import EDGES, HandTrack, detection_rate, track_hands
from cta_learn.video.ingest import ffbin

CLIPS = REPO_ROOT / "data" / "clips"
PERC = REPO_ROOT / "data" / "perception"
OUT = REPO_ROOT / "outputs" / "p2_4"
MODEL = REPO_ROOT / "data" / "pretrained" / "perception" / "hand_landmarker.task"
GATE = 0.90


def overlay(pair: str, height: int = 540) -> Path:
    tracks = {v: HandTrack.load(PERC / pair / f"hands_{v}.npz") for v in ("tpv", "fpv")}
    dst = OUT / f"hands_{pair}.mp4"
    cols = [(255, 80, 80), (80, 160, 255)]  # slot 0 ("Left" label) red-ish, slot 1 blue-ish (RGB)
    conts = [av.open(str(CLIPS / pair / f"{v}.mp4")) for v in ("tpv", "fpv")]
    gens = [c.decode(c.streams.video[0]) for c in conts]
    w = int(1920 * height / 1080)
    ff = subprocess.Popen([ffbin("ffmpeg"), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{2 * w}x{height}",
                           "-r", "30", "-i", "-", "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", str(dst)], stdin=subprocess.PIPE)
    for i, frames in enumerate(zip(*gens)):
        panels = []
        for (v, tr), fr in zip(tracks.items(), frames):
            img = fr.to_ndarray(format="rgb24")
            if i < len(tr.t_s):
                for s in range(2):
                    p = tr.kp_px[i, s]
                    if np.isnan(p[0, 0]):
                        continue
                    for a, b in EDGES:
                        cv2.line(img, tuple(p[a].astype(int)), tuple(p[b].astype(int)), cols[s], 3, cv2.LINE_AA)
                    for q in p:
                        cv2.circle(img, tuple(q.astype(int)), 5, (255, 255, 255), -1)
            img = cv2.resize(img, (w, height), interpolation=cv2.INTER_AREA)
            cv2.putText(img, f"{v.upper()} {i / 30:5.1f}s hands {int(tracks[v].n_hands()[min(i, len(tracks[v].t_s) - 1)])}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
            panels.append(img)
        ff.stdin.write(np.hstack(panels).tobytes())
    ff.stdin.close()
    ff.wait()
    for c in conts:
        c.close()
    return dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="*", default=None)
    ap.add_argument("--views", nargs="*", default=["tpv", "fpv"])
    ap.add_argument("--overlay", nargs="*", default=["pair_20260831_113656"])
    ap.add_argument("--skip_existing", action="store_true")
    a = ap.parse_args()
    labels = yaml.safe_load((CONFIGS_DIR / "tasks" / "cta_demos_labels.yaml").read_text(encoding="utf-8"))["pairs"]
    pairs = a.pairs or sorted(p.name for p in CLIPS.glob("pair_*"))
    OUT.mkdir(parents=True, exist_ok=True)
    rep_path = OUT / "hands_report.json"
    report = json.loads(rep_path.read_text(encoding="utf-8")) if rep_path.exists() else {}
    for pair in pairs:
        if pair in labels:
            intervals = [tuple(ph["t"]) for ph in labels[pair]["phases"]]
        else:  # a new video (interface upload): no hand labels, the whole clip counts
            intervals = [(0.0, 1e9)]
        for v in a.views:
            dst = PERC / pair / f"hands_{v}.npz"
            t0 = time.time()
            if a.skip_existing and dst.exists():
                tr = HandTrack.load(dst)
            else:
                tr = track_hands(CLIPS / pair / f"{v}.mp4", MODEL)
                tr.save(dst)
            n = tr.n_hands()
            r = {"frames": int(len(n)), "seconds": round(time.time() - t0, 1),
                 "rate_any_all_frames": float((n >= 1).mean()),
                 "rate_any_in_phases": detection_rate(n, tr.t_s, intervals, 1),
                 "rate_two_in_phases": detection_rate(n, tr.t_s, intervals, 2),
                 "per_phase_any": [{"t": list(iv), "rate": detection_rate(n, tr.t_s, [iv], 1)} for iv in intervals],
                 "mean_score": float(tr.score[tr.score > 0].mean()) if (tr.score > 0).any() else 0.0}
            r["pass"] = bool(r["rate_any_in_phases"] >= GATE) if pair in labels else None  # no labels: no gate
            report.setdefault(pair, {})[v] = r
            print(f"[hands] {pair} {v}: {r['frames']} frames in {r['seconds']} s | >=1 hand in phases "
                  f"{r['rate_any_in_phases']:.1%} (2 hands {r['rate_two_in_phases']:.1%}) | {'PASS' if r['pass'] else 'FAIL' if r['pass'] is not None else 'no hand labels: rate over the whole clip'}")
            rep_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for pair in a.overlay or []:
        if all((PERC / pair / f"hands_{v}.npz").exists() for v in ("tpv", "fpv")):
            print(f"[hands] overlay -> {overlay(pair)}")


if __name__ == "__main__":
    main()
