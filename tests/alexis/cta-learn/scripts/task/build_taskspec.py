"""P2.5 (M3) — start/end gestures on the 6 pairs + the TaskSpec of the target demo, checked against the labels.

    .venv-learn\\Scripts\\python.exe scripts\\task\\build_taskspec.py
-> configs/taskspecs/cta_profile_upright_113656.json (committed reference), outputs/p2_5/segmentation_report.json
Gates: TaskSpec valid + JSON round trip; steps within 1.5 s of the hand labels; start/end gestures found on the
task pairs (detected interval overlaps the labelled one by >= 50 %).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from cta_learn import CONFIGS_DIR, REPO_ROOT
from cta_learn.core.taskspec import TaskSpec
from cta_learn.perception.hands import HandTrack
from cta_learn.task.build import build_taskspec
from cta_learn.task.segment import start_end_signals

PERC = REPO_ROOT / "data" / "perception"
OUT = REPO_ROOT / "outputs" / "p2_5"
TARGET = "pair_20260831_113656"
TOL_S = 1.5


def overlap_frac(det, lab) -> float:
    if det is None:
        return 0.0
    inter = max(0.0, min(det[1], lab[1]) - max(det[0], lab[0]))
    return inter / max(det[1] - det[0], 1e-6)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main() -> None:
    labels = yaml.safe_load((CONFIGS_DIR / "tasks" / "cta_demos_labels.yaml").read_text(encoding="utf-8"))
    report, ok, n_sig, n_sig_ok = {"tolerance_s": TOL_S, "pairs": {}}, True, 0, 0
    for pair, lab in labels["pairs"].items():
        tp, fp = HandTrack.load(PERC / pair / "hands_tpv.npz"), HandTrack.load(PERC / pair / "hands_fpv.npz")
        sig = start_end_signals(tp.kp_px, fp.kp_px, fps=30.0)
        ph = lab["phases"]
        no_task = len(ph) == 1
        r = {"signals": sig, "no_task": no_task}
        if not no_task:
            r["start_overlap"] = overlap_frac(sig["start"], ph[0]["t"])
            r["end_overlap"] = overlap_frac(sig["end"], ph[-1]["t"])
            r["pass"] = bool(r["start_overlap"] >= 0.5 and r["end_overlap"] >= 0.5)
            n_sig += 1
            n_sig_ok += r["pass"]
        report["pairs"][pair] = r
        print(f"[seg] {pair}: start {sig['start']} (label {ph[0]['t']}), end {sig['end']} (label {ph[-1]['t']})"
              + ("  [no task: reference only]" if no_task else f"  overlap {r['start_overlap']:.0%}/{r['end_overlap']:.0%} "
                                                              f"{'PASS' if r['pass'] else 'FAIL'}"))

    # TaskSpec of the target demo
    task = yaml.safe_load((CONFIGS_DIR / "tasks" / "cta_profile_upright.yaml").read_text(encoding="utf-8"))
    oj_path = PERC / TARGET / "objects.json"
    oj = json.loads(oj_path.read_text(encoding="utf-8"))
    fp = HandTrack.load(PERC / TARGET / "hands_fpv.npz")
    sig = report["pairs"][TARGET]["signals"]
    sizes = {n: o["size_m"] for n, o in task["objects"].items()}
    masses = {n: o["mass_kg"] for n, o in task["objects"].items()}
    source = {"pair": TARGET, "views": ["tpv", "fpv"], "objects_json_sha256_16": sha(oj_path),
              "calibration": "data/calib/tpv.json (set-up A)", "calib_sha256_16": sha(REPO_ROOT / "data" / "calib" / "tpv.json"),
              "parts": "configs/tasks/cta_profile_upright.yaml (approximate dims, light sim masses, D9)",
              "perception": "MediaPipe Hand Landmarker 1.2.3 + SAM 2.1 small (facebookresearch/sam2@2b90b9f5)",
              "built_by": "scripts/task/build_taskspec.py"}
    spec = build_taskspec(oj, sizes, masses, fp.kp_px, fp.t_s, sig, "cta_profile_upright",
                          {"fr": "Pose le bout de profilé court à plat, puis pose le profilé long debout dessus.",
                           "en": "Lay the short aluminium profile flat, then stand the long profile upright on top of it."},
                          source, {"profile_short": "aluminium T-slot profile 40x40, 80 mm",
                                   "profile_long": "aluminium T-slot profile 40x40, ~150 mm"})
    dst = CONFIGS_DIR / "taskspecs" / "cta_profile_upright_113656.json"
    spec.save(dst)
    back = TaskSpec.load(dst)
    rt = back == spec
    # steps vs labels
    tph = [p for p in labels["pairs"][TARGET]["phases"] if p.get("object")]
    steps_rep = []
    for s in spec.steps:
        lp = [p["t"] for p in tph if p["object"] == s.object]
        la, lb = min(p[0] for p in lp), max(p[1] for p in lp)
        e0, e1 = abs(s.t_grasp - la), abs(s.t_release - lb)
        steps_rep.append({"object": s.object, "grasp_release": [s.t_grasp, s.t_release], "label": [la, lb],
                          "errors_s": [round(e0, 2), round(e1, 2)], "pass": bool(max(e0, e1) <= TOL_S), "hand": s.hand,
                          "hand_confidence": s.hand_confidence, "relation": s.goal.relation.type})
        ok &= steps_rep[-1]["pass"]
        print(f"[spec] step {s.index} {s.object}: grasp {s.t_grasp} / release {s.t_release} s vs label [{la}, {lb}] -> "
              f"errors {e0:.2f} / {e1:.2f} s; hand {s.hand} ({s.hand_confidence:.0%}); goal {s.goal.pose.posture} "
              f"{s.goal.relation.type}{' ' + str(s.goal.relation.ref) + ' offset ' + str(s.goal.relation.offset_xy_m) if s.goal.relation.ref else ''}")
    report["taskspec"] = {"path": str(dst.relative_to(REPO_ROOT)), "valid": True, "json_roundtrip": rt, "steps": steps_rep}
    report["taskspec_pass"] = bool(ok and rt)
    report["signals_pass"] = f"{n_sig_ok}/{n_sig}"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "segmentation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[spec] -> {dst} (round trip {'OK' if rt else 'FAIL'}); TaskSpec gate {'PASS' if report['taskspec_pass'] else 'FAIL'}; "
          f"start/end gestures {report['signals_pass']} task pairs")


if __name__ == "__main__":
    main()
