"""P2.5 (M3) — start/end gestures on the 6 pairs + the TaskSpec of one demo, checked against the labels.

    .venv-learn\\Scripts\\python.exe scripts\\task\\build_taskspec.py                       (profile task, 113656)
    .venv-learn\\Scripts\\python.exe scripts\\task\\build_taskspec.py --task_cfg configs/tasks/cta_angle_block.yaml
-> configs/taskspecs/<task_id>_<pair suffix>.json (committed reference), outputs/p2_5/segmentation_report[_<pair>].json
Everything task-specific comes from the task config (pair in ``source.pair``, ``task_id``, ``instruction``, part
sizes, masses, descriptions): no code per task. The instruction is written (or validated) by a human.
Gates: TaskSpec valid + JSON round trip; steps within 1.5 s of the hand labels (when the labels name the object of
each phase); start/end gestures found on the task pairs (detected interval overlaps the labelled one by >= 50 %).
"""
from __future__ import annotations

import argparse
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
FIRST_PAIR = "pair_20260831_113656"  # its report keeps the original name
TOL_S = 1.5


def overlap_frac(det, lab) -> float:
    if det is None:
        return 0.0
    inter = max(0.0, min(det[1], lab[1]) - max(det[0], lab[0]))
    return inter / max(det[1] - det[0], 1e-6)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_cfg", type=Path, default=CONFIGS_DIR / "tasks" / "cta_profile_upright.yaml")
    ap.add_argument("--out", type=Path, default=None, help="default configs/taskspecs/<task_id>_<pair suffix>.json")
    a = ap.parse_args()
    task = yaml.safe_load(a.task_cfg.read_text(encoding="utf-8"))
    pair = task["source"]["pair"]
    labels = yaml.safe_load((CONFIGS_DIR / "tasks" / "cta_demos_labels.yaml").read_text(encoding="utf-8"))
    report, ok, n_sig, n_sig_ok = {"tolerance_s": TOL_S, "pairs": {}}, True, 0, 0
    for p_, lab in labels["pairs"].items():
        tp, fp = HandTrack.load(PERC / p_ / "hands_tpv.npz"), HandTrack.load(PERC / p_ / "hands_fpv.npz")
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
        report["pairs"][p_] = r
        print(f"[seg] {p_}: start {sig['start']} (label {ph[0]['t']}), end {sig['end']} (label {ph[-1]['t']})"
              + ("  [no task: reference only]" if no_task else f"  overlap {r['start_overlap']:.0%}/{r['end_overlap']:.0%} "
                                                              f"{'PASS' if r['pass'] else 'FAIL'}"))

    if pair not in report["pairs"]:  # a new video (interface upload): not in the hand labels
        tp, fp = HandTrack.load(PERC / pair / "hands_tpv.npz"), HandTrack.load(PERC / pair / "hands_fpv.npz")
        report["pairs"][pair] = {"signals": start_end_signals(tp.kp_px, fp.kp_px, fps=30.0), "no_task": False}
    # TaskSpec of the demo named in the task config
    oj_path = PERC / pair / "objects.json"
    oj = json.loads(oj_path.read_text(encoding="utf-8"))
    fp = HandTrack.load(PERC / pair / "hands_fpv.npz")
    sig = report["pairs"][pair]["signals"]
    sizes = {n: o["size_m"] for n, o in task["objects"].items()}
    masses = {n: o["mass_kg"] for n, o in task["objects"].items()}
    calib = REPO_ROOT / "data" / "calib" / "tpv.json"
    setup = json.loads(calib.read_text(encoding="utf-8"))["pair_to_setup"][pair]
    source = {"pair": pair, "views": ["tpv", "fpv"], "objects_json_sha256_16": sha(oj_path),
              "calibration": f"data/calib/tpv.json (set-up {setup})", "calib_sha256_16": sha(calib),
              "parts": f"{rel(a.task_cfg)} ({task.get('parts_note', 'approximate dims, light sim masses, D9')})",
              "perception": "MediaPipe Hand Landmarker 1.2.3 + SAM 2.1 small (facebookresearch/sam2@2b90b9f5)",
              "built_by": "scripts/task/build_taskspec.py"}
    spec = build_taskspec(oj, sizes, masses, fp.kp_px, fp.t_s, sig, task["task_id"], task["instruction"], source,
                          {n: o.get("description", "") for n, o in task["objects"].items()})
    dst = a.out or CONFIGS_DIR / "taskspecs" / f"{task['task_id']}_{pair.rsplit('_', 1)[-1]}.json"
    spec.save(dst)
    back = TaskSpec.load(dst)
    rt = back == spec
    # steps vs labels (only when the labels name the object handled in each phase)
    tph = [p for p in labels["pairs"].get(pair, {}).get("phases", []) if p.get("object")]
    steps_rep = []
    for s in spec.steps:
        lp = [p["t"] for p in tph if p["object"] == s.object]
        r = {"object": s.object, "grasp_release": [s.t_grasp, s.t_release], "hand": s.hand,
             "hand_confidence": s.hand_confidence, "relation": s.goal.relation.type, "posture": s.goal.pose.posture}
        if lp:
            la, lb = min(p[0] for p in lp), max(p[1] for p in lp)
            e0, e1 = abs(s.t_grasp - la), abs(s.t_release - lb)
            r.update({"label": [la, lb], "errors_s": [round(e0, 2), round(e1, 2)], "pass": bool(max(e0, e1) <= TOL_S)})
            ok &= r["pass"]
            lab_txt = f"vs label [{la}, {lb}] -> errors {e0:.2f} / {e1:.2f} s"
        else:
            r["label"] = None
            lab_txt = "no object label to compare"
        steps_rep.append(r)
        print(f"[spec] step {s.index} {s.object}: grasp {s.t_grasp} / release {s.t_release} s {lab_txt}; hand {s.hand} "
              f"({s.hand_confidence:.0%}); goal {s.goal.pose.posture} {s.goal.relation.type}"
              f"{' ' + str(s.goal.relation.ref) + ' offset ' + str(s.goal.relation.offset_xy_m) if s.goal.relation.ref else ''}")
    report["taskspec"] = {"pair": pair, "path": rel(dst), "valid": True, "json_roundtrip": rt, "steps": steps_rep,
                          "labels_checked": any(r["label"] for r in steps_rep)}
    report["taskspec_pass"] = bool(ok and rt)
    report["signals_pass"] = f"{n_sig_ok}/{n_sig}"
    OUT.mkdir(parents=True, exist_ok=True)
    name = "segmentation_report.json" if pair == FIRST_PAIR else f"segmentation_report_{pair}.json"
    (OUT / name).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[spec] -> {dst} (round trip {'OK' if rt else 'FAIL'}); TaskSpec gate {'PASS' if report['taskspec_pass'] else 'FAIL'}; "
          f"start/end gestures {report['signals_pass']} task pairs")


if __name__ == "__main__":
    main()
