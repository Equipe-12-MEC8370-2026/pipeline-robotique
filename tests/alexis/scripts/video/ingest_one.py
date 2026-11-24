r"""Sync ONE FPV/TPV pair by audio (GCC-PHAT) and export the synced 30 fps clips — any file names (interface uploads).

    .venv-learn\Scripts\python.exe scripts\video\ingest_one.py --fpv <fpv.mp4> --tpv <tpv.mp4> --pair <pair_id>
-> data/clips/<pair_id>/{fpv.mp4, tpv.mp4, sync.json}. Same method as ingest_pairs.py, without the file-name clock
prior (an upload has no reliable name): gate = >= 3 confident windows agreeing within one frame. Exit 1 if it fails.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cta_learn import REPO_ROOT
from cta_learn.video.ingest import common_span, export_clip, load_audio, probe, sha256
from cta_learn.video.sync import estimate_offset

SR = 16000
GATE_MS = 33.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fpv", type=Path, required=True)
    ap.add_argument("--tpv", type=Path, required=True)
    ap.add_argument("--pair", required=True)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "clips")
    ap.add_argument("--fps", type=int, default=30)
    a = ap.parse_args()
    pf, pt = probe(a.fpv), probe(a.tpv)
    af, at = load_audio(a.fpv, SR), load_audio(a.tpv, SR)
    r = estimate_offset(af, at, SR, fpv_audio_start_s=pf["audio"]["start_s"], tpv_audio_start_s=pt["audio"]["start_s"])
    fs, ts, dur = common_span(r.offset_s, pf["duration_s"], pt["duration_s"])
    passed = len(r.window_offsets_s) >= 3 and r.spread_ms <= GATE_MS and dur > 1.0
    print(f"[sync] {a.pair}: offset {r.offset_s:+.4f} s | windows {len(r.window_offsets_s)}/{r.n_windows_total} "
          f"spread {r.spread_ms:.1f} ms | common {dur:.1f} s | {'PASS' if passed else 'FAIL'}", flush=True)
    d = a.out / a.pair
    d.mkdir(parents=True, exist_ok=True)
    rec = {"pair_id": a.pair, "passed": passed, "convention": "offset_s = t_tpv - t_fpv (container timelines)",
           "sync": r.to_dict(), "name_offset_s": None, "fpv_start_s": fs, "tpv_start_s": ts, "duration_s": dur, "fps": a.fps,
           "sources": {k: {"path": str(src), "sha256": sha256(src), "probe": pr} for k, src, pr in (("fpv", a.fpv, pf), ("tpv", a.tpv, pt))}}
    if passed:
        export_clip(a.fpv, d / "fpv.mp4", fs, dur, a.fps)
        export_clip(a.tpv, d / "tpv.mp4", ts, dur, a.fps)
        rec["clips"] = {k: probe(d / f"{k}.mp4") for k in ("fpv", "tpv")}
        print(f"[sync] clips -> {d}", flush=True)
    (d / "sync.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
