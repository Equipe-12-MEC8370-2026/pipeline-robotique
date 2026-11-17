"""P2.1 (M1) — pair the CTA FPV/TPV videos, sync them by audio, export synced constant-30-fps clips.

Run from cta-learn with .venv-learn (only numpy + the ffmpeg CLI are needed):
    .venv-learn\\Scripts\\python.exe scripts\\video\\ingest_pairs.py
Writes data/clips/<pair_id>/{fpv.mp4, tpv.mp4, sync.json} and data/clips/catalog.json.
The source videos are only read.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from cta_learn import REPO_ROOT
from cta_learn.video.ingest import common_span, export_clip, load_audio, pair_by_name, probe, sha256
from cta_learn.video.sync import estimate_offset

SR = 16000
GATE_MS = 33.0  # one frame at 30 fps (P2 exit test)
PRIOR_TOL_S = 1.5  # |audio offset - file-name offset| allowed (names have 1 s resolution)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fpv_dir", type=Path, default=REPO_ROOT.parent / "Vidéos FPV")
    ap.add_argument("--tpv_dir", type=Path, default=REPO_ROOT.parent / "Vidéos TPV")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "clips")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--only", nargs="*", default=None, help="pair ids to process (default: all)")
    ap.add_argument("--no_export", action="store_true", help="sync only, do not write clips")
    a = ap.parse_args()

    pairs, unmatched = pair_by_name(sorted(a.fpv_dir.glob("*.mp4")), sorted(a.tpv_dir.glob("*.mp4")))
    print(f"[ingest] {len(pairs)} pairs; unmatched: {[u.name for u in unmatched]}")
    catalog = {"fps": a.fps, "audio_sr": SR, "gate_ms": GATE_MS, "pairs": [], "unmatched": [str(u) for u in unmatched]}
    ok_all = True
    for p in pairs:
        if a.only and p.pair_id not in a.only:
            continue
        t0 = time.time()
        pf, pt = probe(p.fpv), probe(p.tpv)
        af, at = load_audio(p.fpv, SR), load_audio(p.tpv, SR)
        r = estimate_offset(af, at, SR, fpv_audio_start_s=pf["audio"]["start_s"], tpv_audio_start_s=pt["audio"]["start_s"],
                            prior_s=p.name_offset_s)
        fs, ts, dur = common_span(r.offset_s, pf["duration_s"], pt["duration_s"])
        # Gate: >= 3 independent confident windows agree with the global offset within one frame, and the
        # offset matches the file-name clocks (1 s resolution + start latency). The global peak ratio is only
        # informative: repetitive sounds (filing strokes) create secondary peaks without making the match wrong.
        passed = (len(r.window_offsets_s) >= 3 and r.spread_ms <= GATE_MS
                  and abs(r.offset_s - p.name_offset_s) <= PRIOR_TOL_S)
        ok_all &= passed
        print(f"[{p.pair_id}] offset {r.offset_s:+.4f} s (name prior {p.name_offset_s:+.0f} s) peak ratio {r.peak_ratio:.1f} | "
              f"windows {len(r.window_offsets_s)}/{r.n_windows_total} spread {r.spread_ms:.1f} ms "
              f"drift {r.drift_ms_per_min:+.1f} ms/min | common {dur:.1f} s | {'PASS' if passed else 'FAIL'}")
        rec = {
            "pair_id": p.pair_id, "passed": passed,
            "convention": "offset_s = t_tpv - t_fpv (container timelines); clips start at fpv_start_s / tpv_start_s",
            "sync": r.to_dict(), "name_offset_s": p.name_offset_s,
            "fpv_start_s": fs, "tpv_start_s": ts, "duration_s": dur, "fps": a.fps,
            "audio_samples": {"fpv": len(af), "tpv": len(at)},
            "sources": {k: {"path": str(src), "sha256": sha256(src), "probe": pr}
                        for k, src, pr in (("fpv", p.fpv, pf), ("tpv", p.tpv, pt))},
        }
        d = a.out / p.pair_id
        d.mkdir(parents=True, exist_ok=True)
        if not a.no_export:
            export_clip(p.fpv, d / "fpv.mp4", fs, dur, a.fps)
            export_clip(p.tpv, d / "tpv.mp4", ts, dur, a.fps)
            rec["clips"] = {k: probe(d / f"{k}.mp4") for k in ("fpv", "tpv")}
        (d / "sync.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        catalog["pairs"].append({k: rec[k] for k in ("pair_id", "passed", "duration_s", "fpv_start_s", "tpv_start_s")}
                                | {"offset_s": r.offset_s, "spread_ms": r.spread_ms, "peak_ratio": r.peak_ratio})
        print(f"    done in {time.time() - t0:.0f} s -> {d}")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "catalog.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    n_ok = sum(c["passed"] for c in catalog["pairs"])
    print(f"[ingest] {n_ok}/{len(catalog['pairs'])} pairs pass the {GATE_MS:.0f} ms sync gate")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
