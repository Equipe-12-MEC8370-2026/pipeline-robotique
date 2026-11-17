"""P2.1 — visual/aural check of the FPV/TPV sync: one video, every pair side by side.

Left FPV, right TPV, same frame number burnt in on both, and the two audio tracks MIXED: if the sync were
off by more than a few tens of ms, every knock would be heard twice (echo) and motions would visibly lag.
    .venv-learn\\Scripts\\python.exe scripts\\video\\make_sync_preview.py
-> outputs/p2_1/sync_preview.mp4
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from cta_learn import REPO_ROOT
from cta_learn.video.ingest import ffbin

FONT = "C\\:/Windows/Fonts/arial.ttf"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=Path, default=REPO_ROOT / "data" / "clips")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "p2_1" / "sync_preview.mp4")
    ap.add_argument("--height", type=int, default=540)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    parts = []
    with tempfile.TemporaryDirectory() as tmp:
        for d in sorted(a.clips.glob("pair_*")):
            s = json.loads((d / "sync.json").read_text(encoding="utf-8"))
            title = f"{d.name}   offset {s['sync']['offset_s']:+.3f} s   spread {s['sync']['spread_ms']:.1f} ms"
            lab = lambda txt, y: (f"drawtext=fontfile='{FONT}':text='{txt}':x=12:y={y}:fontsize=26:fontcolor=white:"
                                  "box=1:boxcolor=black@0.6:boxborderw=6")
            fc = (f"[0:v]scale=-2:{a.height},{lab('FPV  frame %{frame_num}', 12)}[l];"
                  f"[1:v]scale=-2:{a.height},{lab('TPV  frame %{frame_num}', 12)}[r];"
                  f"[l][r]hstack=2,{lab(title.replace(':', chr(92) + ':'), a.height - 46)},format=yuv420p[v];"
                  "[0:a][1:a]amix=inputs=2:normalize=1[a]")
            part = Path(tmp) / f"{d.name}.mp4"
            subprocess.run([ffbin("ffmpeg"), "-v", "error", "-y", "-i", str(d / "fpv.mp4"), "-i", str(d / "tpv.mp4"),
                            "-filter_complex", fc, "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", "23",
                            "-preset", "fast", "-c:a", "aac", "-ar", "48000", "-r", "30", str(part)], check=True)
            parts.append(part)
            print(f"[preview] {d.name}")
        lst = Path(tmp) / "list.txt"
        lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
        subprocess.run([ffbin("ffmpeg"), "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                        "-c", "copy", "-movflags", "+faststart", str(a.out)], check=True)
    print(f"[preview] -> {a.out}")


if __name__ == "__main__":
    main()
