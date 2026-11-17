r"""P0.5: build a demo montage from an evaluation run (any Python with ffmpeg on PATH).

    python scripts\eval\make_montage.py --run outputs\p0_5\act_p0 --successes 4 --failures 2
    # demo version: chosen trials, title and closing cards, explicit output
    python scripts\eval\make_montage.py --run outputs\p0_5\act_p0_n25_demo --pick 0 45 47 11 ^
        --title "..." --end "..." --out outputs\demo\montage_v1.mp4

Each clip (front | wrist, 1280x480) gets a caption: trial, split, outcome. Default output: <run>/montage.mp4.
Cards: "|" in --title / --end starts a new line.
"""

import argparse
import json
import subprocess
from pathlib import Path

FONT = "C\\:/Windows/Fonts/arial.ttf"
W, H, FPS = 1280, 480, 30
FR_OUTCOME = {"success": "réussi", "never_lifted": "cube jamais soulevé", "cube_off_table": "cube tombé de la table",
              "in_target_no_success": "dans la cible, non relâché", "held_at_timeout": "encore tenu à la fin",
              "placed_outside": "posé hors cible"}


def _esc(s: str) -> str:
    """Escape text for ffmpeg drawtext (inside a single-quoted filter argument)."""
    return s.replace("\\", "\\\\").replace(":", "\\:").replace("'", "’")


def card(lines: list[str], seconds: float, dst: Path) -> None:
    """A dark title card, same format as the clips so the concat can copy streams."""
    n = len(lines)
    draws = []
    for i, line in enumerate(lines):
        size = 34 if i == 0 else 24
        y = f"(h-{n * 44})/2+{i * 44}"
        draws.append(f"drawtext=fontfile='{FONT}':expansion=none:text='{_esc(line)}':x=(w-text_w)/2:y={y}:fontsize={size}:fontcolor=white")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"color=c=0x1e2430:s={W}x{H}:r={FPS}:d={seconds}",
                    "-vf", ",".join(draws), "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-r", str(FPS), str(dst)],
                   check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="outputs/p0_5/act_p0")
    p.add_argument("--successes", type=int, default=4)
    p.add_argument("--failures", type=int, default=2)
    p.add_argument("--pick", type=int, nargs="+", default=None, help="explicit trial ids, in this order")
    p.add_argument("--label", default="ACT policy (P0), SO-101 sim")
    p.add_argument("--title", default=None, help="opening card text ('|' = new line)")
    p.add_argument("--end", default=None, help="closing card text ('|' = new line)")
    p.add_argument("--card_s", type=float, default=4.0)
    p.add_argument("--out", default=None)
    p.add_argument("--fr", action="store_true", help="captions in French")
    a = p.parse_args()
    run = Path(a.run)
    recs = json.loads((run / "report.json").read_text())["records"]
    have = [r for r in recs if (run / "videos" / f"trial_{r['trial']:03d}.mp4").exists()]
    if a.pick:
        by_id = {r["trial"]: r for r in have}
        missing = [t for t in a.pick if t not in by_id]
        if missing:
            raise SystemExit(f"no video for trials {missing}")
        pick = [by_id[t] for t in a.pick]
    else:
        pick = [r for r in have if r["outcome"] == "success"][: a.successes]
        pick += [r for r in have if r["outcome"] != "success"][: a.failures]
    if not pick:
        raise SystemExit("no videos to assemble")
    tmp = run / "montage_parts"
    tmp.mkdir(exist_ok=True)
    parts = []
    if a.title:
        parts.append(tmp / "card_title.mp4")
        card(a.title.split("|"), a.card_s, parts[-1])
    for r in pick:
        src = run / "videos" / f"trial_{r['trial']:03d}.mp4"
        dst = tmp / src.name
        if a.fr:
            split = "hors plage d'entraînement" if r["split"] == "out_of_range" else "dans la plage d'entraînement"
            txt = _esc(f"{a.label} | essai {r['trial']} ({split}) | {FR_OUTCOME.get(r['outcome'], r['outcome'])}")
        else:
            split = "outside training range" if r["split"] == "out_of_range" else "in training range"
            txt = _esc(f"{a.label} | trial {r['trial']} ({split}) | {r['outcome'].replace('_', ' ')}")
        color = "0x2ecc71" if r["outcome"] == "success" else "0xe74c3c"
        vf = (f"drawbox=x=0:y=0:w=iw:h=36:color=black@0.6:t=fill,"
              f"drawtext=fontfile='{FONT}':expansion=none:text='{txt}':x=12:y=9:fontsize=20:fontcolor={color}")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-vf", vf, "-c:v", "libx264",
                        "-crf", "20", "-pix_fmt", "yuv420p", "-r", str(FPS), str(dst)], check=True)
        parts.append(dst)
    if a.end:
        parts.append(tmp / "card_end.mp4")
        card(a.end.split("|"), a.card_s, parts[-1])
    lst = tmp / "list.txt"
    lst.write_text("".join(f"file '{q.resolve().as_posix()}'\n" for q in parts))
    out = Path(a.out) if a.out else run / "montage.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c", "copy", str(out)], check=True)
    print(f"{out} ({len(parts)} parts: {[r['trial'] for r in pick]})")


if __name__ == "__main__":
    main()
