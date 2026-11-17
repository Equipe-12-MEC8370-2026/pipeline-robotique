r"""P3.6: side-by-side video, human demonstration (CTA video) | SO-101 robot in simulation (learned policy).

    .\.venv-learn\Scripts\python.exe scripts\eval\make_human_robot_video.py
        --robot outputs\p3_6\policy_states\videos\trial_000.mp4 --out outputs\p3_6\human_vs_robot.mp4

Two chapters aligned on the task's steps (both sides at real speed; the side that finishes a chapter first holds
its last frame): step 1 = the short part laid flat, step 2 = the long part stood on it.
  human: step times from the TaskSpec (start gesture end, step 0 transport end, end gesture start);
  robot: from the saved part states (eval_profile_policy.py --save_states): step 1 ends when the short part,
         once lifted, is back on the table.
Opening and closing cards (same 1280x480 format as the P0 montage, make_montage.card).
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_montage import FONT, FPS, _esc, card  # noqa: E402

PANEL_W, PANEL_H = 640, 480


def human_chapters(spec: dict, margin: float = 1.0) -> list[tuple[float, float]]:
    """[(t0, t1)] in the human clip for step 1 and step 2."""
    s0 = next(s for s in spec["steps"] if s["index"] == 0)
    t_start = spec["signals"]["start"][1]                                       # end of the start gesture
    t_split = next(p["t"][1] for p in s0["phases"] if p["name"] == "transport") + 0.5  # short part released
    t_end = spec["signals"]["end"][0] + margin                                  # start of the end gesture
    return [(max(0.0, t_start - margin), t_split), (t_split, t_end)]


def robot_split(short_z: np.ndarray, fps: int = FPS, rest_z: float | None = None, lift: float = 0.01,
                settle: float = 0.5) -> float:
    """Time (s) the short part, after being lifted, is back down on the table (+ ``settle``)."""
    z0 = float(short_z[0]) if rest_z is None else rest_z
    up = np.nonzero(short_z > z0 + lift)[0]
    if not len(up):
        raise ValueError("the short part is never lifted")
    down = np.nonzero(short_z[up[0]:] < z0 + 0.002)[0]
    if not len(down):
        raise ValueError("the short part never comes back down")
    return (up[0] + down[0]) / fps + settle


def label(text: str, x: str, y: str, size: int = 20, color: str = "white") -> str:
    return (f"drawtext=fontfile='{FONT}':expansion=none:text='{_esc(text)}':x={x}:y={y}:fontsize={size}"
            f":fontcolor={color}:box=1:boxcolor=black@0.6:boxborderw=6")


def chapter(human: Path, h: tuple, robot: Path, r: tuple, step_text: str, dst: Path, robot_label: str) -> None:
    dh, dr = h[1] - h[0], r[1] - r[0]
    d = max(dh, dr)
    flt = (
        f"[0:v]trim={h[0]:.3f}:{h[1]:.3f},setpts=PTS-STARTPTS,fps={FPS},scale={PANEL_W}:-2,"
        f"pad={PANEL_W}:{PANEL_H}:0:(oh-ih)/2:color=0x1e2430,tpad=stop_mode=clone:stop_duration={d - dh + 0.1:.3f},"
        f"{label('Démonstration humaine (vidéo CTA, vue TPV)', '10', '10')}[h];"
        f"[1:v]trim={r[0]:.3f}:{r[1]:.3f},setpts=PTS-STARTPTS,fps={FPS},crop={PANEL_W}:{PANEL_H}:0:0,"
        f"tpad=stop_mode=clone:stop_duration={d - dr + 0.1:.3f},"
        f"{label(robot_label, '10', '10')}[r];"
        f"[h][r]hstack=inputs=2,trim=0:{d:.3f},{label(step_text, '(w-text_w)/2', 'h-44', 22, 'yellow')}[v]"
    )
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(human), "-i", str(robot), "-filter_complex", flt,
                    "-map", "[v]", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-r", str(FPS), "-an", str(dst)],
                   check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--human", default="data/clips/pair_20260831_113656/tpv.mp4")
    p.add_argument("--taskspec", default="configs/taskspecs/cta_profile_upright_113656.json")
    p.add_argument("--robot", default="outputs/p3_6/policy_states/videos/trial_000.mp4")
    p.add_argument("--out", default="outputs/p3_6/human_vs_robot.mp4")
    p.add_argument("--title", default="Projet CTA : de la vidéo humaine au robot"
                   "|Vidéo humaine (FPV + TPV) -> TaskSpec -> démonstrations générées en simulation -> politique ACT"
                   "|Consigne : poser le bout de profilé court à plat, puis le profilé long debout dessus")
    p.add_argument("--end", default="Politique ACT (SO-101, simulation) : 44/50 essais réussis"
                   "|dans la plage d'entraînement 37/40, hors plage 7/10 (expert scripté : 50/50)"
                   "|Pièces réduites pour le SO-101 (section 25 mm) ; le bout court part couché (D10, D11)")
    p.add_argument("--card_s", type=float, default=5.0)
    p.add_argument("--robot_label", default="Robot SO-101 en simulation : politique ACT")
    a = p.parse_args()

    spec = json.loads(Path(a.taskspec).read_text(encoding="utf-8"))
    robot = Path(a.robot)
    states = np.load(robot.with_suffix(".npz"))
    n_frames = len(states["short_pos"])
    split = robot_split(states["short_pos"][:, 2])
    r_chapters = [(0.0, split), (split, n_frames / FPS)]
    h_chapters = human_chapters(spec)
    steps = ["Étape 1 : poser le bout de profilé court à plat", "Étape 2 : poser le profilé long debout dessus"]
    print(f"human chapters {h_chapters} s; robot chapters {[(round(x, 2), round(y, 2)) for x, y in r_chapters]} s")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        parts = [tmp / "title.mp4"]
        card(a.title.split("|"), a.card_s, parts[0])
        for k, (h, r, s) in enumerate(zip(h_chapters, r_chapters, steps)):
            parts.append(tmp / f"chapter_{k}.mp4")
            chapter(Path(a.human), h, robot, r, s, parts[-1], a.robot_label)
        parts.append(tmp / "end.mp4")
        card(a.end.split("|"), a.card_s, parts[-1])
        lst = tmp / "list.txt"
        lst.write_text("".join(f"file '{q.as_posix()}'\n" for q in parts), encoding="utf-8")
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out)],
                       check=True)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
