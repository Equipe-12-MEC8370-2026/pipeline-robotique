r"""P3.2: demo video of the scripted stacking expert from saved episodes (any Python with numpy + ffmpeg on PATH).

    python scripts\eval\make_expert_demo.py --episodes outputs\p3_2\debug_episodes outputs\p3_2\ood\debug_episodes\episode_000000
        --out outputs\p3_2\demo_expert.mp4
(each --episodes path: one episode folder, or a folder of episodes)

Each episode: front | wrist side by side (1280x480), caption = episode outcome + the expert's current step and
phase (read from data.npz), in French. Opening and closing cards.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_montage import FONT, FPS, H, W, _esc, card  # noqa: E402

from cta_learn.datagen.stack_expert import PHASES  # noqa: E402  (needs PYTHONPATH=src or an installed package)
FR_PHASE = {"ready": "mise en position", "approach": "approche", "descend": "descente", "close": "fermeture de la pince",
            "lift": "levée", "transport": "transport", "lower": "dépose", "release": "ouverture", "clear": "écartement du doigt fixe", "slide": "dégagement latéral", "retreat": "retrait",
            "done": "terminé"}
FR_STEP = {0: "étape 1 : bout court -> couché dans sa zone", 1: "étape 2 : profilé long -> debout sur le bout court"}


def segments(step: np.ndarray, phase: np.ndarray) -> list[tuple[float, float, str]]:
    out, start = [], 0
    for k in range(1, len(step) + 1):
        if k == len(step) or step[k] != step[start] or phase[k] != phase[start]:
            name = PHASES[int(phase[start])]
            text = FR_PHASE[name] if name in ("ready", "done") else f"{FR_STEP[int(step[start])]}  -  {FR_PHASE[name]}"
            out.append((start / FPS, k / FPS, text))
            start = k
    return out


def clip(ep: Path, idx: int, dst: Path) -> None:
    meta = json.loads((ep / "meta.json").read_text(encoding="utf-8"))
    d = np.load(ep / "data.npz")
    outcome = ("réussi" if meta["success"] else meta["outcome"]) + (" - départ hors plage" if meta.get("ood") else "")
    draws = [f"drawtext=fontfile='{FONT}':expansion=none:text='{_esc(f'Expert scripté (depuis la TaskSpec) - essai {idx + 1} - {outcome}')}'"
             ":x=12:y=10:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=6"]
    for a, b, text in segments(d["step"], d["phase"]):
        draws.append(f"drawtext=fontfile='{FONT}':expansion=none:text='{_esc(text)}':x=12:y=h-40:fontsize=22:fontcolor=yellow"
                     f":box=1:boxcolor=black@0.55:boxborderw=6:enable='between(t,{a:.3f},{b:.3f})'")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(ep / "front.mp4"), "-i", str(ep / "wrist.mp4"),
                    "-filter_complex", f"[0][1]hstack=inputs=2,scale={W}:{H},{','.join(draws)}",
                    "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-r", str(FPS), "-an", str(dst)], check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", nargs="+", default=["outputs/p3_2/debug_episodes"])
    p.add_argument("--out", default="outputs/p3_2/demo_expert.mp4")
    p.add_argument("--title", default="Projet CTA - P3 : démonstrations générées depuis la vidéo humaine"
                   "|TaskSpec (vidéo FPV + TPV) -> scène SO-101 -> expert scripté"
                   "|Consigne : poser le bout de profilé court à plat, puis le profilé long debout dessus")
    p.add_argument("--end", default=None, help="closing card ('|' = new line)")
    p.add_argument("--card_s", type=float, default=5.0)
    a = p.parse_args()
    eps = []
    for e in map(Path, a.episodes):
        eps += [e] if (e / "meta.json").exists() else sorted(d for d in e.iterdir() if (d / "meta.json").exists())
    if not eps:
        raise SystemExit(f"no episode in {a.episodes}")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        parts = [tmp / "title.mp4"]
        card(a.title.split("|"), a.card_s, parts[0])
        for i, ep in enumerate(eps):
            parts.append(tmp / f"clip_{i}.mp4")
            clip(ep, i, parts[-1])
        if a.end:
            parts.append(tmp / "end.mp4")
            card(a.end.split("|"), a.card_s, parts[-1])
        lst = tmp / "list.txt"
        lst.write_text("".join(f"file '{q.as_posix()}'\n" for q in parts), encoding="utf-8")
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out)],
                       check=True)
    print(f"{len(eps)} episodes -> {out}")


if __name__ == "__main__":
    main()
