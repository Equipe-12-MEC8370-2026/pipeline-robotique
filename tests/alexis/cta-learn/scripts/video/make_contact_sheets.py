"""P2.2 — one contact sheet per synced pair, to read and label the demonstrated task.

Each sheet: N evenly spaced instants; for each block of 6 instants, a TPV row then the FPV row at the
SAME instants (clips are synced), with the clip time burnt in. Reads data/clips/<pair>/{fpv,tpv}.mp4.
    .venv-learn\\Scripts\\python.exe scripts\\video\\make_contact_sheets.py [--n 18]
-> outputs/p2_2/sheet_<pair>.jpg
"""
from __future__ import annotations

import argparse
from pathlib import Path

import av
import cv2
import numpy as np

from cta_learn import REPO_ROOT


def frames_at(path: Path, times: list[float], width: int) -> list[np.ndarray]:
    """RGB frames (resized to ``width``) closest to the requested times, single decoding pass."""
    out, want = [], list(times)
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        for fr in c.decode(s):
            t = float(fr.pts * s.time_base)
            while want and t >= want[0] - 1e-3:
                img = fr.to_ndarray(format="rgb24")
                h = int(img.shape[0] * width / img.shape[1])
                out.append(cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA))
                want.pop(0)
            if not want:
                break
    while len(out) < len(times):  # past the end: repeat last frame
        out.append(out[-1])
    return out


def label(img: np.ndarray, text: str) -> np.ndarray:
    img = img.copy()
    cv2.rectangle(img, (0, 0), (8 + 11 * len(text), 24), (0, 0, 0), -1)
    cv2.putText(img, text, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=Path, default=REPO_ROOT / "data" / "clips")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "p2_2")
    ap.add_argument("--n", type=int, default=18, help="instants per pair (multiple of --cols)")
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--width", type=int, default=320)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for d in sorted(a.clips.glob("pair_*")):
        with av.open(str(d / "tpv.mp4")) as c:
            dur = float(c.duration) / av.time_base
        times = list((np.arange(a.n) + 0.5) * dur / a.n)
        tpv, fpv = frames_at(d / "tpv.mp4", times, a.width), frames_at(d / "fpv.mp4", times, a.width)
        rows = []
        for b in range(0, a.n, a.cols):
            sl = slice(b, b + a.cols)
            rows.append(np.hstack([label(f, f"TPV {t:5.1f}s") for f, t in zip(tpv[sl], times[sl])]))
            rows.append(np.hstack([label(f, f"FPV {t:5.1f}s") for f, t in zip(fpv[sl], times[sl])]))
            rows.append(np.zeros((6, rows[-1].shape[1], 3), np.uint8))  # separator between blocks
        sheet = np.vstack(rows[:-1])
        dst = a.out / f"sheet_{d.name}.jpg"
        cv2.imwrite(str(dst), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"[sheets] {d.name}: {dur:.1f} s -> {dst}")


if __name__ == "__main__":
    main()
