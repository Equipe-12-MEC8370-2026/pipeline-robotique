"""M1 — ingestion of the CTA human demonstrations: pairing, probing, audio, synced CFR clips.

Original videos are only read (ffprobe/ffmpeg), never modified. Outputs go to data/clips/<pair_id>/:
fpv.mp4 and tpv.mp4 cut to their common time span, constant 30 fps (the sources are variable
frame rate), plus sync.json describing exactly how they were made.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

FPV_RE = re.compile(r"^(\d{8})_(\d{6})\.mp4$", re.I)                        # phone: 20260831_113807.mp4
TPV_RE = re.compile(r"^WIN_(\d{8})_(\d{2})_(\d{2})_(\d{2})_Pro\.mp4$", re.I)  # Windows Camera app


def name_time(path: Path) -> datetime | None:
    """Recording start time encoded in the file name (local time), or None if not a known pattern."""
    if m := FPV_RE.match(path.name):
        return datetime.strptime(m[1] + m[2], "%Y%m%d%H%M%S")
    if m := TPV_RE.match(path.name):
        return datetime.strptime(m[1] + m[2] + m[3] + m[4], "%Y%m%d%H%M%S")
    return None


@dataclass
class Pair:
    pair_id: str
    fpv: Path
    tpv: Path
    name_offset_s: float  # start(fpv) - start(tpv) from file names = rough prior for offset_s


def pair_by_name(fpv_files: list[Path], tpv_files: list[Path], tol_s: float = 10.0) -> tuple[list[Pair], list[Path]]:
    """Match each FPV file to the TPV file whose name time is closest (within tol_s), one-to-one.

    Returns (pairs sorted by time, unmatched files). With offset_s = t_tpv - t_fpv, an event at
    FPV time 0 happened at wall time start(fpv), i.e. TPV time start(fpv) - start(tpv): that is the prior.
    """
    fp = sorted((name_time(f), f) for f in fpv_files if name_time(f))
    tp = sorted((name_time(f), f) for f in tpv_files if name_time(f))
    cands = sorted((abs((tf - tt).total_seconds()), i, j) for i, (tf, _) in enumerate(fp) for j, (tt, _) in enumerate(tp))
    used_f, used_t, pairs = set(), set(), []
    for d, i, j in cands:
        if d > tol_s or i in used_f or j in used_t:
            continue
        used_f.add(i), used_t.add(j)
        tf, f = fp[i]
        tt, t = tp[j]
        pairs.append(Pair(f"pair_{tf:%Y%m%d_%H%M%S}", f, t, (tf - tt).total_seconds()))
    unmatched = [f for k, (_, f) in enumerate(fp) if k not in used_f] + [t for k, (_, t) in enumerate(tp) if k not in used_t]
    return sorted(pairs, key=lambda p: p.pair_id), unmatched


def ffbin(name: str) -> str:
    exe = shutil.which(name)
    if not exe:
        raise FileNotFoundError(f"{name} not found on PATH (install ffmpeg)")
    return exe


def probe(path: Path) -> dict:
    """Container/stream facts needed downstream (times in seconds, relative to the container start)."""
    out = subprocess.run([ffbin("ffprobe"), "-v", "error", "-show_entries",
                          "format=start_time,duration:stream=index,codec_type,codec_name,width,height,"
                          "avg_frame_rate,nb_frames,start_time,sample_rate",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    f0 = float(j["format"].get("start_time", 0) or 0)
    v = next(s for s in j["streams"] if s["codec_type"] == "video")
    a = next((s for s in j["streams"] if s["codec_type"] == "audio"), None)
    num, den = (int(x) for x in v["avg_frame_rate"].split("/"))
    return {
        "duration_s": float(j["format"]["duration"]),
        "video": {"codec": v["codec_name"], "width": v["width"], "height": v["height"],
                  "avg_fps": num / den if den else None, "nb_frames": int(v.get("nb_frames", 0) or 0),
                  "start_s": float(v.get("start_time", f0)) - f0},
        "audio": None if a is None else {"codec": a["codec_name"], "sample_rate": int(a["sample_rate"]),
                                         "start_s": float(a.get("start_time", f0)) - f0},
    }


def load_audio(path: Path, sr: int = 16000) -> np.ndarray:
    """Mono float32 audio at ``sr``; sample 0 = the audio stream's own first sample (see probe audio.start_s)."""
    raw = subprocess.run([ffbin("ffmpeg"), "-v", "error", "-i", str(path), "-map", "0:a:0", "-ac", "1",
                          "-ar", str(sr), "-af", "aresample=async=0", "-copyts", "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def common_span(offset_s: float, fpv_dur: float, tpv_dur: float) -> tuple[float, float, float]:
    """Common time span as (fpv_start, tpv_start, duration), with t_tpv = t_fpv + offset_s."""
    start_f = max(0.0, -offset_s)
    end_f = min(fpv_dur, tpv_dur - offset_s)
    if end_f <= start_f:
        raise ValueError("the two recordings do not overlap")
    return start_f, start_f + offset_s, end_f - start_f


def export_clip(src: Path, dst: Path, start_s: float, dur_s: float, fps: int = 30, crf: int = 18,
                height: int | None = None) -> None:
    """Accurate cut (-ss after decoding start) re-encoded to constant ``fps`` H.264, audio kept (AAC)."""
    vf = f"fps={fps}" + (f",scale=-2:{height}" if height else "")
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([ffbin("ffmpeg"), "-v", "error", "-y", "-i", str(src), "-ss", f"{start_s:.6f}", "-t", f"{dur_s:.6f}",
                    "-vf", vf, "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(dst)], check=True)
