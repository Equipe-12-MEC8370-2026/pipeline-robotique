"""P2.1 (M1): pairing by file name, audio offset estimation, common span, ffmpeg round trip."""
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from cta_learn.video.ingest import common_span, load_audio, name_time, pair_by_name, probe
from cta_learn.video.sync import estimate_offset, gcc_phat

SR = 16000


def _scene(seconds: float, seed: int = 0) -> np.ndarray:
    """Workshop-like audio: noise floor + sparse clicks/knocks (what the sync relies on)."""
    rng = np.random.default_rng(seed)
    x = 0.02 * rng.standard_normal(int(seconds * SR))
    for t in rng.uniform(0, seconds, int(seconds * 2)):
        i = int(t * SR)
        n = min(800, len(x) - i)
        x[i:i + n] += rng.uniform(0.2, 1.0) * np.exp(-np.arange(n) / 120) * rng.standard_normal(n)
    return x


def _record(scene: np.ndarray, start_s: float, dur_s: float, gain: float, seed: int, lowpass: int = 1) -> np.ndarray:
    """A 'microphone' recording of the scene from start_s: own gain, own noise, optional crude low-pass."""
    seg = scene[int(start_s * SR):int((start_s + dur_s) * SR)] * gain
    if lowpass > 1:
        seg = np.convolve(seg, np.ones(lowpass) / lowpass, mode="same")
    return (seg + 0.01 * np.random.default_rng(seed).standard_normal(len(seg))).astype(np.float32)


def test_name_time_and_pairing():
    f = [Path("20260831_113807.mp4"), Path("20260831_114518.mp4"), Path("notes.mp4")]
    t = [Path("WIN_20260831_11_38_08_Pro.mp4"), Path("WIN_20260831_11_45_16_Pro.mp4"), Path("WIN_20260831_11_21_53_Pro.mp4")]
    assert name_time(f[0]) == datetime(2026, 8, 31, 11, 38, 7)
    assert name_time(t[0]) == datetime(2026, 8, 31, 11, 38, 8)
    assert name_time(f[2]) is None
    pairs, unmatched = pair_by_name(f, t)
    assert [(p.fpv.name, p.tpv.name) for p in pairs] == [(f[0].name, t[0].name), (f[1].name, t[1].name)]
    assert pairs[0].name_offset_s == -1.0 and pairs[1].name_offset_s == 2.0  # start(fpv) - start(tpv)
    assert pairs[0].pair_id == "pair_20260831_113807"
    assert [u.name for u in unmatched] == [t[2].name]


def test_gcc_phat_sign_and_subsample():
    x = _scene(6, seed=1)
    d = 0.25 + 0.4 / SR  # 0.25 s + a fraction of a sample
    y = np.interp(np.arange(len(x)) - d * SR, np.arange(len(x)), x)  # y(t) = x(t - d): delayed copy
    lag, ratio = gcc_phat(x, y, SR)
    assert abs(lag - d) < 0.5 / SR and ratio > 3  # half a sample = 0.03 ms, far below one frame (33 ms)


@pytest.mark.parametrize("true_offset", [-2.3456, 0.0, 1.0789, 4.5])
def test_estimate_offset_recovers_known_offset(true_offset):
    # FPV started at scene time 5 s, TPV at 5 - true_offset s => event at FPV t is at TPV t + true_offset
    scene = _scene(60, seed=2)
    fpv = _record(scene, 5.0, 40.0, gain=1.0, seed=3)
    tpv = _record(scene, 5.0 - true_offset, 45.0, gain=0.3, seed=4, lowpass=4)
    r = estimate_offset(fpv, tpv, SR, prior_s=round(true_offset))  # file-name prior is whole seconds
    assert abs(r.offset_s - true_offset) < 1e-3
    assert r.peak_ratio > 3 and len(r.window_offsets_s) >= 5
    assert r.spread_ms < 2.0 and abs(r.drift_ms_per_min) < 1.0


def test_audio_start_shift_is_applied():
    scene = _scene(40, seed=5)
    fpv = _record(scene, 5.0, 30.0, 1.0, 6)
    tpv = _record(scene, 5.0 + 0.095, 30.0, 0.5, 7)  # same start as FPV, but audio begins 95 ms into the file
    r = estimate_offset(fpv, tpv, SR, tpv_audio_start_s=0.095, prior_s=0.0)
    assert abs(r.offset_s) < 1e-3


def test_clock_drift_is_reported():
    scene = _scene(60, seed=8)
    fpv = _record(scene, 5.0, 40.0, 1.0, 9)
    t = np.arange(int(42 * SR)) / SR
    src = 5.0 - 1.0 + t * (1 + 2e-4)  # TPV clock 200 ppm fast -> 12 ms/min drift
    tpv = np.interp(src * SR, np.arange(len(scene)), scene).astype(np.float32)
    r = estimate_offset(fpv, tpv, SR, prior_s=1.0)
    assert -14 < r.drift_ms_per_min < -10  # TPV time for an FPV event shrinks as the TPV clock runs fast
    assert r.spread_ms > 3


def test_common_span():
    assert common_span(2.0, 40.0, 45.0) == (0.0, 2.0, 40.0)
    fs, ts, d = common_span(-1.5, 40.0, 45.0)
    assert (fs, ts) == (1.5, 0.0) and d == pytest.approx(38.5)
    with pytest.raises(ValueError):
        common_span(50.0, 40.0, 45.0)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_ffmpeg_round_trip(tmp_path):
    """Two real files with a known offset and a delayed audio stream: probe + load_audio + estimate."""
    scene = _scene(30, seed=10)
    wav = tmp_path / "scene.f32"
    scene.astype(np.float32).tofile(wav)

    def make(name, start, dur, adelay_ms):
        out = tmp_path / name
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size=160x120:rate=30:duration={dur}",
                        "-f", "f32le", "-ar", str(SR), "-ac", "1", "-ss", str(start), "-t", str(dur), "-i", str(wav),
                        "-af", f"adelay={adelay_ms}:all=1" if adelay_ms else "anull",
                        "-c:v", "libx264", "-c:a", "aac", "-shortest", str(out)], check=True)
        return out

    f = make("fpv.mp4", 3.0, 20, 0)
    t = make("tpv.mp4", 1.75, 20, 0)  # TPV started 1.25 s earlier -> offset +1.25 s
    pf, pt = probe(f), probe(t)
    af, at = load_audio(f, SR), load_audio(t, SR)
    assert abs(len(af) / SR - 20) < 0.1
    r = estimate_offset(af, at, SR, fpv_audio_start_s=pf["audio"]["start_s"], tpv_audio_start_s=pt["audio"]["start_s"],
                        prior_s=1.0)
    assert abs(r.offset_s - 1.25) < 0.005  # AAC priming/encoder delay stays well under one frame
