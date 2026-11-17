"""M1 — audio-based synchronisation of a FPV/TPV video pair (pure numpy, no I/O).

Convention used everywhere: ``offset_s`` = t_tpv - t_fpv for the same physical event,
both times measured on each file's own container timeline (seconds from the file start).
An event at FPV time t therefore appears at TPV time t + offset_s.

Method: GCC-PHAT cross-correlation of the two mono audio tracks (robust to the very
different microphones of a phone and a webcam), parabolic sub-sample refinement, then a
consistency check on short windows: the spread of the per-window offsets is the
measured sync error (the P2 exit test asks for <= 33 ms, one frame at 30 fps).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np


def gcc_phat(ref: np.ndarray, sig: np.ndarray, sr: int, max_lag_s: float | None = None,
             center_s: float = 0.0) -> tuple[float, float]:
    """Delay of ``sig`` relative to ``ref`` in seconds (positive: the event comes later in ``sig``).

    Searches lags in ``center_s ± max_lag_s`` (whole range if None). Returns (lag_s, peak_ratio),
    peak_ratio = highest peak / second highest peak outside ±10 ms (>~3 means a clear match).
    """
    n = len(ref) + len(sig)
    nfft = 1 << (n - 1).bit_length()
    R = np.fft.rfft(sig, nfft) * np.conj(np.fft.rfft(ref, nfft))
    R /= np.abs(R) + 1e-12
    cc = np.fft.irfft(R, nfft)
    lags = np.arange(nfft)
    lags = np.where(lags < nfft // 2, lags, lags - nfft)  # circular index -> signed lag (samples)
    ok = np.ones(nfft, bool)
    if max_lag_s is not None:
        ok &= np.abs(lags - center_s * sr) <= max_lag_s * sr
    ok &= (lags > -len(ref)) & (lags < len(sig))  # lags with some real overlap
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        raise ValueError("empty lag search range")
    vals = cc[idx]
    best = idx[np.argmax(vals)]
    peak = cc[best]
    # parabolic sub-sample refinement
    y0, y1, y2 = cc[(best - 1) % nfft], peak, cc[(best + 1) % nfft]
    den = y0 - 2 * y1 + y2
    frac = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    lag = (lags[best] + float(np.clip(frac, -0.5, 0.5))) / sr
    far = np.abs(lags[idx] - lags[best]) > 0.010 * sr
    second = vals[far].max() if far.any() else 0.0
    ratio = float(peak / second) if second > 0 else float("inf")
    return float(lag), ratio


@dataclass
class SyncResult:
    offset_s: float                 # t_tpv - t_fpv (container timelines)
    peak_ratio: float               # confidence of the global match
    window_offsets_s: list[float] = field(default_factory=list)  # per-window estimates (confident ones)
    window_times_s: list[float] = field(default_factory=list)    # FPV centre time of each window
    n_windows_total: int = 0
    spread_ms: float = float("nan")  # max |window - global| over confident windows
    drift_ms_per_min: float = float("nan")  # slope of the window offsets (clock drift)

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_offset(fpv: np.ndarray, tpv: np.ndarray, sr: int, *, fpv_audio_start_s: float = 0.0,
                    tpv_audio_start_s: float = 0.0, prior_s: float | None = None, search_s: float = 10.0,
                    win_s: float = 8.0, hop_s: float = 4.0, min_ratio: float = 3.0,
                    local_search_s: float = 0.25) -> SyncResult:
    """Offset between two audio tracks + per-window consistency check.

    ``*_audio_start_s`` is where each audio stream starts on its container timeline (ffprobe
    stream start_time minus format start_time): audio sample k is at start + k/sr.
    ``prior_s`` (e.g. from file-name timestamps) narrows the global search to prior ± search_s.
    """
    shift = tpv_audio_start_s - fpv_audio_start_s  # container offset = audio-sample lag + shift
    center = 0.0 if prior_s is None else prior_s - shift
    lag, ratio = gcc_phat(fpv, tpv, sr, max_lag_s=None if prior_s is None else search_s, center_s=center)
    offset = lag + shift

    win, hop = int(win_s * sr), int(hop_s * sr)
    offs, times, total = [], [], 0
    for s in range(0, max(len(fpv) - win, 0) + 1, hop):
        a = fpv[s:s + win]
        t0 = s + int(round(lag * sr)) - int(local_search_s * sr)  # matching TPV segment, with margin
        lo, hi = max(t0, 0), min(t0 + win + 2 * int(local_search_s * sr), len(tpv))
        if hi - lo < win // 2:
            continue  # window falls outside the TPV recording
        total += 1
        b = tpv[lo:hi]
        if np.std(a) < 1e-6 or np.std(b) < 1e-6:
            continue
        expected_local = lag - (lo - s) / sr
        l, r = gcc_phat(a, b, sr, max_lag_s=local_search_s, center_s=expected_local)
        if r >= min_ratio:
            offs.append(l + (lo - s) / sr + shift)
            times.append(fpv_audio_start_s + (s + win / 2) / sr)
    res = SyncResult(offset_s=offset, peak_ratio=ratio, window_offsets_s=offs, window_times_s=times,
                     n_windows_total=total)
    if offs:
        res.spread_ms = float(np.max(np.abs(np.asarray(offs) - offset)) * 1000)
    if len(offs) >= 3:
        res.drift_ms_per_min = float(np.polyfit(np.asarray(times) / 60.0, np.asarray(offs) * 1000, 1)[0])
    return res
