"""P2.5: hand openness, signal runs, start/end gestures, working-hand side."""
import numpy as np

from cta_learn.task.segment import both_open, hand_side_fpv, openness, runs, start_end_signals


def _hand(open_: bool, x0=500.0) -> np.ndarray:
    """Synthetic 21-point hand: wrist at (x0, 500), MCPs 60 px up, tips 120 px (open) or 50 px (fist) up."""
    kp = np.zeros((21, 2))
    kp[0] = [x0, 500]
    for k, (tip, mcp) in enumerate(((8, 5), (12, 9), (16, 13), (20, 17))):
        dx = (k - 1.5) * 20
        kp[mcp] = [x0 + dx, 440]
        kp[tip] = [x0 + dx * 1.3, 380 if open_ else 450]
    return kp


def test_openness_open_vs_fist():
    assert openness(_hand(True)) > 1.8 and openness(_hand(False)) < 1.2
    assert np.isnan(openness(np.full((21, 2), np.nan)))


def test_both_open_needs_two_open_hands():
    kp = np.full((3, 2, 21, 2), np.nan)
    kp[0, 0], kp[0, 1] = _hand(True), _hand(True, 900)
    kp[1, 0], kp[1, 1] = _hand(True), _hand(False, 900)
    kp[2, 0] = _hand(True)
    assert both_open(kp).tolist() == [True, False, False]


def test_runs_bridge_gaps_and_drop_short():
    fps = 10
    f = np.zeros(100, bool)
    f[10:30] = True; f[33:40] = True   # 2.0 s + gap 0.3 s + 0.7 s -> one run 1.0-4.0 s
    f[60:65] = True                    # 0.5 s -> dropped
    r = runs(f, fps, smooth_s=0.1, min_s=1.5, max_gap_s=0.5)
    assert r == [(1.0, 4.0)]


def test_start_end_signals():
    fps, T = 10, 200
    kp = np.full((T, 2, 21, 2), np.nan)
    for i in list(range(10, 40)) + list(range(160, 190)):
        kp[i, 0], kp[i, 1] = _hand(True), _hand(True, 900)
    s = start_end_signals(kp, kp, fps, smooth_s=0.1)
    assert s["start"] == (1.0, 4.0) and s["end"] == (16.0, 19.0)
    kp2 = kp.copy(); kp2[100:] = np.nan
    assert start_end_signals(kp2, kp2, fps, smooth_s=0.1)["end"] is None
    # end gesture seen by one view only still counts; a lone gesture late in the clip is an end, not a start
    none = np.full_like(kp, np.nan)
    late = none.copy(); late[160:190] = kp[160:190]
    s2 = start_end_signals(late, none, fps, smooth_s=0.1)
    assert s2["start"] is None and s2["end"] == (16.0, 19.0)


def test_hand_side_fpv():
    T = 50
    t = np.arange(T) / 10
    kp = np.full((T, 2, 21, 2), np.nan)
    kp[:, 0, 0] = np.c_[np.linspace(1300, 1600, T), np.full(T, 600)]  # moving hand on the right
    kp[:, 1, 0] = [400, 600]                                           # still hand on the left
    assert hand_side_fpv(kp, t, 0, 5) == ("right", 1.0)
    assert hand_side_fpv(np.full((T, 2, 21, 2), np.nan), t, 0, 5) == ("unknown", 0.0)
