"""P2.4b: hand slot assignment, detection rate, save/load (pure helpers, no mediapipe needed)."""
import numpy as np

from cta_learn.perception.hands import HandTrack, assign_slots, detection_rate


def _kp(v):
    return np.full((21, 3), v, np.float32)


def test_assign_slots_by_label():
    out = assign_slots([("Right", 0.9, _kp(1)), ("Left", 0.8, _kp(2))])
    assert sorted((s, float(k[0, 0])) for s, _, k in out) == [(0, 2.0), (1, 1.0)]


def test_assign_slots_same_label_best_keeps_it():
    out = assign_slots([("Left", 0.6, _kp(1)), ("Left", 0.95, _kp(2))])
    d = {s: float(k[0, 0]) for s, _, k in out}
    assert d == {0: 2.0, 1: 1.0}


def test_assign_slots_drops_third_hand():
    out = assign_slots([("Left", 0.9, _kp(1)), ("Right", 0.8, _kp(2)), ("Left", 0.3, _kp(3))])
    assert len(out) == 2 and 3.0 not in [float(k[0, 0]) for _, _, k in out]


def test_detection_rate_in_intervals():
    t = np.arange(100) / 10.0  # 0..9.9 s
    n = np.zeros(100, int)
    n[20:40] = 1       # 2.0-4.0 s: one hand
    n[60:65] = 2       # 6.0-6.5 s: two hands
    assert detection_rate(n, t, [(2.0, 4.0)]) == 1.0
    assert detection_rate(n, t, [(2.0, 4.0), (6.0, 7.0)]) == 25 / 30
    assert detection_rate(n, t, [(6.0, 7.0)], min_hands=2) == 0.5
    assert np.isnan(detection_rate(n, t, [(20.0, 30.0)]))


def test_track_save_load_roundtrip(tmp_path):
    T = 5
    kp = np.full((T, 2, 21, 2), np.nan, np.float32)
    kp[1:3, 0] = 10.0
    tr = HandTrack(kp, np.zeros((T, 2, 21), np.float32), np.zeros((T, 2), np.float32), np.arange(T, dtype=np.float32) / 30)
    tr.save(tmp_path / "h.npz")
    back = HandTrack.load(tmp_path / "h.npz")
    assert np.array_equal(back.n_hands(), [0, 1, 1, 0, 0])
    assert np.allclose(back.t_s, tr.t_s)
