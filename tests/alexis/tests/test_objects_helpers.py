"""P2.4c: chunk planning, boxes, jump detection (pure helpers, no SAM 2 needed)."""
import numpy as np
import pytest

from cta_learn.perception.objects import box_iou, jump_frames, mask_bbox, plan_chunks


def _covered(chunks):
    s = set()
    for a, b in chunks:
        s |= set(range(a, b + 1))
    return s


@pytest.mark.parametrize("n,start,chunk", [(540, 315, 150), (540, 0, 150), (540, 539, 150), (10, 4, 3), (2, 1, 2)])
def test_plan_chunks_cover_everything_with_seed_overlap(n, start, chunk):
    fwd, bwd = plan_chunks(n, start, chunk)
    assert _covered(fwd) | _covered(bwd) | {start} == set(range(n))
    assert all(b - a + 1 <= chunk for a, b in fwd + bwd)
    if fwd:
        assert fwd[0][0] == start and all(fwd[i + 1][0] == fwd[i][1] for i in range(len(fwd) - 1))
    if bwd:
        assert bwd[0][1] == start and all(bwd[i + 1][1] == bwd[i][0] for i in range(len(bwd) - 1))


def test_plan_chunks_rejects_bad_input():
    with pytest.raises(ValueError):
        plan_chunks(10, 10, 5)
    with pytest.raises(ValueError):
        plan_chunks(10, 3, 1)


def test_mask_bbox_and_iou():
    m = np.zeros((50, 60), bool)
    m[10:20, 5:25] = True
    assert np.array_equal(mask_bbox(m), [5, 10, 25, 20])
    assert mask_bbox(np.zeros((5, 5), bool)) is None
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert box_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(50 / 150)
    assert box_iou([0, 0, 1, 1], [2, 2, 3, 3]) == 0.0


def test_jump_frames_flags_teleports_and_dropouts():
    T = 8
    masks = np.zeros((T, 100, 100), bool)
    for i in range(T):
        masks[i, 40:50, 10 + 2 * i:20 + 2 * i] = True  # smooth motion, 2 px per frame
    masks[4] = False                                   # dropout at 4, back at 5
    masks[6] = False
    masks[6, 80:90, 80:90] = True                      # teleport at 6, back at 7
    assert jump_frames(masks, max_jump_px=15) == [4, 5, 6, 7]


def test_merge_tracks_prefers_nonempty_then_nearest_prompt():
    from cta_learn.perception.objects import merge_tracks
    T = 6
    a = np.zeros((T, 4, 4), bool); b = np.zeros((T, 4, 4), bool)
    a[0:3, 0, 0] = True          # track A (prompt at 0) sees frames 0-2
    b[2:6, 3, 3] = True          # track B (prompt at 5) sees frames 2-5
    m, src = merge_tracks([a, b], [0, 5])
    assert src.tolist() == [0, 0, 0, 1, 1, 1]    # frame 2: both non-empty, A's prompt is nearer
    a2 = a.copy(); a2[4] = True; b2 = b.copy(); b2[4] = False
    _, src2 = merge_tracks([a2, b2], [0, 5])
    assert src2[4] == 0                          # only A non-empty at 4
    _, src3 = merge_tracks([np.zeros_like(a), np.zeros_like(b)], [0, 5])
    assert (src3 == -1).all()


def test_plan_chunks_subrange():
    fwd, bwd = plan_chunks(100, 50, 20, lo=30, hi=70)
    assert fwd[0][0] == 50 and fwd[-1][1] == 70 and bwd[-1][0] == 30


def test_save_load_masks_roundtrip(tmp_path):
    from cta_learn.perception.objects import load_masks, save_masks
    m = np.random.default_rng(0).random((4, 7, 13)) > 0.5
    save_masks(tmp_path / "m.npz", m, 15)
    back, fps = load_masks(tmp_path / "m.npz")
    assert fps == 15 and np.array_equal(back, m)
