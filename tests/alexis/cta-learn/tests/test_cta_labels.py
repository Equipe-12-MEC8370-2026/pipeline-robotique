"""P2.2: the hand-written labels of the CTA demos are well-formed (used later to check M3 segmentation)."""
import re

import yaml

from cta_learn import CONFIGS_DIR


def test_labels_well_formed():
    lab = yaml.safe_load((CONFIGS_DIR / "tasks" / "cta_demos_labels.yaml").read_text(encoding="utf-8"))
    assert lab["status"] in {"draft", "confirmed"}
    assert len(lab["pairs"]) == 6
    assert lab["target_pair"] in lab["pairs"]
    for pid, p in lab["pairs"].items():
        assert re.fullmatch(r"pair_\d{8}_\d{6}", pid)
        assert p["task"] and p["objects"] and p["phases"]
        prev_end = 0.0
        for ph in p["phases"]:
            t0, t1 = ph["t"]
            assert 0 <= t0 < t1 and t0 >= prev_end - 1e-9, (pid, ph)  # ordered, non-overlapping
            prev_end = t1
