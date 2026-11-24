"""Interface helpers (app/core.py): progress read from the scripts' logs; generic outcome names."""
import sys
from pathlib import Path

import pytest

pytest.importorskip("psutil")
pytest.importorskip("ruamel.yaml")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import core  # noqa: E402

from cta_learn.eval.stack_metrics import classify_task_outcome  # noqa: E402


def test_parse_generation():
    log = ("[attempt    0] success                                    steps= 416 successes=1/1\n"
           "[attempt    1] expert: step 0: timeout in phase 'descend' steps= 327 successes=1/2\n")
    g = core.parse_generation(log)
    assert g == {"attempts": 2, "successes": 1, "failures": [(1, "expert: step 0: timeout in phase 'descend'")]}


def test_parse_training():
    log = ("INFO ot_train.py:641 step:200 smpl:2K ep:4 epch:0.37 loss:6.726 grdn:9\n"
           "Training:  50%|#####     | 1000/2000 [03:37<03:37,  4.6step/s]"
           "INFO ot_train.py:641 step:1K smpl:8K loss:1.801 grdn:9\nINFO step:1K smpl:8K loss:1.800\n")
    t = core.parse_training(log)
    assert t["step"] == 1000 and t["total"] == 2000 and t["remaining"] == "03:37"
    assert t["losses"] == [(200.0, 6.726), (1000.0, 1.8)]


def test_parse_eval():
    log = ("[trial   0 train_range ] success                       416 steps (18.7 s) successes=1/1 of 3\n"
           "[trial   1 train_range ] angle_block_never_lifted      600 steps (53.1 s) successes=1/2 of 3\n")
    assert core.parse_eval(log) == {"done": 2, "total": 3, "successes": 1, "outcomes": {"success": 1, "angle_block_never_lifted": 1}}


def test_generic_outcomes():
    order = ["a", "b"]
    checks = {"a_posture": True, "a_in_zone": True, "b_posture": False, "b_on_a": True, "released": True}
    assert classify_task_outcome(True, order, {}, {}, checks) == "success"
    assert classify_task_outcome(False, order, {"b": True}, {"a": True, "b": True}, checks) == "b_dropped"
    assert classify_task_outcome(False, order, {}, {"a": True, "b": False}, checks) == "b_never_lifted"
    assert classify_task_outcome(False, order, {}, {"a": True, "b": True}, checks) == "b_posture"
    ok = dict(checks, b_posture=True, released=False)
    assert classify_task_outcome(False, order, {}, {"a": True, "b": True}, ok) == "not_released"
