"""P3.5 evaluation helpers: trial sampling for the profile-stack task and outcome classification."""
import yaml

from cta_learn import CONFIGS_DIR
from cta_learn.eval.stack_metrics import boxes, classify_stack_outcome, make_stack_trials

LAYOUT = yaml.safe_load((CONFIGS_DIR / "tasks" / "so101_profile_stack.yaml").read_text(encoding="utf-8"))
OK = {"short_lying": True, "short_on_table": True, "short_in_zone": True, "long_upright": True, "long_on_short": True,
      "released": True}


def test_trials_deterministic_and_split():
    a, b = make_stack_trials(LAYOUT, 40, 10, 7), make_stack_trials(LAYOUT, 40, 10, 7)
    assert a == b and len(a) == 50
    assert [t["trial"] for t in a] == list(range(50))
    train, ood = boxes(LAYOUT, "range"), boxes(LAYOUT, "range_ood")
    inside = lambda s, box: all(box[p][k][0] <= s[p][k] <= box[p][k][1] for p in box for k in box[p])  # noqa: E731
    assert all(inside(t["offsets"], train) for t in a[:40] if t["split"] == "train_range")
    assert all(t["split"] == "out_of_range" and inside(t["offsets"], ood) and not inside(t["offsets"], train) for t in a[40:])
    assert make_stack_trials(LAYOUT, 40, 10, 8) != a


def test_outcome_classification_in_task_order():
    no = {"profile_short": False, "profile_long": False}
    yes = {"profile_short": True, "profile_long": True}
    assert classify_stack_outcome(True, no, yes, OK, False) == "success"
    assert classify_stack_outcome(False, {"profile_short": False, "profile_long": True}, yes, OK, False) == "long_dropped"
    assert classify_stack_outcome(False, no, no, OK, False) == "short_never_lifted"
    assert classify_stack_outcome(False, no, yes, {**OK, "short_in_zone": False}, False) == "short_not_placed"
    assert classify_stack_outcome(False, no, {"profile_short": True, "profile_long": False}, OK, False) == "long_never_lifted"
    assert classify_stack_outcome(False, no, yes, {**OK, "long_upright": False}, False) == "long_fell"
    assert classify_stack_outcome(False, no, yes, {**OK, "long_on_short": False}, True) == "held_at_timeout"
    assert classify_stack_outcome(False, no, yes, {**OK, "long_on_short": False}, False) == "long_not_on_short"
    assert classify_stack_outcome(False, no, yes, {**OK, "released": False}, False) == "stack_not_released"
