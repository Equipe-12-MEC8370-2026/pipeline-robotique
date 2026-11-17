"""TaskSpec: validation rules and JSON round trip (stdlib only, must import in every environment)."""
import copy

import pytest

from cta_learn.core.taskspec import (Goal, ObjectSpec, Phase, Pose, Relation, Step, TaskSpec, TaskSpecError)


def _spec() -> TaskSpec:
    short = ObjectSpec("profile_short", "box", [0.08, 0.04, 0.04], 0.05, Pose(0.82, 0.16, 0.0, "upright"))
    long_ = ObjectSpec("profile_long", "box", [0.15, 0.04, 0.04], 0.08, Pose(0.88, 0.37, 15.0, "upright"))
    s0 = Step(0, "profile_short", "right", 0.9, 12.6, 19.6,
              [Phase("reach", [11.0, 12.6]), Phase("transport", [12.6, 19.6]), Phase("retreat", [19.6, 20.5])],
              Goal(Pose(0.156, 0.157, 89.0, "lying"), Relation("on_table")))
    s1 = Step(1, "profile_long", "right", 0.8, 21.1, 26.9,
              [Phase("reach", [20.5, 21.1]), Phase("transport", [21.1, 26.9])],
              Goal(Pose(0.16, 0.155, 13.0, "upright", 0.04), Relation("on_top_of", "profile_short", [0.002, -0.004], 14.0)))
    return TaskSpec("cta_profile_upright", {"fr": "Pose...", "en": "Place..."}, "table frame", [short, long_], [s0, s1],
                    ["profile_short lying", "profile_long upright on profile_short"], {"start": [2.0, 11.0], "end": None})


def test_valid_spec_roundtrips(tmp_path):
    s = _spec().validate()
    s.save(tmp_path / "t.json")
    back = TaskSpec.load(tmp_path / "t.json")
    assert back == s
    assert back.steps[1].goal.relation.ref == "profile_short"


@pytest.mark.parametrize("mutate,msg", [
    (lambda s: setattr(s.objects[1], "id", "profile_short"), "unique"),
    (lambda s: setattr(s.steps[0], "object", "bolt"), "unknown object"),
    (lambda s: setattr(s.steps[0].goal.pose, "posture", "sideways"), "posture"),
    (lambda s: setattr(s.steps[1].goal.relation, "ref", "profile_long"), "earlier step"),
    (lambda s: setattr(s.steps[1], "t_grasp", 15.0), "previous step"),
    (lambda s: setattr(s.steps[0], "t_release", 10.0), "t_grasp must be < t_release"),
    (lambda s: s.steps[0].phases.reverse(), "ordered"),
    (lambda s: setattr(s.steps[0], "hand", "both"), "hand"),
    (lambda s: s.instruction.pop("en"), "instruction"),
    (lambda s: setattr(s.objects[0], "size_m", [0.08, 0.0, 0.04]), "size_m"),
    (lambda s: setattr(s, "schema_version", "9.9"), "schema_version"),
])
def test_invalid_specs_are_rejected(mutate, msg):
    s = copy.deepcopy(_spec())
    mutate(s)
    with pytest.raises(TaskSpecError, match=msg):
        s.validate()
