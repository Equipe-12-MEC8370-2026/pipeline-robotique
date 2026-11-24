"""TaskSpec construction: postures from the fitted box, choice of the step's move, schema 0.1 still read."""
import json

import numpy as np

from cta_learn.core.taskspec import SCHEMA_VERSION, TaskSpec
from cta_learn.task.build import build_taskspec, main_move, posture_name, rest_pose

BLOCK = [0.071, 0.069, 0.062]


def _rest(x, y, ext, ambiguous=False, t=(0.0, 1.0)):
    return {"t": list(t), "posture_ambiguous": ambiguous,
            "pose": {"x": x, "y": y, "yaw_deg": 10.0, "upright": ext[2] == max(BLOCK), "z_base": 0.0,
                     "extents_m": ext, "yaw_period_deg": 180.0}}


def test_posture_names():
    assert posture_name({"extents_m": [0.071, 0.069, 0.062]}, BLOCK) == "lying"
    assert posture_name({"extents_m": [0.071, 0.062, 0.069]}, BLOCK) == "on_edge"
    assert posture_name({"extents_m": [0.069, 0.062, 0.071]}, BLOCK) == "upright"
    assert posture_name({"extents_m": [0.071, 0.069, 0.062], "posture_ambiguous": True}, BLOCK) == "unknown"
    # profile, older objects.json without extents
    assert posture_name({"upright": True}, [0.15, 0.04, 0.04]) == "upright"
    assert posture_name({"upright": False}, [0.15, 0.04, 0.04]) == "lying"
    # profile with extents: lying on any long face
    assert posture_name({"extents_m": [0.15, 0.04, 0.04]}, [0.15, 0.04, 0.04]) == "lying"


def test_main_move_is_the_largest_displacement():
    o = {"rests": [_rest(0.8, 0.1, BLOCK), _rest(0.36, 0.17, BLOCK), _rest(0.363, 0.167, BLOCK)],
         "moves": [{"t_grasp": 8.0, "t_release": 12.8}, {"t_grasp": 18.9, "t_release": 29.0}]}
    assert main_move(o)["t_grasp"] == 8.0  # the longer 2nd move only lifts the part and puts it back


def test_block_spec_unknown_posture_on_table():
    oj = {"frame": "table", "objects": {"angle_block": {
        "rests": [_rest(0.8, 0.1, BLOCK, True), _rest(0.36, 0.17, [0.071, 0.062, 0.069], True)],
        "moves": [{"t_grasp": 8.0, "t_motion": [8.1, 12.7], "t_release": 12.8}]}}}
    kp = np.full((10, 2, 21, 2), np.nan)
    spec = build_taskspec(oj, {"angle_block": BLOCK}, {"angle_block": 0.03}, kp, np.linspace(0, 20, 10),
                          {"start": [6.0, 7.5], "end": None}, "t", {"en": "x"}, {})
    g = spec.steps[0].goal
    assert spec.objects[0].initial.posture == "unknown" and g.pose.posture == "unknown"
    assert g.relation.type == "on_table" and spec.success == ["angle_block on the table, released"]
    assert rest_pose(oj["objects"]["angle_block"]["rests"][0])["posture_ambiguous"]


def test_schema_01_is_read_and_upgraded(tmp_path):
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "configs" / "taskspecs" / "cta_profile_upright_113656.json"
    d = json.loads(src.read_text(encoding="utf-8"))
    d["schema_version"] = "0.1"
    (tmp_path / "old.json").write_text(json.dumps(d), encoding="utf-8")
    spec = TaskSpec.load(tmp_path / "old.json")
    assert spec.schema_version == SCHEMA_VERSION == "0.2"
