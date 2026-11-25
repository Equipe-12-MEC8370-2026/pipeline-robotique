r"""Tests sans matériel de mesure.py (lancer depuis tests/tristan/passerelle) :
    ..\..\alexis\.venv-learn\Scripts\python.exe -m pytest -q
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "alexis" / "src"))
import mesure  # noqa: E402

# copie de Greg.json (2026-11-17) : le test ne dépend pas du fichier sur le disque
GREG = {
    "shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": 1055, "range_min": 745, "range_max": 3461},
    "shoulder_lift": {"id": 2, "drive_mode": 0, "homing_offset": 1053, "range_min": 820, "range_max": 3112},
    "elbow_flex": {"id": 3, "drive_mode": 0, "homing_offset": 43, "range_min": 940, "range_max": 3145},
    "wrist_flex": {"id": 4, "drive_mode": 0, "homing_offset": -4, "range_min": 892, "range_max": 3201},
    "wrist_roll": {"id": 5, "drive_mode": 0, "homing_offset": -149, "range_min": 0, "range_max": 4095},
    "gripper": {"id": 6, "drive_mode": 0, "homing_offset": -854, "range_min": 2046, "range_max": 3462},
}


@pytest.fixture
def calib(tmp_path):
    f = tmp_path / "greg.json"
    f.write_text(json.dumps(GREG))
    return mesure.load_calibration(f)


def lerobot_bus(calib, degrees: bool):
    """Bus LeRobot NON connecté : on n'utilise que sa normalisation (aucun port ouvert)."""
    from lerobot.motors import Motor, MotorCalibration, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    body = MotorNormMode.DEGREES if degrees else MotorNormMode.RANGE_M100_100
    motors = {j: Motor(calib[j].id, "sts3215", body) for j in mesure.BODY}
    motors["gripper"] = Motor(calib["gripper"].id, "sts3215", MotorNormMode.RANGE_0_100)
    cal = {j: MotorCalibration(c.id, c.drive_mode, c.homing_offset, c.range_min, c.range_max) for j, c in calib.items()}
    return FeetechMotorsBus(port="COM_INEXISTANT", motors=motors, calibration=cal)


RAWS = [
    {j: int((GREG[j]["range_min"] + GREG[j]["range_max"]) / 2) for j in mesure.JOINTS},  # milieux
    {j: GREG[j]["range_min"] for j in mesure.JOINTS},
    {j: GREG[j]["range_max"] for j in mesure.JOINTS},
    {"shoulder_pan": 1000, "shoulder_lift": 3000, "elbow_flex": 1200, "wrist_flex": 2900, "wrist_roll": 4000,
     "gripper": 2500},
]


@pytest.mark.parametrize("raw", RAWS)
def test_lerobot_degrees_match_lerobot(calib, raw):
    bus = lerobot_bus(calib, degrees=True)
    ref = bus._normalize({calib[j].id: v for j, v in raw.items()})
    ours = mesure.raw_to_lerobot(raw, calib)
    for j in mesure.JOINTS:
        assert ours[j] == pytest.approx(ref[calib[j].id], abs=1e-9), j


@pytest.mark.parametrize("raw", RAWS)
def test_m100_matches_lerobot(calib, raw):
    bus = lerobot_bus(calib, degrees=False)
    ref = bus._normalize({calib[j].id: v for j, v in raw.items()})
    ours = mesure.raw_to_m100(raw, calib)
    for j in mesure.JOINTS:
        assert ours[j] == pytest.approx(ref[calib[j].id], abs=1e-9), j


def test_leisaac_hypothesis_matches_cta_learn_units(calib):
    """L'hypothèse « leisaac » doit être exactement ce que fait cta_learn.core.units (format « lerobot »)."""
    from cta_learn.core import load_embodiment
    from cta_learn.core.units import to_rad

    emb = load_embodiment("so101")
    for raw in RAWS:
        m = mesure.raw_to_m100(raw, calib)
        ref = np.degrees(to_rad([m[j] for j in mesure.JOINTS], "lerobot", emb))
        ours = mesure.sim_deg_leisaac(raw, calib)
        assert [ours[j] for j in mesure.JOINTS] == pytest.approx(list(ref), abs=1e-9)


def test_sim_limits_match_embodiment():
    from cta_learn.core import load_embodiment

    emb = load_embodiment("so101")
    for j, lo, hi in zip(emb.joint_names, emb.lower, emb.upper):
        assert mesure.SIM_LIMITS_DEG[j] == pytest.approx((math.degrees(lo), math.degrees(hi)))


def test_middle_of_range_is_zero_degree(calib):
    mid = {j: (GREG[j]["range_min"] + GREG[j]["range_max"]) / 2 for j in mesure.JOINTS}
    d = mesure.raw_to_lerobot(mid, calib)
    assert all(abs(d[j]) < 1e-9 for j in mesure.BODY)
    assert d["gripper"] == pytest.approx(50.0)


def test_gripper_is_clamped(calib):
    raw = dict(RAWS[0], gripper=0)
    assert mesure.raw_to_lerobot(raw, calib)["gripper"] == 0.0
    raw = dict(RAWS[0], gripper=4095)
    assert mesure.raw_to_lerobot(raw, calib)["gripper"] == 100.0


def test_calibration_mismatches(calib):
    motor = {j: {k: GREG[j][k] for k in ("homing_offset", "range_min", "range_max")} for j in mesure.JOINTS}
    assert mesure.calibration_mismatches(calib, motor) == []
    motor["elbow_flex"]["range_max"] += 1
    assert mesure.calibration_mismatches(calib, motor) == ["elbow_flex.range_max: fichier 3145 != moteur 3146"]


def test_load_calibration_rejects_bad_files(tmp_path):
    f = tmp_path / "bad.json"
    f.write_text(json.dumps({k: v for k, v in GREG.items() if k != "gripper"}))
    with pytest.raises(ValueError, match="manquantes"):
        mesure.load_calibration(f)
    bad = json.loads(json.dumps(GREG))
    bad["wrist_flex"]["range_min"] = bad["wrist_flex"]["range_max"]
    f.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="range_min"):
        mesure.load_calibration(f)


def test_fit_sign_offset_recovers_sign_and_offset():
    real = np.array([-80.0, 10.0, 45.0])
    a, b, r = mesure.fit_sign_offset(real, -real + 12.0)
    assert (a, b) == pytest.approx((-1.0, 12.0)) and r < 1e-9
    with pytest.raises(ValueError):
        mesure.fit_sign_offset(np.array([5.0, 5.0]), np.array([1.0, 2.0]))
