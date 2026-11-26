r"""Tests sans matériel de conversion.py et generer_correspondance.py (depuis tests/tristan/passerelle) :
    ..\..\alexis\.venv-learn\Scripts\python.exe -m pytest -q
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

import chemins  # noqa: E402
import generer_correspondance  # noqa: E402
import mesure  # noqa: E402
from conversion import Correspondance, HorsLimites  # noqa: E402
from test_mesure import GREG  # noqa: E402

DECAL = {"shoulder_pan": 1.89, "shoulder_lift": -6.35, "elbow_flex": -4.84, "wrist_flex": -6.39, "wrist_roll": 7.86}
CORR = {
    "decalage_deg": DECAL,
    "pince_sim_deg": [-10.0, 100.0],
    "calibration": {j: {k: GREG[j][k] for k in ("homing_offset", "range_min", "range_max")} for j in mesure.JOINTS},
}
# poses D et E mesurées le 2026-11-25 (brut moteur) et leurs angles sim visés
POSE_D = ({"shoulder_pan": 2079, "shoulder_lift": 1892, "elbow_flex": 2268, "wrist_flex": 3130, "wrist_roll": 1952,
           "gripper": 2172}, [0.0, -14.39, 15.92, 85.02, 0.0, 0.0])
POSE_E = ({"shoulder_pan": 2082, "shoulder_lift": 1862, "elbow_flex": 1269, "wrist_flex": 2019, "wrist_roll": 1952,
           "gripper": 2171}, [0.0, -13.95, -73.78, -4.97, 0.0, 0.0])


@pytest.fixture
def calib(tmp_path):
    f = tmp_path / "greg.json"
    f.write_text(json.dumps(GREG))
    return mesure.load_calibration(f)


@pytest.fixture
def corr(calib):
    return Correspondance.depuis_dict(CORR, calib)


def obs_from_raw(raw, calib):
    return {f"{j}.pos": v for j, v in mesure.raw_to_lerobot(raw, calib).items()}


def test_known_values(corr):
    obs = {"shoulder_pan.pos": 0.0, "shoulder_lift.pos": 0.0, "elbow_flex.pos": 0.0, "wrist_flex.pos": 0.0,
           "wrist_roll.pos": 0.0, "gripper.pos": 0.0}
    q = np.degrees(corr.reel_vers_sim(obs))
    assert q[:5] == pytest.approx([DECAL[j] for j in mesure.BODY])
    assert q[5] == pytest.approx(-10.0)
    obs["gripper.pos"] = 100.0
    assert np.degrees(corr.reel_vers_sim(obs))[5] == pytest.approx(100.0)


def _valid_real_range(corr, j):
    """Zone valide des deux côtés : plage calibrée réelle ∩ limites sim ramenées en valeurs réelles."""
    lo_s, hi_s = mesure.SIM_LIMITS_DEG[j]
    if j == "gripper":
        return 0.0, 100.0
    return max(corr.reel_min_deg[j], lo_s - DECAL[j]), min(corr.reel_max_deg[j], hi_s - DECAL[j])


def test_round_trip(corr):
    rng = np.random.default_rng(0)
    for _ in range(500):
        obs = {f"{j}.pos": rng.uniform(*_valid_real_range(corr, j)) for j in mesure.JOINTS}
        back = corr.sim_vers_reel(corr.reel_vers_sim(obs))
        assert list(back) == [f"{j}.pos" for j in mesure.JOINTS]
        assert [back[k] for k in obs] == pytest.approx(list(obs.values()), abs=1e-9)


@pytest.mark.parametrize("pose", [POSE_D, POSE_E], ids=["D", "E"])
def test_measured_poses_land_on_sim_targets(corr, calib, pose):
    """Les poses précises D et E, mesurées sur le vrai bras, doivent retomber sur leurs angles sim visés
    à l'incertitude de mesure près (poignet ± 3,9°, le reste < 2°)."""
    raw, target = pose
    q = np.degrees(corr.reel_vers_sim(obs_from_raw(raw, calib)))
    tol = [1.0, 2.0, 2.0, 4.0, 1.0, 2.0]
    assert np.all(np.abs(q - np.array(target)) <= tol), (q - np.array(target)).round(2)


def test_real_out_of_range_is_refused(corr):
    obs = {f"{j}.pos": 0.0 for j in mesure.JOINTS}
    obs["elbow_flex.pos"] = corr.reel_max_deg["elbow_flex"] + 10
    with pytest.raises(HorsLimites, match="elbow_flex"):
        corr.reel_vers_sim(obs)
    obs["elbow_flex.pos"] = float("nan")
    with pytest.raises(HorsLimites):
        corr.reel_vers_sim(obs)


def test_real_beyond_sim_limits_is_refused(corr):
    """Le vrai bras va plus loin que la sim (wrist_roll ±180° réel, ±160° sim) : l'observation est refusée,
    la politique ne recevra jamais un état qu'elle n'a pas pu voir en sim."""
    obs = {f"{j}.pos": 0.0 for j in mesure.JOINTS}
    obs["wrist_roll.pos"] = 170.0
    with pytest.raises(HorsLimites, match="wrist_roll"):
        corr.reel_vers_sim(obs)


def test_sim_action_out_of_range_is_refused(corr):
    q = np.zeros(6)
    q[1] = math.radians(130)  # au-delà de la limite sim de shoulder_lift (100°)
    with pytest.raises(HorsLimites, match="shoulder_lift"):
        corr.sim_vers_reel(q)
    q = np.zeros(6)
    q[1] = math.radians(99)  # dans les limites sim, mais 99 + 6,35 = 105° > plage réelle (100,7° + 3)
    with pytest.raises(HorsLimites, match="réelles"):
        corr.sim_vers_reel(q)


def test_degrees_passed_as_radians_are_refused(corr):
    """Erreur d'unité typique (cause de l'échec de septembre) : des degrés envoyés comme radians."""
    with pytest.raises(HorsLimites):
        corr.sim_vers_reel(np.array([0.0, -10.0, 20.0, 75.0, 0.0, 0.0]))


def test_bad_shapes_and_missing_keys(corr):
    with pytest.raises(ValueError):
        corr.sim_vers_reel(np.zeros(5))
    with pytest.raises(KeyError, match="gripper.pos"):
        corr.reel_vers_sim({f"{j}.pos": 0.0 for j in mesure.BODY})


def test_other_calibration_is_refused(calib):
    other = json.loads(json.dumps(CORR))
    other["calibration"]["wrist_flex"]["range_min"] += 5
    with pytest.raises(ValueError, match="poses D et E"):
        Correspondance.depuis_dict(other, calib)
    incomplete = json.loads(json.dumps(CORR))
    del incomplete["decalage_deg"]["wrist_roll"]
    with pytest.raises(ValueError, match="incomplète"):
        Correspondance.depuis_dict(incomplete, calib)


def test_generator_reproduces_offsets(calib):
    """Le générateur donne exactement les décalages ci-dessus à partir des 5 poses enregistrées."""
    f = chemins.MESURES / "poses_reelles.json"
    if not f.is_file():
        pytest.skip("pas de mesures sur ce poste")
    corr = generer_correspondance.calculer(json.loads(f.read_text(encoding="utf-8")), calib)
    assert corr["decalage_deg"] == pytest.approx(DECAL, abs=0.01)
    assert Correspondance.depuis_dict(corr, calib).decalage_deg == pytest.approx(DECAL, abs=0.01)
