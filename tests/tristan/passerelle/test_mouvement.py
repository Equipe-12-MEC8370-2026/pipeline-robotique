r"""Tests sans matériel de mouvement.py (depuis tests/tristan/passerelle) :
    ..\..\alexis\.venv-learn\Scripts\python.exe -m pytest -q
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mesure  # noqa: E402
import mouvement as mv  # noqa: E402
from conversion import Correspondance, HorsLimites  # noqa: E402
from test_conversion import CORR  # noqa: E402
from test_mesure import GREG  # noqa: E402

REPOS = np.array([-1.3, -100.7, 96.8, 74.8, -7.6, 11.7])  # pose repliée réelle relevée le 2026-11-25
POSE_D_SIM = [0.0, -14.39, 15.92, 85.02, 0.0, 0.0]


@pytest.fixture
def corr(tmp_path):
    f = tmp_path / "greg.json"
    f.write_text(json.dumps(GREG))
    return Correspondance.depuis_dict(CORR, mesure.load_calibration(f))


def test_trajectory_reaches_target_smoothly(corr):
    cible = mv.cible_reelle(corr, POSE_D_SIM)
    traj = mv.planifier(REPOS, cible, duree_s=6.0, hz=30.0)
    assert len(traj) >= 180
    assert traj[-1] == pytest.approx(cible)
    full = np.vstack([REPOS, traj])
    d = np.diff(full, axis=0)
    assert np.abs(d).max() <= mv.PAS_MAX
    # profil cosinus : premier et dernier pas petits (pas d'à-coup)
    assert np.abs(d[0]).max() < 0.05 and np.abs(d[-1]).max() < 0.05
    # monotone par articulation (ligne droite, pas de dépassement)
    for k in range(6):
        assert np.all(d[:, k] * np.sign(cible[k] - REPOS[k]) >= -1e-12)
    mv.verifier_dans_plage_reelle(corr, traj)


def test_long_moves_get_more_steps():
    a, b = np.zeros(6), np.array([200.0, 0, 0, 0, 0, 0])
    traj = mv.planifier(a, b, duree_s=1.0, hz=30.0)
    assert np.abs(np.diff(np.vstack([a, traj]), axis=0)).max() <= mv.PAS_MAX


def test_target_goes_through_bridge_and_its_limits(corr):
    cible = mv.cible_reelle(corr, POSE_D_SIM)
    # décalages : réel = sim - décalage
    assert cible[1] == pytest.approx(-14.39 + 6.35) and cible[3] == pytest.approx(85.02 + 6.39)
    with pytest.raises(HorsLimites):
        mv.cible_reelle(corr, [0.0, -130.0, 0.0, 0.0, 0.0, 0.0])


def test_step_guard():
    with pytest.raises(mv.MouvementRefuse, match="shoulder_lift"):
        mv.verifier_pas(np.array([np.zeros(6), [0, 3.0, 0, 0, 0, 0]]))
    with pytest.raises(mv.MouvementRefuse):
        mv.planifier(np.zeros(6), np.full(6, np.nan), 6.0, 30.0)


def test_real_range_guard(corr):
    bad = REPOS.copy()
    bad[2] = 150.0
    with pytest.raises(HorsLimites, match="elbow_flex"):
        mv.verifier_dans_plage_reelle(corr, bad)


def test_no_motion_when_already_there():
    traj = mv.planifier(REPOS, REPOS, 6.0, 30.0)
    assert np.allclose(traj, REPOS)


def test_tracking_error_ignores_gripper():
    e, j = mv.ecart_suivi([0, 0, 0, 0, 0, 0], [0, 0, 20.0, 0, 0, 90.0])
    assert (e, j) == (20.0, "elbow_flex")


def test_vector_dict_round_trip():
    d = mv.dictionnaire(REPOS)
    assert list(d) == [f"{j}.pos" for j in mesure.JOINTS]
    assert mv.vecteur(d) == pytest.approx(REPOS)
