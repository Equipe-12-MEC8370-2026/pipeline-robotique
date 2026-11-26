r"""Tests sans matériel de boucle.py (faux redresseurs, fausse politique).
    ..\..\alexis\.venv-learn\Scripts\python.exe -m pytest -q
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

import boucle as bl  # noqa: E402
import mesure  # noqa: E402
import mouvement as mv  # noqa: E402
from conversion import Correspondance  # noqa: E402
from test_conversion import CORR  # noqa: E402
from test_mesure import GREG  # noqa: E402

POSE_A_SIM = [0.0, -95.0, 85.0, 50.0, 0.0, 0.0]


@pytest.fixture
def corr(tmp_path):
    f = tmp_path / "g.json"
    f.write_text(json.dumps(GREG))
    return Correspondance.depuis_dict(CORR, mesure.load_calibration(f))


class FausseePolitique:
    def __init__(self, action_deg):
        self.action, self.recu = np.radians(action_deg), []

    def act(self, obs):
        self.recu.append(obs)
        return self.action.astype(np.float32)


def faux_red(img):  # « redresse » : BGR -> RGB 640x480
    return np.ascontiguousarray(img[..., ::-1])


BGR = np.zeros((480, 640, 3), np.uint8)
BGR[..., 0] = 200  # bleu


def test_policy_gets_sim_state_and_images(corr):
    pol = FausseePolitique(POSE_A_SIM)
    b = bl.Boucle(corr, faux_red, faux_red, pol, observation=False)
    reel = mv.cible_reelle(corr, POSE_A_SIM)
    pas, imgs = b.pas(reel, BGR, BGR)
    obs = pol.recu[0]
    assert obs["observation.state"].dtype == np.float32 and obs["observation.state"].shape == (6,)
    assert np.degrees(obs["observation.state"]) == pytest.approx(POSE_A_SIM, abs=1e-4)
    for cam in ("front", "wrist"):
        im = obs[f"observation.images.{cam}"]
        assert im.shape == (480, 640, 3) and im.dtype == np.uint8 and im[0, 0].tolist() == [0, 0, 200]
    # action = état : rien ne bouge (float32 de la politique -> ~1e-5 °)
    assert pas.arret is None and pas.consigne == pytest.approx(reel, abs=1e-3)


def test_step_is_capped(corr):
    loin = list(POSE_A_SIM)
    loin[1] += 30.0  # la politique demande 30° d'un coup sur l'épaule
    loin[5] = 90.0  # et d'ouvrir la pince en grand
    b = bl.Boucle(corr, faux_red, faux_red, FausseePolitique(loin), observation=False)
    reel = mv.cible_reelle(corr, POSE_A_SIM)
    pas, _ = b.pas(reel, BGR, BGR)
    d = pas.consigne - reel
    assert d[1] == pytest.approx(bl.MAX_PAS_BRAS) and d[5] == pytest.approx(bl.MAX_PAS_PINCE)
    assert set(pas.plafonnees) == {"shoulder_lift", "gripper"}
    assert pas.cible_reelle[1] - reel[1] == pytest.approx(30.0)  # la cible complète est gardée pour le journal


def test_observation_mode_never_commands(corr):
    loin = list(POSE_A_SIM)
    loin[2] -= 20.0
    b = bl.Boucle(corr, faux_red, faux_red, FausseePolitique(loin), observation=True)
    pas, _ = b.pas(mv.cible_reelle(corr, POSE_A_SIM), BGR, BGR)
    assert pas.consigne is None and pas.arret is None and pas.action_sim_rad is not None


def test_out_of_range_action_stops(corr):
    bad = list(POSE_A_SIM)
    bad[2] = 150.0  # au-delà de la limite sim du coude
    b = bl.Boucle(corr, faux_red, faux_red, FausseePolitique(bad), observation=False)
    pas, _ = b.pas(mv.cible_reelle(corr, POSE_A_SIM), BGR, BGR)
    assert pas.consigne is None and "hors limites" in pas.arret


def test_out_of_range_state_stops_without_calling_policy(corr):
    pol = FausseePolitique(POSE_A_SIM)
    b = bl.Boucle(corr, faux_red, faux_red, pol, observation=False)
    repos = np.array([-1.9, -101.2, 96.7, 75.5, -8.3, 8.8])  # repos réel : épaule hors limites sim
    pas, _ = b.pas(repos, BGR, BGR)
    assert pas.consigne is None and "état réel" in pas.arret and pol.recu == []


def test_invalid_policy_output_stops(corr):
    class Cassee:
        def act(self, obs):
            return np.array([math.nan] * 6)

    pas, _ = bl.Boucle(corr, faux_red, faux_red, Cassee(), observation=False).pas(
        mv.cible_reelle(corr, POSE_A_SIM), BGR, BGR)
    assert pas.consigne is None and "invalide" in pas.arret


def test_cap_is_relative_to_reference(corr):
    """Plafond relatif à la consigne précédente : les consignes avancent régulièrement même si le bras traîne."""
    loin = list(POSE_A_SIM)
    loin[1] += 30.0
    b = bl.Boucle(corr, faux_red, faux_red, FausseePolitique(loin), observation=False)
    reel = mv.cible_reelle(corr, POSE_A_SIM)
    pas, _ = b.proposer(reel, BGR, BGR)
    assert pas.consigne is None and pas.cible_reelle is not None
    ref = reel.copy()
    ref[1] += 2.0  # consigne précédente déjà 2° en avance sur le bras (qui traîne)
    pas = b.commander(pas, ref)
    assert pas.consigne[1] - reel[1] == pytest.approx(2.0 + bl.MAX_PAS_BRAS)


class PolitiqueLente:
    """Comme policy_server : une inférence lente (0,25 s) toutes les 5 actions, sinon instantanée."""

    def __init__(self, action_deg, tous=5, duree=0.25):
        self.action, self.tous, self.duree, self.n = np.radians(action_deg), tous, duree, 0

    def act(self, obs):
        import time

        if self.n % self.tous == 0:
            time.sleep(self.duree)
        self.n += 1
        return self.action.astype(np.float32)


def test_async_policy_hides_inference_pauses(corr):
    import time

    b = bl.Boucle(corr, faux_red, faux_red, PolitiqueLente(POSE_A_SIM), observation=False)
    reel = mv.cible_reelle(corr, POSE_A_SIM)
    pa = bl.PolitiqueAsynchrone(b, lambda: (reel, BGR, BGR), avance=3)
    try:
        time.sleep(0.5)  # remplissage initial (comme la chauffe)
        famine = 0
        for _ in range(30):  # 3 s à 10 Hz
            if pa.suivant() is None:
                famine += 1
            time.sleep(0.1)
        assert famine == 0  # 3 actions d'avance (0,3 s) > pause d'inférence (0,25 s)
    finally:
        pa.arreter()


def test_async_policy_propagates_stop_and_errors(corr):
    import time

    bad = list(POSE_A_SIM)
    bad[2] = 150.0
    reel = mv.cible_reelle(corr, POSE_A_SIM)
    pa = bl.PolitiqueAsynchrone(bl.Boucle(corr, faux_red, faux_red, FausseePolitique(bad), False),
                                lambda: (reel, BGR, BGR))
    time.sleep(0.2)
    pas, _ = pa.suivant()
    assert "hors limites" in pas.arret and pa.n_appels == 1  # le fil s'arrête au premier refus
    pa.arreter()

    class Plante:
        def act(self, obs):
            raise ConnectionError("serveur perdu")

    pa = bl.PolitiqueAsynchrone(bl.Boucle(corr, faux_red, faux_red, Plante(), False), lambda: (reel, BGR, BGR))
    time.sleep(0.2)
    with pytest.raises(RuntimeError, match="serveur perdu"):
        pa.suivant()
    pa.arreter()


def test_limiter_pas_no_clip_when_small():
    c, plaf = bl.limiter_pas([1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 0])
    assert c.tolist() == [1, 2, 3, 4, 5, 6] and plaf == []
