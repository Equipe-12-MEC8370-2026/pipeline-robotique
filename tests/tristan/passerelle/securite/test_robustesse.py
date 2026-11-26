r"""Tests sans matériel : retour lent et coupure du couple quand la liaison USB tombe (faux bus).
    ..\..\alexis\.venv-learn\Scripts\python.exe -m pytest -q
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

import aller_a_pose as ap  # noqa: E402
import mesure  # noqa: E402
from test_mesure import GREG  # noqa: E402


class FauxBus:
    """Suit les consignes à la perfection ; peut « perdre la liaison » après N lectures."""

    instances: list = []

    def __init__(self, calib, panne_apres=None, raw0=None):
        self.calib = calib
        self.raw = dict(raw0) if raw0 else {j: int(c.mid) for j, c in calib.items()}
        self.panne_apres, self.lectures, self.couple = panne_apres, 0, True
        FauxBus.instances.append(self)

    def connect(self, handshake=True):
        pass

    def disconnect(self, disable_torque=False):
        pass

    def _panne(self):
        if self.panne_apres is not None and self.lectures >= self.panne_apres:
            raise OSError("ClearCommError failed (Accès refusé)")

    def sync_read(self, reg, normalize=False, num_retry=0):
        self._panne()
        self.lectures += 1
        return dict(self.raw)

    def sync_write(self, reg, vals, normalize=True, num_retry=0):
        self._panne()
        if reg == "Goal_Position" and normalize:
            for j, v in vals.items():
                c = self.calib[j]
                if j == "gripper":
                    self.raw[j] = int(round(c.range_min + v / 100 * (c.range_max - c.range_min)))
                else:
                    self.raw[j] = int(round(v * 4095 / 360 + c.mid))

    def disable_torque(self, num_retry=0):
        self._panne()
        self.couple = False


@pytest.fixture
def calib(tmp_path):
    f = tmp_path / "g.json"
    f.write_text(json.dumps(GREG))
    return mesure.load_calibration(f)


@pytest.fixture(autouse=True)
def reset():
    FauxBus.instances.clear()


def test_return_survives_usb_loss(calib, monkeypatch):
    depart = np.array([0.0, -60.0, 60.0, 60.0, 0.0, 10.0])
    bus = FauxBus(calib, panne_apres=3)
    monkeypatch.setattr(ap, "make_bus", lambda port, c: FauxBus(c))
    monkeypatch.setattr(ap.time, "sleep", lambda s: None)
    journal = []
    nouveau = ap.retour_robuste(bus, "COM3", calib, depart, duree=0.05, hz=1000.0, journal=journal)
    assert nouveau is not bus  # reconnecté sur un nouveau bus
    assert any(e["phase"] == "liaison_perdue" for e in journal)
    _, fin = ap.lire_reel(nouveau, calib)
    assert fin[:5] == pytest.approx(depart[:5], abs=0.2)  # retour terminé sur le nouveau bus


def test_return_gives_up_cleanly_when_link_never_comes_back(calib, monkeypatch):
    def toujours_en_panne(port, c):
        raise OSError("port absent")

    monkeypatch.setattr(ap, "make_bus", toujours_en_panne)
    monkeypatch.setattr(ap.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="couper_couple"):
        ap.retour_robuste(FauxBus(calib, panne_apres=0), "COM3", calib, np.zeros(6), 0.05, 1000.0, [])


def test_torque_cut_survives_usb_loss(calib, monkeypatch):
    monkeypatch.setattr(ap, "make_bus", lambda port, c: FauxBus(c))
    monkeypatch.setattr(ap.time, "sleep", lambda s: None)
    bus = ap.couper_couple_robuste(FauxBus(calib, panne_apres=0), "COM3", calib)
    assert bus.couple is False
