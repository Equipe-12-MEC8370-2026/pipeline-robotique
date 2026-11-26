"""Fonctions pures (sans matériel) du test 1 de la passerelle : lecture brute des moteurs -> unités LeRobot ->
hypothèses d'angles « sim ».

Conventions LeRobot 0.6.1, robot ``so101_follower`` (``use_degrees=True`` par défaut) :
  * 5 articulations du bras : degrés = (brut - milieu) * 360 / 4095, milieu = (range_min + range_max) / 2
    (pas de bornage) ;
  * pince : [0, 100] = (brut borné - range_min) / (range_max - range_min) * 100.
Les plages viennent du fichier de calibration (ex. ``Greg.json``). ``drive_mode`` est ignoré en degrés par LeRobot
(il vaut 0 partout chez nous de toute façon).

Deux hypothèses pour passer aux angles de la sim (degrés, zéro de l'URDF) — le test 1 dit laquelle est juste :
  * ``leisaac`` : la plage calibrée réelle est étirée linéairement sur les limites articulaires de la sim
    (c'est ce que fait ``cta_learn.core.units`` pour le format « lerobot ») ;
  * ``degres`` : degrés LeRobot = degrés sim (même zéro, même sens, même échelle).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
BODY = JOINTS[:5]
STS3215_MAX_RES = 4095  # résolution 4096 pas/tour - 1, comme LeRobot

# limites articulaires de la sim (degrés), configs/embodiments/so101.yaml d'Alexis
SIM_LIMITS_DEG = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 90.0),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-160.0, 160.0),
    "gripper": (-10.0, 100.0),
}


@dataclass(frozen=True)
class Calib:
    id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int

    @property
    def mid(self) -> float:
        return (self.range_min + self.range_max) / 2


def load_calibration(path: str | Path) -> dict[str, Calib]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [j for j in JOINTS if j not in raw]
    if missing:
        raise ValueError(f"calibration {path}: articulations manquantes {missing}")
    calib = {j: Calib(**raw[j]) for j in JOINTS}
    for j, c in calib.items():
        if not c.range_min < c.range_max:
            raise ValueError(f"calibration {path}: {j} range_min >= range_max")
    return calib


def raw_to_lerobot(raw: dict[str, int], calib: dict[str, Calib]) -> dict[str, float]:
    """Valeurs que ``so101_follower`` (use_degrees=True) rapporterait : degrés pour le bras, [0, 100] pour la pince."""
    out = {}
    for j in BODY:
        out[j] = (raw[j] - calib[j].mid) * 360.0 / STS3215_MAX_RES
    c = calib["gripper"]
    g = min(c.range_max, max(c.range_min, raw["gripper"]))
    out["gripper"] = (g - c.range_min) / (c.range_max - c.range_min) * 100.0
    return out


def raw_to_m100(raw: dict[str, int], calib: dict[str, Calib]) -> dict[str, float]:
    """Plage normalisée LeRobot [-100, 100] (bras, borné) / [0, 100] (pince), comme use_degrees=False."""
    out = {}
    for j in BODY:
        c = calib[j]
        v = min(c.range_max, max(c.range_min, raw[j]))
        out[j] = (v - c.range_min) / (c.range_max - c.range_min) * 200.0 - 100.0
    out["gripper"] = raw_to_lerobot(raw, calib)["gripper"]
    return out


def sim_deg_leisaac(raw: dict[str, int], calib: dict[str, Calib]) -> dict[str, float]:
    """Hypothèse « leisaac » : plage calibrée étirée sur les limites sim (= cta_learn.core.units, sans validation)."""
    m = raw_to_m100(raw, calib)
    out = {}
    for j in JOINTS:
        lo, hi = SIM_LIMITS_DEG[j]
        frac = (m[j] + 100.0) / 200.0 if j in BODY else m[j] / 100.0
        out[j] = lo + frac * (hi - lo)
    return out


def sim_deg_degres(raw: dict[str, int], calib: dict[str, Calib]) -> dict[str, float]:
    """Hypothèse « degres » : degrés LeRobot = degrés sim (bras) ; pince : comme « leisaac » (pas d'équivalent)."""
    d = raw_to_lerobot(raw, calib)
    out = {j: d[j] for j in BODY}
    out["gripper"] = sim_deg_leisaac(raw, calib)["gripper"]
    return out


def calibration_mismatches(file_calib: dict[str, Calib], motor_calib: dict[str, dict]) -> list[str]:
    """Différences entre le fichier de calibration et ce que les moteurs ont en mémoire (homing_offset, min, max)."""
    diffs = []
    for j in JOINTS:
        for key in ("homing_offset", "range_min", "range_max"):
            a, b = getattr(file_calib[j], key), motor_calib[j][key]
            if a != b:
                diffs.append(f"{j}.{key}: fichier {a} != moteur {b}")
    return diffs


def errors_vs_target(measured_sim_deg: dict[str, float], target_deg: dict[str, float]) -> dict[str, float]:
    return {j: measured_sim_deg[j] - target_deg[j] for j in target_deg}


def fit_sign_offset(real_deg: np.ndarray, sim_deg: np.ndarray) -> tuple[float, float, float]:
    """Ajuste sim = pente * réel + décalage (moindres carrés) ; renvoie (pente, décalage, résidu max).
    Il faut au moins 2 poses distinctes ; pente ~ +1 ou -1 attendue (signe), décalage en degrés."""
    real_deg, sim_deg = np.asarray(real_deg, float), np.asarray(sim_deg, float)
    if real_deg.size < 2 or np.ptp(real_deg) < 1e-6:
        raise ValueError("il faut au moins 2 poses avec des angles réels différents")
    a, b = np.polyfit(real_deg, sim_deg, 1)
    resid = float(np.max(np.abs(a * real_deg + b - sim_deg)))
    return float(a), float(b), resid
