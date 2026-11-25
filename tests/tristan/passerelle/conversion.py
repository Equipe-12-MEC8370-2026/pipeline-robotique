"""Passerelle sim <-> SO-101 réel : conversion des positions articulaires, dans les deux sens.

    réel -> sim : observation LeRobot ``so101_follower`` (``{"shoulder_pan.pos": degrés, …, "gripper.pos": 0-100}``)
                  -> vecteur d'angles sim en RADIANS, ordre de l'embodiment (entrée de la politique) ;
    sim -> réel : action de la politique (radians sim) -> consignes LeRobot (même format que l'observation).

Correspondance mesurée (``correspondance_greg.json``, générée par ``generer_correspondance.py``) :
  * 5 articulations du bras : ``angle_sim_deg = angle_lerobot_deg + decalage_deg`` (échelle 1, aucun signe inversé :
    vérifié sur 5 poses le 2026-11-25) ;
  * pince : ``angle_sim_deg = sim_min + (valeur_0_100 / 100) * (sim_max - sim_min)`` (linéaire, [0, 100] -> [-10, 100]°).

Sécurité : toute valeur hors des limites (plage calibrée réelle d'un côté, limites articulaires sim de l'autre,
plus une petite tolérance) lève ``HorsLimites`` au lieu d'être corrigée en silence. La correspondance est liée à
une calibration précise : si ``Greg.json`` change (recalibration), le chargement échoue -> refaire les poses D et E.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mesure import BODY, JOINTS, SIM_LIMITS_DEG, STS3215_MAX_RES, Calib

TOL_DEG = 3.0  # tolérance au-delà des limites (bruit des encodeurs, dépassement léger en sim)


class HorsLimites(ValueError):
    """Valeur articulaire hors de la plage autorisée (mauvaise unité, mauvais ordre, ou consigne dangereuse)."""


@dataclass(frozen=True)
class Correspondance:
    decalage_deg: dict[str, float]  # bras : sim = réel + décalage
    pince_sim_deg: tuple[float, float]  # angles sim pour 0 et 100 (réel)
    calibration: dict[str, dict]  # homing_offset, range_min, range_max par articulation (empreinte)
    reel_min_deg: dict[str, float]
    reel_max_deg: dict[str, float]

    # ------------------------------------------------------------------ construction / chargement
    @staticmethod
    def depuis_dict(d: dict, calib: dict[str, Calib] | None = None) -> Correspondance:
        manque = [j for j in BODY if j not in d.get("decalage_deg", {})] + \
                 [j for j in JOINTS if j not in d.get("calibration", {})]
        if manque or "pince_sim_deg" not in d:
            raise ValueError(f"correspondance incomplète : manque {manque or ['pince_sim_deg']}")
        if calib is not None:
            for j in JOINTS:
                for k in ("homing_offset", "range_min", "range_max"):
                    if d["calibration"][j][k] != getattr(calib[j], k):
                        raise ValueError(
                            f"la correspondance a été mesurée avec une autre calibration ({j}.{k} : "
                            f"{d['calibration'][j][k]} != {getattr(calib[j], k)}) -> refaire les poses D et E")
        cal = d["calibration"]
        half = {j: (cal[j]["range_max"] - cal[j]["range_min"]) / 2 * 360.0 / STS3215_MAX_RES for j in BODY}
        lo, hi = {j: -half[j] for j in BODY}, dict(half)
        lo["gripper"], hi["gripper"] = 0.0, 100.0
        return Correspondance({j: float(d["decalage_deg"][j]) for j in BODY},
                              tuple(float(v) for v in d["pince_sim_deg"]), cal, lo, hi)

    @staticmethod
    def charger(path: str | Path, calib: dict[str, Calib] | None = None) -> Correspondance:
        return Correspondance.depuis_dict(json.loads(Path(path).read_text(encoding="utf-8")), calib)

    # ------------------------------------------------------------------ vérifications
    def _verif_reel(self, valeurs: dict[str, float]) -> None:
        bad = [f"{j}={v:.1f} hors [{self.reel_min_deg[j]:.1f}, {self.reel_max_deg[j]:.1f}]"
               for j, v in valeurs.items()
               if not self.reel_min_deg[j] - TOL_DEG <= v <= self.reel_max_deg[j] + TOL_DEG or math.isnan(v)]
        if bad:
            raise HorsLimites("valeurs réelles hors de la plage calibrée : " + ", ".join(bad))

    @staticmethod
    def _verif_sim(deg: dict[str, float]) -> None:
        bad = [f"{j}={v:.1f}° hors [{SIM_LIMITS_DEG[j][0]:.0f}, {SIM_LIMITS_DEG[j][1]:.0f}]"
               for j, v in deg.items()
               if not SIM_LIMITS_DEG[j][0] - TOL_DEG <= v <= SIM_LIMITS_DEG[j][1] + TOL_DEG or math.isnan(v)]
        if bad:
            raise HorsLimites("angles sim hors des limites articulaires : " + ", ".join(bad))

    # ------------------------------------------------------------------ conversions
    def reel_vers_sim(self, obs: dict[str, float]) -> np.ndarray:
        """Observation LeRobot (clés ``<articulation>.pos``) -> angles sim en radians, ordre JOINTS."""
        manque = [j for j in JOINTS if f"{j}.pos" not in obs]
        if manque:
            raise KeyError(f"observation incomplète : manque {[m + '.pos' for m in manque]}")
        reel = {j: float(obs[f"{j}.pos"]) for j in JOINTS}
        self._verif_reel(reel)
        lo, hi = self.pince_sim_deg
        sim = {j: reel[j] + self.decalage_deg[j] for j in BODY}
        sim["gripper"] = lo + reel["gripper"] / 100.0 * (hi - lo)
        self._verif_sim(sim)
        return np.radians([sim[j] for j in JOINTS])

    def sim_vers_reel(self, q_rad) -> dict[str, float]:
        """Action sim (radians, ordre JOINTS) -> consignes LeRobot ``{"<articulation>.pos": valeur}``."""
        q = np.asarray(q_rad, dtype=float).reshape(-1)
        if q.shape != (len(JOINTS),):
            raise ValueError(f"attendu {len(JOINTS)} angles (ordre {JOINTS}), reçu {q.shape}")
        sim = dict(zip(JOINTS, np.degrees(q).tolist()))
        self._verif_sim(sim)
        lo, hi = self.pince_sim_deg
        reel = {j: sim[j] - self.decalage_deg[j] for j in BODY}
        reel["gripper"] = (sim["gripper"] - lo) / (hi - lo) * 100.0
        self._verif_reel(reel)
        return {f"{j}.pos": reel[j] for j in JOINTS}
