"""Fonctions pures (sans matériel) des mouvements commandés : trajectoire lente et garde-fous.

Toutes les valeurs sont au format LeRobot réel (degrés pour le bras, [0, 100] pour la pince), dans l'ordre JOINTS.
La correspondance sim <-> réel étant un décalage (échelle 1), une ligne droite en valeurs réelles est aussi une
ligne droite en angles sim : la trajectoire est la même dans les deux espaces.
"""

from __future__ import annotations

import math

import numpy as np

from conversion import TOL_DEG, Correspondance, HorsLimites
from mesure import JOINTS

PAS_MAX = 2.0  # degrés (ou unités pince) par pas de commande, au plus
ECART_SUIVI_MAX = 15.0  # écart consigne / position mesurée au-delà duquel on fige le bras (blocage, collision)


class MouvementRefuse(ValueError):
    """Trajectoire ou pas de commande refusé par un garde-fou."""


def vecteur(valeurs: dict[str, float]) -> np.ndarray:
    """{"<art>.pos" ou "<art>": valeur} -> vecteur ordre JOINTS."""
    return np.array([float(valeurs.get(f"{j}.pos", valeurs.get(j))) for j in JOINTS])


def dictionnaire(v) -> dict[str, float]:
    return {f"{j}.pos": float(x) for j, x in zip(JOINTS, v)}


def planifier(depart, cible, duree_s: float, hz: float, pas_max: float = PAS_MAX) -> np.ndarray:
    """Trajectoire en ligne droite de ``depart`` à ``cible`` (valeurs réelles), profil en cosinus (départ et arrivée
    sans à-coup), au moins ``duree_s`` secondes et jamais plus de ``pas_max`` par pas. Renvoie (N, 6), sans le départ,
    dernière ligne = cible."""
    depart, cible = np.asarray(depart, float), np.asarray(cible, float)
    if depart.shape != (len(JOINTS),) or cible.shape != (len(JOINTS),):
        raise ValueError("départ et cible : 6 valeurs, ordre JOINTS")
    if not (np.all(np.isfinite(depart)) and np.all(np.isfinite(cible))):
        raise MouvementRefuse("valeur non finie")
    delta = cible - depart
    # profil cosinus : vitesse max = (pi/2) * vitesse moyenne -> nombre de pas pour respecter pas_max
    n_vitesse = math.ceil(np.max(np.abs(delta)) * (math.pi / 2) / pas_max) if np.any(delta) else 1
    n = max(int(math.ceil(duree_s * hz)), n_vitesse, 1)
    s = (1 - np.cos(np.pi * np.arange(1, n + 1) / n)) / 2
    traj = depart + s[:, None] * delta
    verifier_pas(np.vstack([depart, traj]), pas_max)
    return traj


def verifier_pas(traj, pas_max: float = PAS_MAX) -> None:
    """Refuse toute trajectoire dont un pas dépasse ``pas_max`` sur une articulation."""
    d = np.abs(np.diff(np.asarray(traj, float), axis=0))
    if d.size and d.max() > pas_max + 1e-9:
        i, j = np.unravel_index(int(d.argmax()), d.shape)
        raise MouvementRefuse(f"pas de {d[i, j]:.2f} > {pas_max} sur {JOINTS[j]} (pas {i})")


def verifier_dans_plage_reelle(corr: Correspondance, traj) -> None:
    """Chaque point doit rester dans la plage calibrée réelle (± tolérance)."""
    for k, p in enumerate(np.atleast_2d(traj)):
        for j, v in zip(JOINTS, p):
            if not corr.reel_min_deg[j] - TOL_DEG <= v <= corr.reel_max_deg[j] + TOL_DEG:
                raise HorsLimites(f"point {k} : {j}={v:.1f} hors plage réelle "
                                  f"[{corr.reel_min_deg[j]:.1f}, {corr.reel_max_deg[j]:.1f}]")


def cible_reelle(corr: Correspondance, q_sim_deg) -> np.ndarray:
    """Pose sim (degrés) -> valeurs réelles, en passant par la passerelle (refus hors limites sim ou réelles)."""
    return vecteur(corr.sim_vers_reel(np.radians(np.asarray(q_sim_deg, float))))


def ecart_suivi(commande, mesure_reelle) -> tuple[float, str]:
    """Plus grand écart consigne / mesure (hors pince, qui peut légitimement bloquer sur un objet)."""
    d = np.abs(np.asarray(commande, float) - np.asarray(mesure_reelle, float))[:5]
    i = int(d.argmax())
    return float(d[i]), JOINTS[i]
