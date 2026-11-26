"""Logique (sans matériel) d'un pas de la boucle politique sim -> vrai robot.

Un pas : état réel (valeurs LeRobot) -> passerelle -> radians sim ; images réelles -> redressées comme la sim ;
politique (policy_server d'Alexis) -> action en radians sim -> passerelle -> valeurs réelles -> plafond de
déplacement par pas -> consigne (ou rien en mode observation). Tout refus de la passerelle ARRÊTE la boucle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from conversion import Correspondance, HorsLimites
from mesure import JOINTS
from mouvement import dictionnaire, vecteur

MAX_PAS_BRAS = 3.0  # degrés par pas de commande (sim à 30 Hz : <= 0,95°/action pour l'expert)
MAX_PAS_PINCE = 10.0  # unités 0-100 par pas (sim : pince jusqu'à ~5 unités/action)


def limiter_pas(cible, actuel, max_bras: float = MAX_PAS_BRAS, max_pince: float = MAX_PAS_PINCE):
    """Rapproche ``cible`` de ``actuel`` pour qu'aucune articulation ne bouge de plus que le plafond.
    Renvoie (consigne, liste des articulations plafonnées)."""
    cible, actuel = np.asarray(cible, float), np.asarray(actuel, float)
    plafond = np.array([max_bras] * 5 + [max_pince])
    delta = np.clip(cible - actuel, -plafond, plafond)
    limites = [j for j, c, d in zip(JOINTS, cible - actuel, delta) if abs(c - d) > 1e-9]
    return actuel + delta, limites


@dataclass
class Pas:
    etat_sim_rad: np.ndarray | None = None
    action_sim_rad: np.ndarray | None = None
    cible_reelle: np.ndarray | None = None
    consigne: np.ndarray | None = None  # None : rien à envoyer (observation ou arrêt)
    plafonnees: list = field(default_factory=list)
    arret: str | None = None


class Boucle:
    def __init__(self, corr: Correspondance, red_front, red_wrist, client, observation: bool,
                 max_bras: float = MAX_PAS_BRAS, max_pince: float = MAX_PAS_PINCE):
        self.corr, self.red = corr, {"front": red_front, "wrist": red_wrist}
        self.client, self.observation = client, observation
        self.max_bras, self.max_pince = max_bras, max_pince

    def images(self, bgr_front, bgr_wrist) -> dict[str, np.ndarray]:
        """Images envoyées à la politique : RGB uint8 (480, 640, 3), noms de la sim."""
        return {"front": self.red["front"](bgr_front), "wrist": self.red["wrist"](bgr_wrist)}

    def pas(self, reel, bgr_front, bgr_wrist) -> tuple[Pas, dict[str, np.ndarray]]:
        """Proposer puis commander, plafond relatif à la position mesurée (version synchrone)."""
        pas, imgs = self.proposer(reel, bgr_front, bgr_wrist)
        return self.commander(pas, reel), imgs

    def commander(self, pas: Pas, reference) -> Pas:
        """Plafond par pas relatif à ``reference`` (consigne précédente, ou position mesurée au premier pas)."""
        if pas.arret or pas.cible_reelle is None:
            return pas
        consigne, plaf = limiter_pas(pas.cible_reelle, reference, self.max_bras, self.max_pince)
        pas.consigne, pas.plafonnees = (None if self.observation else consigne), plaf
        return pas

    def proposer(self, reel, bgr_front, bgr_wrist) -> tuple[Pas, dict[str, np.ndarray]]:
        """Observation -> politique -> passerelle : cible réelle (sans plafond ni consigne) ou raison d'arrêt."""
        reel = np.asarray(reel, float)
        imgs = self.images(bgr_front, bgr_wrist)
        try:
            q = self.corr.reel_vers_sim(dictionnaire(reel))
        except HorsLimites as e:
            return Pas(arret=f"état réel hors des limites sim : {e}"), imgs
        a = np.asarray(self.client.act({
            "observation.state": q.astype(np.float32),
            "observation.images.front": imgs["front"],
            "observation.images.wrist": imgs["wrist"],
        }), float).reshape(-1)
        if a.shape != (len(JOINTS),) or not np.all(np.isfinite(a)):
            return Pas(etat_sim_rad=q, arret=f"action invalide de la politique : forme {a.shape}"), imgs
        try:
            cible = vecteur(self.corr.sim_vers_reel(a))
        except HorsLimites as e:
            return Pas(etat_sim_rad=q, action_sim_rad=a, arret=f"action hors limites : {e}"), imgs
        return Pas(q, a, cible), imgs


class PolitiqueAsynchrone:
    """Interroge la politique dans un fil séparé, ``avance`` actions à l'avance : la boucle de commande dépile à
    rythme régulier et ne bloque plus pendant l'inférence (~200 ms toutes les n_action_steps actions).
    ``source()`` renvoie (position réelle publiée par la boucle, image front BGR, image wrist BGR) ou None ;
    ce fil ne touche JAMAIS au bus du bras."""

    def __init__(self, boucle: Boucle, source, avance: int = 3):
        import threading
        from collections import deque

        self.boucle, self.source, self.avance = boucle, source, avance
        self.file: deque = deque()
        self.erreur: Exception | None = None
        self.n_appels = 0
        self._verrou = threading.Lock()
        self._actif = True
        self._fil = threading.Thread(target=self._travail, daemon=True)
        self._fil.start()

    def _travail(self):
        import time

        while self._actif:
            with self._verrou:
                plein = len(self.file) >= self.avance
            src = None if plein else self.source()
            if src is None:
                time.sleep(0.005)
                continue
            try:
                pas, imgs = self.boucle.proposer(*src)
            except Exception as e:  # noqa: BLE001  (transmise à la boucle par suivant())
                self.erreur = e
                return
            self.n_appels += 1
            with self._verrou:
                self.file.append((pas, imgs))
            if pas.arret:
                return

    def suivant(self):
        """Prochaine (Pas, images) dans l'ordre, ou None si rien n'est prêt (la boucle garde alors sa consigne)."""
        if self.erreur is not None:
            raise RuntimeError(f"politique : {self.erreur}") from self.erreur
        with self._verrou:
            return self.file.popleft() if self.file else None

    def arreter(self):
        self._actif = False
        self._fil.join(timeout=2.0)
