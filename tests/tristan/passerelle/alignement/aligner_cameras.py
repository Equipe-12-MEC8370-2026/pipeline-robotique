r"""Alignement en direct d'une vraie caméra sur sa vue de référence sim (cameras\ref_<pose>_<camera>.png).

L'image réelle est redressée comme la sim (calibration cameras\calib_<camera>.json), puis superposée à la
référence. Ne touche pas au robot : pour que la scène soit la même, faire tenir la pose au bras dans un AUTRE
terminal (aller_a_pose.py --pose D) et poser cube + cible comme au test 2.

Touches (fenêtre au premier plan) :
  [m] mode : fondu 50 % -> contours sim (vert) sur image réelle -> côte à côte
  [f] / [w] caméra avant / poignet     [r] tourner l'image réelle de 90° (diagnostic, affichage seulement)
  [s] enregistrer une capture (cameras\alignement_*.png)              [q] quitter
Correction d'orientation PAR LOGICIEL (support non réglable), pas de 0,5° :
  [j]/[l] lacet (gauche/droite)   [i]/[k] tangage (haut/bas)   [u]/[o] roulis   [x] remettre à zéro
  [e] enregistrer la correction dans cameras\calib_<camera>.json (utilisée ensuite par la boucle SmolVLA)
Score affiché : % des contours de la référence qui tombent à moins de 3 px d'un contour réel (plus haut = mieux ;
ne vise pas 100 %, la texture du bois et l'éclairage diffèrent).

Depuis tests\tristan\passerelle, dans TON terminal :
    ..\..\alexis\.venv-learn\Scripts\python.exe alignement\aligner_cameras.py --camera front
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chemins  # noqa: E402  (rend les autres dossiers de la passerelle importables)
import cameras as cam  # noqa: E402
from affichage import Fenetre  # noqa: E402

MODES = ("fondu", "contours", "cote a cote")


def contours(bgr: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    return cv2.Canny(g, 50, 150)


def score_alignement(ref_contours: np.ndarray, reel_bgr: np.ndarray, tol_px: float = 3.0) -> float:
    """Part (0-1) des pixels de contour de la référence situés à moins de ``tol_px`` d'un contour réel."""
    reel = contours(reel_bgr)
    if not ref_contours.any() or not reel.any():
        return 0.0
    dist = cv2.distanceTransform(255 - reel, cv2.DIST_L2, 3)
    return float((dist[ref_contours > 0] < tol_px).mean())


def composer(ref_bgr, reel_bgr, ref_cont, mode: str) -> np.ndarray:
    if mode == "fondu":
        return cv2.addWeighted(ref_bgr, 0.5, reel_bgr, 0.5, 0)
    if mode == "contours":
        out = reel_bgr.copy()
        out[ref_cont > 0] = (0, 255, 0)
        return out
    return np.hstack([ref_bgr, reel_bgr])


PAS_DEG = 0.5
TOUCHES_CORR = {"j": ("lacet", -PAS_DEG), "l": ("lacet", PAS_DEG), "i": ("tangage", -PAS_DEG),
                "k": ("tangage", PAS_DEG), "u": ("roulis", -PAS_DEG), "o": ("roulis", PAS_DEG)}


class Camera:
    def __init__(self, nom: str, pose: str):
        self.path = chemins.VISION / f"calib_{nom}.json"
        self.calib = cam.charger_calibration(self.path)
        self.corr = {k: float((self.calib.get("correction_deg") or {}).get(k, 0.0))
                     for k in ("lacet", "tangage", "roulis")}
        self.nom, self.red = nom, cam.Redresseur(self.calib)
        self.cap = cam.ouvrir(int(self.calib["index_opencv"]), *self.calib["taille"])
        self.ref = cv2.imread(str(chemins.ALIGNEMENT / f"ref_{pose}_{nom}.png"))
        if self.ref is None:
            raise FileNotFoundError(f"référence absente : cameras/ref_{pose}_{nom}.png (lancer rendre_vues_sim.py)")
        self.ref_cont = contours(self.ref)

    def corriger(self, axe: str | None = None, pas: float = 0.0, zero: bool = False) -> None:
        if zero:
            self.corr = dict.fromkeys(self.corr, 0.0)
        elif axe:
            self.corr[axe] = round(self.corr[axe] + pas, 2)
        self.red = cam.Redresseur(dict(self.calib, correction_deg=self.corr))

    def enregistrer_correction(self) -> None:
        self.calib["correction_deg"] = dict(self.corr)
        self.path.write_text(json.dumps(self.calib, indent=1), encoding="utf-8")

    def fermer(self):
        self.cap.release()


def texte(img, s, y):
    cv2.putText(img, s, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", choices=["front", "wrist"], default="front")
    p.add_argument("--pose", default="D")
    args = p.parse_args()
    c = Camera(args.camera, args.pose.upper())
    fen = Fenetre("alignement camera / reference sim")
    mode, rot, n = 0, 0, 0
    try:
        while True:
            reel = cv2.cvtColor(c.red(cam.lire(c.cap)), cv2.COLOR_RGB2BGR)
            if rot:
                reel = np.ascontiguousarray(np.rot90(reel, rot))
                if reel.shape[:2] != c.ref.shape[:2]:
                    reel = cv2.resize(reel, (c.ref.shape[1], c.ref.shape[0]))
            if n % 5 == 0:
                score = score_alignement(c.ref_cont, reel)
            n += 1
            aff = composer(c.ref, reel, c.ref_cont, MODES[mode])
            texte(aff, f"{c.nom}  mode {MODES[mode]}  score {100 * score:.0f} %   correction lacet {c.corr['lacet']:+.1f}"
                       f"  tangage {c.corr['tangage']:+.1f}  roulis {c.corr['roulis']:+.1f} deg", 18)
            texte(aff, "[m] mode [f]/[w] camera [j][l] [i][k] [u][o] correction [x] zero [e] enregistrer "
                       "[s] capture [q] quitter", 38)
            fen.montrer(aff)
            k = fen.touche()
            if k == "q":
                return
            if k in TOUCHES_CORR:
                c.corriger(*TOUCHES_CORR[k])
                n = 0  # recalculer le score tout de suite
            if k == "x":
                c.corriger(zero=True)
                n = 0
            if k == "e":
                c.enregistrer_correction()
                print(f"correction enregistrée dans {c.path} : {c.corr} (score {100 * score:.0f} %)")
            if k == "m":
                mode = (mode + 1) % len(MODES)
            if k == "r":
                rot = (rot + 1) % 4
            if k in ("f", "w"):
                nouveau = "front" if k == "f" else "wrist"
                if nouveau != c.nom:
                    c.fermer()
                    c = Camera(nouveau, args.pose.upper())
                    rot = 0
            if k == "s":
                f = chemins.ALIGNEMENT / f"alignement_{c.nom}_{datetime.now():%H%M%S}.png"
                cv2.imwrite(str(f), np.hstack([c.ref, reel, composer(c.ref, reel, c.ref_cont, "fondu")]))
                print(f"capture : {f}  (score {100 * score:.0f} %, rotation {90 * rot}°, correction {c.corr})")
    finally:
        c.fermer()
        fen.fermer()


if __name__ == "__main__":
    main()
