r"""Calibration fisheye d'UNE caméra réelle avec le damier imprimé (cameras\damier_9x7_25mm_lettre.pdf).

Fenêtre en direct : coins détectés en couleur ; points déjà capturés en vert (il faut couvrir TOUTE l'image,
surtout les bords et les coins, où la distorsion fisheye est la plus forte).
Touches : [ESPACE] capturer (si le damier est détecté)   [c] calibrer (>= 12 vues)   [q] quitter
Après calibration : aperçu côte à côte (brut | redressé comme la sim), puis [s] sauver, [q] quitter sans sauver.
Ne touche pas au robot.

Depuis tests\tristan\passerelle, dans TON terminal :
    ..\..\alexis\.venv-learn\Scripts\python.exe vision\calibrer_camera.py --index 0 --nom wrist
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chemins  # noqa: E402  (rend les autres dossiers de la passerelle importables)
import cameras as cam  # noqa: E402
from affichage import Fenetre  # noqa: E402


def texte(img, s, y, couleur=(255, 255, 255)):
    cv2.putText(img, s, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, couleur, 1, cv2.LINE_AA)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index", type=int, required=True, help="index OpenCV de la caméra (0, 1…)")
    p.add_argument("--nom", required=True, choices=["front", "wrist"])
    p.add_argument("--largeur", type=int, default=640)
    p.add_argument("--hauteur", type=int, default=480)
    args = p.parse_args()

    out_dir = chemins.VISION
    vues_dir = out_dir / f"vues_{args.nom}"
    vues_dir.mkdir(parents=True, exist_ok=True)
    cap = cam.ouvrir(args.index, args.largeur, args.hauteur)
    print(f"Caméra {args.index} ({args.nom}) : {args.largeur}x{args.hauteur}, {cam.mesurer_cadence(cap, 1.0):.0f} img/s")
    vues: list[np.ndarray] = []
    couverture = np.zeros((args.hauteur, args.largeur, 3), np.uint8)
    calib = None
    fen = Fenetre(f"calibration {args.nom} (camera {args.index})")
    try:
        while True:
            bgr = cam.lire(cap)
            if calib is None:
                coins = cam.detecter_damier(bgr, precis=False)  # aperçu rapide
                aff = cv2.addWeighted(bgr, 1.0, couverture, 0.8, 0)
                if coins is not None:
                    cv2.drawChessboardCorners(aff, cam.DAMIER_COINS, coins.reshape(-1, 1, 2).astype(np.float32), True)
                texte(aff, f"{args.nom} : {len(vues)} vues   [ESPACE] capturer  [c] calibrer (>=12)  [q] quitter", 20)
                texte(aff, "damier detecte" if coins is not None else "damier NON detecte", 42,
                      (0, 255, 0) if coins is not None else (0, 0, 255))
                fen.montrer(aff)
                k = fen.touche()
                if k == "q":
                    return
                if k == " " and coins is not None:
                    coins = cam.detecter_damier(bgr, precis=True)  # détection précise pour la vue gardée
                    if coins is None:
                        print("damier perdu à la capture : réessaie en tenant la caméra immobile")
                        continue
                    vues.append(coins)
                    cv2.imwrite(str(vues_dir / f"vue_{len(vues):02d}.png"), bgr)
                    for x, y in coins[0]:
                        cv2.circle(couverture, (int(x), int(y)), 3, (0, 255, 0), -1)
                    print(f"vue {len(vues)} capturée")
                if k == "c":
                    if len(vues) < 12:
                        print(f"seulement {len(vues)} vues : il en faut au moins 12")
                        continue
                    calib = cam.calibrer(vues, (args.largeur, args.hauteur))
                    red = cam.Redresseur(calib)
                    K = np.array(calib["K"])
                    print(f"Calibration : erreur RMS {calib['rms_px']:.3f} px sur {len(vues)} vues ; "
                          f"fx {K[0, 0]:.1f}  fy {K[1, 1]:.1f}  cx {K[0, 2]:.1f}  cy {K[1, 2]:.1f}  D {np.round(calib['D'], 4)}")
                    if calib["rms_px"] > 1.0:
                        print("ATTENTION : erreur > 1 px. Recommence avec des vues plus nettes et mieux réparties.")
            else:
                droite = cv2.cvtColor(red(bgr), cv2.COLOR_RGB2BGR)
                aff = np.hstack([bgr, droite])
                texte(aff, f"brut (fisheye)  |  redresse comme la sim (90 deg)   RMS {calib['rms_px']:.2f} px   "
                           "[s] sauver  [q] quitter", 20)
                fen.montrer(aff)
                k = fen.touche()
                if k == "s":
                    path = out_dir / f"calib_{args.nom}.json"
                    cam.sauver_calibration(calib, path, args.nom, args.index)
                    print(f"Sauvé : {path}")
                    return
                if k == "q":
                    print("Quitté sans sauver.")
                    return
    finally:
        cap.release()
        fen.fermer()


if __name__ == "__main__":
    main()
