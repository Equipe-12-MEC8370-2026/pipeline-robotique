r"""TEST 3 : politique entraînée en sim (SmolVLA P4, via policy_server d'Alexis) sur le vrai SO-101, en boucle fermée.

Avant : lancer le serveur de politique dans un AUTRE terminal (depuis tests\alexis) :
    $env:PYTHONPATH = "$PWD\src"; $env:HF_HUB_OFFLINE = "1"
    .\.venv-learn\Scripts\python.exe scripts\eval\policy_server.py --ckpt outputs\train\smolvla_p4\checkpoints\020000\pretrained_model --n_action_steps 25

Déroulé : vérifications -> « oui » -> couple (consigne = position actuelle) -> approche lente (6 s) de la POSE A
(pose de départ des épisodes sim ; le repos replié est hors des limites sim) -> pause : poser le cube -> « go »
-> boucle à --hz (10 par défaut = 3x plus lent que la sim) jusqu'à --duree s, [ESPACE] ou [q]
-> bras figé -> Entrée : retour lent -> Entrée : couple coupé.
--observation : identique, mais les actions de la politique ne sont JAMAIS envoyées (le bras tient la pose A) :
  on regarde ce que la politique voit et ce qu'elle voudrait faire. Déplacer le cube à la main est permis.
--essai_a_blanc : n'écrit rien sur le bras ; un appel à la politique avec les vraies images et l'état de la pose A.
Garde-fous : RAM seulement ; plafond par pas (3° bras, 10 pince) relatif à la consigne précédente ; tout refus de
la passerelle (état ou action hors limites) ARRÊTE et fige ; écart de suivi > 15° fige ; toute erreur dans la boucle
fige puis retour lent ; perte USB : reconnexion pour le retour.
Politique interrogée dans un fil séparé, --avance actions à l'avance : plus de pause pendant l'inférence.
ARRÊT D'URGENCE = débrancher l'alimentation du bras. Enregistrement : mesures\smolvla_<date>\ (vidéo + données).

Depuis tests\tristan\passerelle, dans TON terminal :
    ..\..\alexis\.venv-learn\Scripts\python.exe smolvla\smolvla_reel.py --essai_a_blanc
    ..\..\alexis\.venv-learn\Scripts\python.exe smolvla\smolvla_reel.py --observation
    ..\..\alexis\.venv-learn\Scripts\python.exe smolvla\smolvla_reel.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chemins  # noqa: E402  (rend les autres dossiers de la passerelle importables)
sys.path.insert(0, str(chemins.ALEXIS / "src"))
import boucle as bl  # noqa: E402
import cameras as cam  # noqa: E402
import mesure  # noqa: E402
import mouvement as mv  # noqa: E402
from affichage import Fenetre  # noqa: E402
from aller_a_pose import (couper_couple_robuste, envoyer, espace_appuye, executer, figer, lire_reel,  # noqa: E402
                          retour_robuste)
from conversion import Correspondance  # noqa: E402
from lire_articulations import DEFAULT_CALIB, make_bus  # noqa: E402

JOINTS = mesure.JOINTS


def texte(img, s, y, couleur=(255, 255, 255)):
    cv2.putText(img, s, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, couleur, 1, cv2.LINE_AA)


def vue(imgs, pas: bl.Pas, k: int, mode: str) -> np.ndarray:
    """Ce que la politique voit (front | wrist) + état et action en degrés sim."""
    aff = cv2.cvtColor(np.hstack([imgs["front"], imgs["wrist"]]), cv2.COLOR_RGB2BGR)
    fmt = lambda v: " ".join(f"{x:6.1f}" for x in np.degrees(v)) if v is not None else "-"  # noqa: E731
    texte(aff, f"{mode}  pas {k}   [ESPACE]/[q] arreter", 16)
    texte(aff, f"etat sim   : {fmt(pas.etat_sim_rad)}", 36)
    texte(aff, f"action sim : {fmt(pas.action_sim_rad)}", 54, (0, 255, 255))
    if pas.plafonnees:
        texte(aff, f"plafonne : {', '.join(pas.plafonnees)}", 72, (0, 165, 255))
    if pas.arret:
        texte(aff, f"ARRET : {pas.arret[:110]}", 92, (0, 0, 255))
    return aff


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--observation", action="store_true", help="la politique tourne, ses actions ne sont pas envoyées")
    p.add_argument("--essai_a_blanc", action="store_true", help="rien n'est écrit sur le bras")
    p.add_argument("--hz", type=float, default=10.0)
    p.add_argument("--duree", type=float, default=60.0, help="durée maximale de la boucle (s)")
    p.add_argument("--pose_depart", default="A")
    p.add_argument("--max_pas", type=float, default=bl.MAX_PAS_BRAS)
    p.add_argument("--max_pince", type=float, default=bl.MAX_PAS_PINCE)
    p.add_argument("--avance", type=int, default=3, help="actions demandées à l'avance à la politique (fil séparé)")
    p.add_argument("--port", default="COM3")
    p.add_argument("--calib", default=str(DEFAULT_CALIB))
    p.add_argument("--couple", type=int, default=500)
    p.add_argument("--couple_pince", type=int, default=300)
    p.add_argument("--serveur", default="127.0.0.1:50765")
    args = p.parse_args()

    from cta_learn.eval.protocol import PolicyClient

    calib = mesure.load_calibration(args.calib)
    corr = Correspondance.charger(chemins.CORRESPONDANCE, calib)
    cal_cam = {n: cam.charger_calibration(chemins.VISION / f"calib_{n}.json") for n in ("front", "wrist")}
    poses = json.loads(chemins.POSES.read_text(encoding="utf-8"))["poses"]
    nom_depart = next(n for n in poses if n.startswith(args.pose_depart.upper() + "_"))
    depart_sim = poses[nom_depart]["target_deg"]
    cible_depart = mv.cible_reelle(corr, depart_sim)

    hote, port_srv = args.serveur.split(":")
    try:
        client = PolicyClient((hote, int(port_srv)))
    except OSError as e:
        raise SystemExit(f"Serveur de politique injoignable sur {args.serveur} ({e}) : lance policy_server.py "
                         "dans un autre terminal (voir l'aide : -h).")
    info = client.info()
    print(f"Politique : {info.get('policy')}  ckpt {Path(info.get('ckpt', '?')).parts[-4:]}  "
          f"n_action_steps {info.get('n_action_steps')}  tâche « {info.get('task')} »")

    flux = {n: cam.FluxCamera(int(c["index_opencv"]), *c["taille"]) for n, c in cal_cam.items()}
    red = {n: cam.Redresseur(c) for n, c in cal_cam.items()}
    b = bl.Boucle(corr, red["front"], red["wrist"], client, observation=args.observation or args.essai_a_blanc,
                  max_bras=args.max_pas, max_pince=args.max_pince)
    fen = Fenetre("ce que voit la politique (front | wrist)")
    mode = "ESSAI A BLANC" if args.essai_a_blanc else ("OBSERVATION" if args.observation else "REEL")

    if args.essai_a_blanc:
        try:
            client.reset()
            t = time.perf_counter()
            pas, imgs = b.pas(cible_depart, flux["front"].derniere()[0], flux["wrist"].derniere()[0])
            dt = time.perf_counter() - t
            print(f"1 pas (état simulé = pose {args.pose_depart}) en {1000 * dt:.0f} ms ; arrêt : {pas.arret}")
            print("  état sim  :", np.round(np.degrees(pas.etat_sim_rad), 1).tolist())
            if pas.action_sim_rad is not None:
                print("  action sim:", np.round(np.degrees(pas.action_sim_rad), 1).tolist())
            fen.montrer(vue(imgs, pas, 0, mode))
            input("ESSAI À BLANC : rien n'a été écrit sur le bras. Entrée pour fermer... ")
        finally:
            for f in flux.values():
                f.fermer()
            fen.fermer()
            client.close()
        return

    bus = make_bus(args.port, calib)
    bus.connect(handshake=True)
    couple_actif = False
    journal: list[dict] = []
    dossier = chemins.MESURES / f"smolvla_{datetime.now():%Y%m%d_%H%M%S}_{mode.lower()}"
    video = None
    donnees = {k: [] for k in ("t", "reel", "etat_sim", "action_sim", "cible", "consigne", "plafonne")}
    try:
        mc = {j: {"homing_offset": c.homing_offset, "range_min": c.range_min, "range_max": c.range_max}
              for j, c in bus.read_calibration().items()}
        if mesure.calibration_mismatches(calib, mc):
            raise SystemExit("ABANDON : calibration des moteurs != fichier")
        if any(bus.sync_read("Operating_Mode", normalize=False).values()):
            raise SystemExit("ABANDON : moteurs pas tous en mode position")
        if any(bus.sync_read("Torque_Enable", normalize=False).values()):
            raise SystemExit("ABANDON : un moteur a déjà le couple actif (lancer couper_couple.py)")
        raw0, depart = lire_reel(bus, calib)
        approche = mv.planifier(depart, cible_depart, 6.0, 30.0)
        mv.verifier_dans_plage_reelle(corr, approche)
        print(f"\nMode {mode} ; approche de la pose {nom_depart} en 6 s (déplacement max "
              f"{np.abs(cible_depart - depart).max():.1f}°) ; boucle {args.hz:.0f} Hz, {args.duree:.0f} s max ; "
              f"plafond {args.max_pas}°/pas (pince {args.max_pince})")
        print("AVANT DE CONTINUER : zone dégagée, main sur l'alimentation (débrancher = arrêt), personne dans "
              "le champ de la caméra avant, pas d'objet sur la table hormis le cube et le carré blanc.")
        if input("Taper « oui » puis Entrée pour activer le couple et aller en pose de départ : ").strip().lower() != "oui":
            print("Annulé : rien n'a été écrit.")
            return

        bus.sync_write("Goal_Position", raw0, normalize=False)
        bus.sync_write("Torque_Limit", {j: (args.couple_pince if j == "gripper" else args.couple) for j in JOINTS},
                       normalize=False)
        bus.sync_write("Acceleration", {j: 30 for j in JOINTS}, normalize=False)
        bus.sync_write("Goal_Velocity", {j: 600 for j in JOINTS}, normalize=False)
        couple_actif = True
        bus.enable_torque(num_retry=3)
        time.sleep(0.5)
        saut, js = mv.ecart_suivi(depart, lire_reel(bus, calib)[1])
        print(f"Couple activé. Variation à l'activation : {saut:.1f}° ({js})")
        if saut > 3.0:
            figer(bus, calib)
            raise RuntimeError("le bras a bougé à l'activation du couple : arrêt")
        if not executer(bus, calib, approche, 30.0, journal, "approche"):
            raise RuntimeError("approche interrompue")
        # chauffe du GPU : le 1er appel à la politique prend ~16 s (compilation) ; son action est ignorée
        print("Chauffe de la politique (~15 s, le bras tient la pose)...")
        b.pas(lire_reel(bus, calib)[1], flux["front"].derniere()[0], flux["wrist"].derniere()[0])
        client.reset()

        rep = input(f"\nBras en pose de départ. Pose le cube, puis « go » + Entrée pour lancer la politique "
                    f"({'ses actions NE seront PAS envoyées' if args.observation else 'le bras VA BOUGER'}) : ")
        if rep.strip().lower() == "go":
            dossier.mkdir(parents=True, exist_ok=True)
            video = cv2.VideoWriter(str(dossier / "vue_politique.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                    args.hz, (1280, 480))
            client.reset()
            # la boucle publie la position mesurée ; le fil de la politique la lit (il ne touche jamais au bus)
            publie = {"reel": lire_reel(bus, calib)[1]}

            def source():
                return publie["reel"], flux["front"].derniere()[0], flux["wrist"].derniere()[0]

            politique = bl.PolitiqueAsynchrone(b, source, avance=args.avance)
            periode, t0, derniere_consigne, k, famine = 1.0 / args.hz, time.perf_counter(), None, 0, 0
            raison = "durée maximale atteinte"
            try:
                while time.perf_counter() - t0 < args.duree:
                    tk = time.perf_counter()
                    _, reel = lire_reel(bus, calib)
                    publie["reel"] = reel
                    if derniere_consigne is not None:
                        e, j = mv.ecart_suivi(derniere_consigne, reel)
                        if e > mv.ECART_SUIVI_MAX:
                            raison = f"écart de suivi {e:.1f}° sur {j}"
                            break
                    item = politique.suivant()
                    if item is None:  # politique en retard : on garde la consigne courante
                        famine += 1
                    else:
                        pas, imgs = item
                        if pas.arret:
                            fen.montrer(vue(imgs, pas, k, mode))
                            raison = pas.arret
                            break
                        pas = b.commander(pas, derniere_consigne if derniere_consigne is not None else reel)
                        if pas.consigne is not None:
                            envoyer(bus, pas.consigne)
                            derniere_consigne = pas.consigne
                        aff = vue(imgs, pas, k, mode)
                        video.write(aff)
                        fen.montrer(aff)
                        for key, v in (("t", tk - t0), ("reel", reel), ("etat_sim", pas.etat_sim_rad),
                                       ("action_sim", pas.action_sim_rad), ("cible", pas.cible_reelle),
                                       ("consigne", pas.consigne), ("plafonne", len(pas.plafonnees))):
                            donnees[key].append(np.full(6, np.nan) if v is None and key not in ("t", "plafonne")
                                                else v)
                        k += 1
                    touche = fen.touche()
                    if touche in (" ", "q") or espace_appuye():
                        raison = "arrêt demandé"
                        break
                    age = max(flux["front"].derniere()[1], flux["wrist"].derniere()[1])
                    if age > 0.5:
                        raison = f"image trop ancienne ({age:.1f} s) : caméra bloquée ?"
                        break
                    time.sleep(max(0.0, periode - (time.perf_counter() - tk)))
            except Exception as e:  # noqa: BLE001  toute erreur dans la boucle : on fige, puis retour lent
                raison = f"erreur dans la boucle : {type(e).__name__}: {e}"
            finally:
                politique.arreter()
            figer(bus, calib)
            try:  # statistiques : une erreur ici ne doit JAMAIS empêcher le retour lent qui suit
                n_plaf = int(np.sum(np.asarray(donnees["plafonne"], dtype=float) > 0))
                print(f"\nBoucle terminée après {k} pas ({time.perf_counter() - t0:.0f} s) : {raison}. "
                      f"Pas plafonnés : {n_plaf}/{max(k, 1)} ; pas sans action prête : {famine}. Bras figé.")
                journal.append({"phase": "boucle", "mode": mode, "pas": k, "raison": raison, "plafonnes": n_plaf,
                                "famine": famine, "politique": info})
            except Exception as e:  # noqa: BLE001
                print(f"\nBoucle terminée ({raison}) ; statistiques impossibles : {e}. Bras figé.")

        input("\nEntrée = retour lent à la position de départ (repliée)... ")
        bus = retour_robuste(bus, args.port, calib, depart, 6.0, 30.0, journal)
        time.sleep(0.5)
    except KeyboardInterrupt:
        if couple_actif:
            figer(bus, calib)
        print("\nInterrompu : bras figé.")
    finally:
        if video is not None:
            video.release()
        if donnees["t"]:
            np.savez(dossier / "donnees.npz", **{k: np.array(v, dtype=float) for k, v in donnees.items()})
        if journal:
            dossier.mkdir(parents=True, exist_ok=True)
            (dossier / "journal.json").write_text(json.dumps(journal, default=str), encoding="utf-8")
            print(f"Enregistrement : {dossier}")
        if couple_actif:
            input("TIENS LE BRAS si besoin, puis Entrée pour couper le couple... ")
            bus = couper_couple_robuste(bus, args.port, calib)
        try:
            bus.disconnect(disable_torque=False)
        except OSError:
            pass
        for f in flux.values():
            f.fermer()
        fen.fermer()
        client.close()


if __name__ == "__main__":
    main()
