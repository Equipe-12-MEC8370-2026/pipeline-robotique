r"""TEST 2 : rejoue lentement une démo de l'expert sim (P4) sur le vrai SO-101, via la passerelle (boucle ouverte).

Déroulé : vérifications -> plan + placement du cube/cible affichés -> « oui » -> couple (consigne = position actuelle)
-> approche lente (6 s) de la 1re pose de la démo -> pause (poser le cube) -> démo ralentie x--facteur
-> maintien -> retour lent à la position de départ -> couple coupé sur confirmation.
Garde-fous de aller_a_pose.py : RAM seulement (couple, accélération, vitesse), <= 2°/pas, chaque action passe par
sim_vers_reel (refus hors limites), [ESPACE] fige le bras, écart de suivi > 15° (hors pince) fige le bras.
Pince : couple limité à part (--couple_pince) pour ne pas écraser le cube. ARRÊT D'URGENCE = débrancher l'alimentation.

Depuis tests\tristan\passerelle, dans TON terminal :
    ..\..\alexis\.venv-learn\Scripts\python.exe rejeu\rejouer_demo.py --essai_a_blanc
    ..\..\alexis\.venv-learn\Scripts\python.exe rejeu\rejouer_demo.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chemins  # noqa: E402  (rend les autres dossiers de la passerelle importables)
import mesure  # noqa: E402
import mouvement as mv  # noqa: E402
from aller_a_pose import couper_couple_robuste, executer, figer, lire_reel, retour_robuste  # noqa: E402
from conversion import Correspondance  # noqa: E402
from lire_articulations import DEFAULT_CALIB, make_bus  # noqa: E402

JOINTS = mesure.JOINTS
RAW_ROOT = chemins.DEMOS_P4
PAN_AXIS_ENV = np.array([0.0231, 0.0208])  # axe de shoulder_pan dans le repère de l'env (m), mesuré en sim
CUBE = 0.03  # m


def yaw_deg(q_xyzw) -> float:
    x, y, z, w = (float(v) for v in q_xyzw)
    return (math.degrees(2 * math.atan2(z, w)) + 45) % 90 - 45  # cube : symétrie 90°


def placement(d) -> dict:
    """Cube et cible au début de la démo, par rapport à l'axe de la base : avant (+) / gauche (+), en cm."""
    c = (d["cube_pos"][0, :2] - PAN_AXIS_ENV) * 100
    t = (d["target_pos"][0, :2] - PAN_AXIS_ENV) * 100
    return {"cube_avant_cm": round(float(c[0]), 1), "cube_gauche_cm": round(float(c[1]), 1),
            "cube_rotation_deg": round(yaw_deg(d["cube_quat"][0]), 1),
            "cible_avant_cm": round(float(t[0]), 1), "cible_gauche_cm": round(float(t[1]), 1)}


def cote(v: float) -> str:
    return f"{abs(v):.1f} cm à {'GAUCHE' if v > 0 else 'DROITE'}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episode", default="episode_000096")
    p.add_argument("--raw", default=str(RAW_ROOT))
    p.add_argument("--facteur", type=int, default=3, help="ralentissement (3 = 3x plus lent)")
    p.add_argument("--port", default="COM3")
    p.add_argument("--calib", default=str(DEFAULT_CALIB))
    p.add_argument("--hz", type=float, default=30.0)
    p.add_argument("--couple", type=int, default=500, help="Torque_Limit bras (RAM), 500 = 50 %%")
    p.add_argument("--couple_pince", type=int, default=300, help="Torque_Limit pince (RAM), 300 = 30 %%")
    p.add_argument("--essai_a_blanc", action="store_true", help="vérifier et montrer le plan, sans rien écrire")
    args = p.parse_args()

    calib = mesure.load_calibration(args.calib)
    corr = Correspondance.charger(chemins.CORRESPONDANCE, calib)
    ep = Path(args.raw) / args.episode
    d = np.load(ep / "data.npz")
    meta = json.loads((ep / "meta.json").read_text(encoding="utf-8"))
    demo = mv.ralentir(mv.demo_en_reel(corr, d["action"]), args.facteur)  # refus si une action sort des limites
    mv.verifier_pas(demo)
    mv.verifier_dans_plage_reelle(corr, demo)
    pl = placement(d)

    print(f"\nDémo {args.episode} : {meta['num_frames']} actions, succès en sim : {meta['success']}, "
          f"reprises : {meta.get('regrasps', 0)}")
    print(f"Rejouée x{args.facteur} plus lent : {len(demo)} pas à {args.hz:.0f} Hz = {len(demo) / args.hz:.0f} s, "
          f"pas max {np.abs(np.diff(demo, axis=0)).max():.2f}")
    print("\nPLACEMENT (mesuré depuis l'AXE VERTICAL de la base, « avant » = direction du bras quand pan = 0,")
    print("gauche/droite vus DERRIÈRE le robot, en regardant dans la même direction que lui) :")
    print(f"  centre du CUBE (3 cm) : {pl['cube_avant_cm']:.1f} cm devant, {cote(pl['cube_gauche_cm'])}, "
          f"faces parallèles aux axes (rotation {pl['cube_rotation_deg']:+.1f}°)")
    print(f"  centre de la CIBLE (carré 6 cm) : {pl['cible_avant_cm']:.1f} cm devant, {cote(pl['cible_gauche_cm'])}")

    bus = make_bus(args.port, calib)
    bus.connect(handshake=True)
    couple_actif = False
    journal: list[dict] = [{"phase": "info", "episode": args.episode, "facteur": args.facteur, "placement": pl}]
    try:
        mc = {j: {"homing_offset": c.homing_offset, "range_min": c.range_min, "range_max": c.range_max}
              for j, c in bus.read_calibration().items()}
        if mesure.calibration_mismatches(calib, mc):
            raise SystemExit("ABANDON : calibration des moteurs != fichier")
        if any(bus.sync_read("Operating_Mode", normalize=False).values()):
            raise SystemExit("ABANDON : moteurs pas tous en mode position")
        if any(bus.sync_read("Torque_Enable", normalize=False).values()):
            raise SystemExit("ABANDON : un moteur a déjà le couple actif (état inattendu)")
        raw0, depart = lire_reel(bus, calib)
        approche = mv.planifier(depart, demo[0], 6.0, args.hz)
        mv.verifier_dans_plage_reelle(corr, approche)
        print(f"\nApproche de la 1re pose de la démo : {len(approche) / args.hz:.1f} s, "
              f"déplacement max {np.abs(demo[0] - depart).max():.1f}°")
        if args.essai_a_blanc:
            print("\nESSAI À BLANC : toute la démo passe par la passerelle ; rien n'a été écrit sur les moteurs.")
            return

        print("\nAVANT DE CONTINUER : zone dégagée, PAS ENCORE DE CUBE, main sur l'alimentation (débrancher = arrêt).")
        print("Pendant les mouvements : [ESPACE] = figer le bras sur place.")
        if input("Taper « oui » puis Entrée pour activer le couple et aller à la 1re pose : ").strip().lower() != "oui":
            print("Annulé : rien n'a été écrit.")
            return

        bus.sync_write("Goal_Position", raw0, normalize=False)
        bus.sync_write("Torque_Limit", {j: (args.couple_pince if j == "gripper" else args.couple) for j in JOINTS},
                       normalize=False)
        bus.sync_write("Acceleration", {j: 30 for j in JOINTS}, normalize=False)
        bus.sync_write("Goal_Velocity", {j: 600 for j in JOINTS}, normalize=False)
        couple_actif = True  # AVANT l'activation : si elle échoue à mi-chemin, le finally coupe quand même le couple
        bus.enable_torque(num_retry=3)
        time.sleep(0.5)
        _, apres = lire_reel(bus, calib)
        saut, js = mv.ecart_suivi(depart, apres)
        print(f"Couple activé. Variation à l'activation : {saut:.1f}° ({js})")
        if saut > 3.0:
            figer(bus, calib)
            raise RuntimeError("le bras a bougé à l'activation du couple : arrêt")

        if executer(bus, calib, approche, args.hz, journal, "approche"):
            print("\nBras à la 1re pose de la démo (au-dessus de la table).")
            rep = input("Pose le cube (ou pas, pour un essai à vide), puis « go » + Entrée pour lancer la démo "
                        "(autre chose = ne pas la lancer) : ").strip().lower()
            if rep == "go":
                ok = executer(bus, calib, demo, args.hz, journal, "demo")
                time.sleep(1.0)
                _, fin = lire_reel(bus, calib)
                journal.append({"phase": "fin_demo", "ok": ok, "mesure": fin.round(2).tolist(),
                                "charge": bus.sync_read("Present_Load", normalize=False),
                                "temperature": bus.sync_read("Present_Temperature", normalize=False)})
                print(f"Démo {'terminée' if ok else 'INTERROMPUE'}. Températures : {journal[-1]['temperature']}")

        input("\nEntrée = retour lent à la position de départ (repliée)... ")
        bus = retour_robuste(bus, args.port, calib, depart, 6.0, args.hz, journal)
        time.sleep(0.5)
    except KeyboardInterrupt:
        if couple_actif:
            figer(bus, calib)
        print("\nInterrompu : bras figé.")
    finally:
        if len(journal) > 1:
            out = chemins.MESURES / f"rejeu_{args.episode}_{datetime.now():%Y%m%d_%H%M%S}.json"
            out.write_text(json.dumps(journal), encoding="utf-8")
            print(f"Journal : {out}")
        if couple_actif:
            input("TIENS LE BRAS si besoin, puis Entrée pour couper le couple... ")
            bus = couper_couple_robuste(bus, args.port, calib)
        try:
            bus.disconnect(disable_torque=False)
        except OSError:
            pass


if __name__ == "__main__":
    main()
