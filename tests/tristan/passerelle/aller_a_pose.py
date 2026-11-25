r"""PREMIER MOUVEMENT COMMANDÉ : amène lentement le vrai SO-101 dans une pose de la sim, via la passerelle.

Sécurité :
  * n'utilise pas robot.connect() de LeRobot ; avant d'activer le couple, la consigne = position actuelle
    (le bras ne saute pas) ; abandon si un moteur a déjà le couple ou n'est pas en mode position ;
  * n'écrit QUE dans la RAM des moteurs (Torque_Limit, Acceleration, Goal_Velocity) : tout revient à l'origine
    à la coupure d'alimentation ; rien dans l'EEPROM ;
  * cible = pose sim convertie par la passerelle (refus hors limites sim / plage réelle) ; trajectoire en ligne droite,
    profil cosinus, >= --duree s, <= 2° par pas à 30 Hz ; chaque point vérifié dans la plage réelle ;
  * [ESPACE] pendant le mouvement : le bras se FIGE sur place (consigne = position mesurée, couple maintenu) ;
    écart consigne/mesure > 15° (blocage, collision) : idem ;
  * couple coupé seulement à la fin, après retour en position de départ (repliée), sur confirmation ;
  * ARRÊT D'URGENCE = débrancher l'alimentation du bras.
--essai_a_blanc : lit la position, montre le plan, n'écrit RIEN.

Depuis tests\tristan\passerelle, dans TON terminal :
    ..\..\alexis\.venv-learn\Scripts\python.exe aller_a_pose.py --pose D --essai_a_blanc
    ..\..\alexis\.venv-learn\Scripts\python.exe aller_a_pose.py --pose D
"""

from __future__ import annotations

import argparse
import json
import math
import msvcrt
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mesure  # noqa: E402
import mouvement as mv  # noqa: E402
from conversion import Correspondance, HorsLimites  # noqa: E402
from lire_articulations import DEFAULT_CALIB, make_bus  # noqa: E402

JOINTS = mesure.JOINTS


def lire_reel(bus, calib) -> tuple[dict[str, int], np.ndarray]:
    raw = {j: int(v) for j, v in bus.sync_read("Present_Position", normalize=False, num_retry=3).items()}
    return raw, mv.vecteur(mesure.raw_to_lerobot(raw, calib))


def sim_txt(corr, reel) -> str:
    try:
        return " ".join(f"{math.degrees(v):7.1f}" for v in corr.reel_vers_sim(mv.dictionnaire(reel)))
    except HorsLimites:
        return "(hors limites sim)"


def envoyer(bus, consigne) -> None:
    bus.sync_write("Goal_Position", {j: float(v) for j, v in zip(JOINTS, consigne)})


def figer(bus, calib) -> np.ndarray:
    """Consigne = position mesurée : le bras s'arrête et reste en place (couple maintenu)."""
    raw, reel = lire_reel(bus, calib)
    bus.sync_write("Goal_Position", raw, normalize=False)
    return reel


def espace_appuye() -> bool:
    hit = False
    while msvcrt.kbhit():
        hit |= msvcrt.getwch() == " "
    return hit


def executer(bus, calib, traj, hz, journal, etiquette) -> bool:
    """Joue la trajectoire ; renvoie False si le bras a été figé (espace ou écart de suivi)."""
    periode = 1.0 / hz
    t0 = time.perf_counter()
    for k, consigne in enumerate(traj):
        if espace_appuye():
            reel = figer(bus, calib)
            print(f"\n[ESPACE] bras figé au pas {k}/{len(traj)} : {np.round(reel, 1).tolist()}")
            return False
        envoyer(bus, consigne)
        _, reel = lire_reel(bus, calib)
        e, j = mv.ecart_suivi(consigne, reel)
        journal.append({"phase": etiquette, "k": k, "consigne": consigne.round(2).tolist(),
                        "mesure": reel.round(2).tolist(), "ecart_max": round(e, 2)})
        if e > mv.ECART_SUIVI_MAX and k > 5:
            reel = figer(bus, calib)
            print(f"\nÉCART DE SUIVI {e:.1f}° sur {j} (> {mv.ECART_SUIVI_MAX}) : bras figé. Mesure {np.round(reel, 1)}")
            return False
        if k % int(hz) == 0:
            print(f"  {etiquette} {100 * (k + 1) / len(traj):5.1f} %  écart suivi max {e:4.1f}° ({j})", end="\r")
        time.sleep(max(0.0, t0 + (k + 1) * periode - time.perf_counter()))
    print()
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pose", default="D", help="lettre de la pose dans poses/poses.json (D, E, B…)")
    p.add_argument("--port", default="COM3")
    p.add_argument("--calib", default=str(DEFAULT_CALIB))
    p.add_argument("--duree", type=float, default=6.0, help="durée minimale du mouvement (s)")
    p.add_argument("--hz", type=float, default=30.0)
    p.add_argument("--couple", type=int, default=500, help="Torque_Limit (RAM), 0-1000 ; 500 = 50 %%")
    p.add_argument("--essai_a_blanc", action="store_true", help="lire et montrer le plan, sans rien écrire")
    args = p.parse_args()

    calib = mesure.load_calibration(args.calib)
    corr = Correspondance.charger(HERE / "correspondance_greg.json", calib)
    poses = json.loads((HERE / "poses" / "poses.json").read_text(encoding="utf-8"))["poses"]
    nom = next(n for n in poses if n.startswith(args.pose.upper() + "_"))
    cible_sim = poses[nom]["target_deg"]
    cible = mv.cible_reelle(corr, cible_sim)  # passe par la passerelle et ses refus

    bus = make_bus(args.port, calib)
    bus.connect(handshake=True)
    couple_actif = False
    journal: list[dict] = []
    try:
        mc = {j: {"homing_offset": c.homing_offset, "range_min": c.range_min, "range_max": c.range_max}
              for j, c in bus.read_calibration().items()}
        if mesure.calibration_mismatches(calib, mc):
            raise SystemExit("ABANDON : calibration des moteurs != fichier")
        modes = bus.sync_read("Operating_Mode", normalize=False)
        if any(modes.values()):
            raise SystemExit(f"ABANDON : moteurs pas tous en mode position : {modes}")
        if any(bus.sync_read("Torque_Enable", normalize=False).values()):
            raise SystemExit("ABANDON : un moteur a déjà le couple actif (état inattendu)")

        raw0, depart = lire_reel(bus, calib)
        traj = mv.planifier(depart, cible, args.duree, args.hz)
        mv.verifier_dans_plage_reelle(corr, traj)
        retour = mv.planifier(cible, depart, args.duree, args.hz)

        print(f"\nPose cible : {nom}")
        print(f"{'':16s}" + "".join(f"{j[:11]:>12s}" for j in JOINTS))
        print(f"{'départ réel':16s}" + "".join(f"{v:12.1f}" for v in depart))
        print(f"{'cible réelle':16s}" + "".join(f"{v:12.1f}" for v in cible))
        print(f"{'déplacement':16s}" + "".join(f"{v:12.1f}" for v in cible - depart))
        print(f"{'cible sim (°)':16s}" + "".join(f"{v:12.1f}" for v in cible_sim))
        print(f"départ en sim : {sim_txt(corr, depart)}")
        print(f"\n{len(traj)} pas à {args.hz:.0f} Hz = {len(traj) / args.hz:.1f} s, pas max "
              f"{np.abs(np.diff(np.vstack([depart, traj]), axis=0)).max():.2f}°, couple limité à {args.couple / 10:.0f} %")
        if args.essai_a_blanc:
            print("\nESSAI À BLANC : rien n'a été écrit sur les moteurs.")
            return

        print("\nAVANT DE CONTINUER : zone dégagée, pince vide, main sur l'alimentation (débrancher = arrêt d'urgence).")
        print("Pendant le mouvement : [ESPACE] = figer le bras sur place.")
        if input("Taper « oui » puis Entrée pour activer le couple et bouger : ").strip().lower() != "oui":
            print("Annulé : rien n'a été écrit.")
            return

        # consigne = position actuelle AVANT le couple ; bridage en RAM
        bus.sync_write("Goal_Position", raw0, normalize=False)
        bus.sync_write("Torque_Limit", {j: args.couple for j in JOINTS}, normalize=False)
        bus.sync_write("Acceleration", {j: 30 for j in JOINTS}, normalize=False)
        bus.sync_write("Goal_Velocity", {j: 600 for j in JOINTS}, normalize=False)  # ~53°/s max
        bus.enable_torque()
        couple_actif = True
        time.sleep(0.5)
        _, apres = lire_reel(bus, calib)
        saut, js = mv.ecart_suivi(depart, apres)
        print(f"Couple activé. Variation à l'activation : {saut:.1f}° ({js})")
        if saut > 3.0:
            figer(bus, calib)
            raise RuntimeError("le bras a bougé à l'activation du couple : arrêt")

        ok = executer(bus, calib, traj, args.hz, journal, "aller")
        time.sleep(1.0)
        _, fin = lire_reel(bus, calib)
        print(f"\n{'':16s}" + "".join(f"{j[:11]:>12s}" for j in JOINTS))
        print(f"{'cible sim (°)':16s}" + "".join(f"{v:12.1f}" for v in cible_sim))
        print(f"{'mesure → sim':16s}" + "".join(f"{v:>12s}" for v in sim_txt(corr, fin).split()))
        print(f"{'cible réelle':16s}" + "".join(f"{v:12.1f}" for v in cible))
        print(f"{'mesure réelle':16s}" + "".join(f"{v:12.1f}" for v in fin))
        charge = bus.sync_read("Present_Load", normalize=False)
        temp = bus.sync_read("Present_Temperature", normalize=False)
        print(f"charge {charge}\ntempérature (°C) {temp}")
        journal.append({"phase": "fin_aller", "ok": ok, "mesure": fin.round(2).tolist(), "charge": charge,
                        "temperature": temp})

        print("\nLe bras tient la pose : vérifie au fil à plomb.")
        input("Entrée = retour lent à la position de départ (repliée)... ")
        _, ici = lire_reel(bus, calib)
        retour = mv.planifier(ici, depart, args.duree, args.hz)
        executer(bus, calib, retour, args.hz, journal, "retour")
        time.sleep(0.5)
    except KeyboardInterrupt:
        if couple_actif:
            figer(bus, calib)
        print("\nInterrompu : bras figé.")
    finally:
        out = HERE / "mesures" / f"mouvement_{datetime.now():%Y%m%d_%H%M%S}.json"
        if journal:
            out.write_text(json.dumps(journal), encoding="utf-8")
            print(f"Journal : {out}")
        if couple_actif:
            input("TIENS LE BRAS si besoin, puis Entrée pour couper le couple... ")
            bus.disable_torque()
            print("Couple coupé.")
        bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()
