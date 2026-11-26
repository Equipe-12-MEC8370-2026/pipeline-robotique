r"""Test 1 de la passerelle : LECTURE SEULE des articulations du vrai SO-101, couple coupé, bras placé à la main.

Ne commande AUCUN mouvement. Seule écriture possible : couper le couple (Torque_Enable = 0) si un moteur l'a actif,
après confirmation (le bras devient alors mou : le tenir). N'utilise PAS robot.connect() de LeRobot, qui réactive
le couple à la fin de sa configuration.

Affiche en direct, pour chaque articulation : valeur brute, valeur LeRobot (degrés / pince 0-100), et l'angle sim
prédit par les deux hypothèses (« leisaac » = cta_learn.core.units, « degres » = degrés LeRobot tels quels).
Touches : [a] / [b] / [c]… enregistrer la pose correspondante, [q] quitter. Images des poses : poses\*.png
Les poses déjà enregistrées dans mesures\poses_reelles.json sont conservées d'un lancement à l'autre.

Lancer dans TON terminal (le script attend des touches), depuis tests\tristan\passerelle :
    ..\..\alexis\.venv-learn\Scripts\python.exe correspondance\lire_articulations.py --port COM3
"""

from __future__ import annotations

import argparse
import json
import msvcrt
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chemins  # noqa: E402  (rend les autres dossiers de la passerelle importables)
import math  # noqa: E402

import mesure  # noqa: E402
from conversion import Correspondance, HorsLimites  # noqa: E402

DEFAULT_CALIB = Path(os.environ["USERPROFILE"]) / ".cache/huggingface/lerobot/calibration/robots/so_follower/Greg.json"
TARGETS = {k: dict(zip(mesure.JOINTS, v["target_deg"]))
           for k, v in json.loads(chemins.POSES.read_text(encoding="utf-8"))["poses"].items()}
KEYS = {k[0].lower(): k for k in TARGETS}  # a, b, c… = première lettre du nom de la pose


def make_bus(port: str, calib: dict[str, mesure.Calib]):
    from lerobot.motors import Motor, MotorCalibration, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    motors = {j: Motor(calib[j].id, "sts3215", MotorNormMode.DEGREES) for j in mesure.BODY}
    motors["gripper"] = Motor(calib["gripper"].id, "sts3215", MotorNormMode.RANGE_0_100)
    cal = {j: MotorCalibration(c.id, c.drive_mode, c.homing_offset, c.range_min, c.range_max) for j, c in calib.items()}
    return FeetechMotorsBus(port=port, motors=motors, calibration=cal)


def passerelle(raw, calib, corr) -> dict[str, str]:
    """Angles sim (degrés) donnés par le module de conversion, ou la raison du refus."""
    if corr is None:
        return {j: "-" for j in mesure.JOINTS}
    try:
        q = corr.reel_vers_sim({f"{j}.pos": v for j, v in mesure.raw_to_lerobot(raw, calib).items()})
        return {j: f"{math.degrees(v):.1f}°" for j, v in zip(mesure.JOINTS, q)}
    except HorsLimites as e:
        return {j: ("HORS" if j in str(e) else "?") for j in mesure.JOINTS}


def table(raw, calib, corr=None) -> str:
    lr = mesure.raw_to_lerobot(raw, calib)
    h1 = mesure.sim_deg_leisaac(raw, calib)
    h2 = mesure.sim_deg_degres(raw, calib)
    pa = passerelle(raw, calib, corr)
    lines = [f"{'articulation':14s} {'brut':>6s} {'LeRobot':>9s} {'sim leisaac':>12s} {'sim degres':>11s}"
             f" {'PASSERELLE':>11s}" + "".join(f" {'cible ' + k.upper():>8s}" for k in KEYS)]
    for j in mesure.JOINTS:
        unit = " " if j == "gripper" else "°"
        lines.append(f"{j:14s} {raw[j]:6d} {lr[j]:8.1f}{unit} {h1[j]:11.1f}° {h2[j]:10.1f}° {pa[j]:>11s}"
                     + "".join(f" {TARGETS[p][j]:7.1f}°" for p in KEYS.values()))
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", default="COM3")
    p.add_argument("--calib", default=str(DEFAULT_CALIB))
    p.add_argument("--out", default=str(chemins.MESURES))
    p.add_argument("--hz", type=float, default=5.0)
    args = p.parse_args()

    calib = mesure.load_calibration(args.calib)
    corr_file = chemins.CORRESPONDANCE
    corr = Correspondance.charger(corr_file, calib) if corr_file.is_file() else None  # colonne PASSERELLE
    bus = make_bus(args.port, calib)
    bus.connect(handshake=True)  # ping des 6 moteurs + version du firmware (lecture seule)
    print(f"Connecté à {args.port} : 6 moteurs trouvés.")
    try:
        motor_cal = {j: {"homing_offset": c.homing_offset, "range_min": c.range_min, "range_max": c.range_max}
                     for j, c in bus.read_calibration().items()}
        diffs = mesure.calibration_mismatches(calib, motor_cal)
        print("Calibration moteurs = fichier : OK" if not diffs else "ATTENTION calibration différente :\n  "
              + "\n  ".join(diffs))

        torque = bus.sync_read("Torque_Enable", normalize=False)
        if any(torque.values()):
            print(f"Couple ACTIF sur : {[j for j, t in torque.items() if t]}.")
            input("TIENS LE BRAS (il va devenir mou), puis appuie sur Entrée pour couper le couple...")
            bus.disable_torque()
        print("Couple coupé sur tous les moteurs. Place le bras à la main.\n")
        time.sleep(1.0)

        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        saved_file = out / "poses_reelles.json"
        # garde les poses déjà mesurées lors des lancements précédents (une nouvelle mesure remplace la même pose)
        saved = json.loads(saved_file.read_text(encoding="utf-8")) if saved_file.is_file() else {}
        while True:
            raw = {j: int(v) for j, v in bus.sync_read("Present_Position", normalize=False, num_retry=2).items()}
            sys.stdout.write("\033[H\033[J")
            print("   ".join(f"[{k}] enregistrer pose {k.upper()}" for k in KEYS) + "   [q] quitter"
                  f"   enregistrées : {sorted(saved) or '-'}\n")
            print(table(raw, calib, corr))
            if msvcrt.kbhit():
                k = msvcrt.getwch().lower()
                if k == "q":
                    break
                if k in KEYS:
                    hint = KEYS[k]  # ré-appuyer remplace la mesure de cette pose
                    saved[hint] = {
                        "time": datetime.now().isoformat(timespec="seconds"),
                        "raw": raw,
                        "lerobot": mesure.raw_to_lerobot(raw, calib),
                        "sim_deg_leisaac": mesure.sim_deg_leisaac(raw, calib),
                        "sim_deg_degres": mesure.sim_deg_degres(raw, calib),
                        "target_sim_deg": TARGETS[hint],
                    }
                    saved_file.write_text(json.dumps(saved, indent=1), encoding="utf-8")
            time.sleep(1.0 / args.hz)
    finally:
        bus.disconnect(disable_torque=True)  # ne fait que (re)couper le couple
    print(f"\nMesures : {Path(args.out) / 'poses_reelles.json'}")


if __name__ == "__main__":
    main()
