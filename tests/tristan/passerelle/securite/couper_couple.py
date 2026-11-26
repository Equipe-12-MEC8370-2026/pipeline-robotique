r"""Coupe le couple de tous les moteurs du SO-101 (le bras devient mou), avec nouvelles tentatives, puis vérifie.

À utiliser si un script s'est arrêté en laissant des moteurs sous couple. Le bras doit être dans une position où il
ne tombe pas (repliée), ou tenu à la main.

Depuis tests\tristan\passerelle, dans TON terminal :
    ..\..\alexis\.venv-learn\Scripts\python.exe securite\couper_couple.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chemins  # noqa: E402  (rend les autres dossiers de la passerelle importables)
import mesure  # noqa: E402
from lire_articulations import DEFAULT_CALIB, make_bus  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", default="COM3")
    p.add_argument("--calib", default=str(DEFAULT_CALIB))
    args = p.parse_args()
    bus = make_bus(args.port, mesure.load_calibration(args.calib))
    bus.connect(handshake=True)
    try:
        avant = bus.sync_read("Torque_Enable", normalize=False, num_retry=3)
        actifs = [j for j, t in avant.items() if t]
        if not actifs:
            print("Couple déjà coupé sur tous les moteurs : rien à faire.")
            return
        print(f"Couple ACTIF sur : {actifs}")
        input("Bras replié ou TENU à la main ? Entrée pour couper le couple (le bras devient mou)... ")
        bus.disable_torque(num_retry=3)
        apres = bus.sync_read("Torque_Enable", normalize=False, num_retry=3)
        print("Couple coupé sur tous les moteurs." if not any(apres.values()) else f"ATTENTION, encore actif : {apres}")
    finally:
        bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()
