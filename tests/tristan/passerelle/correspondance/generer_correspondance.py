r"""Génère correspondance_greg.json à partir des poses mesurées (mesures\poses_reelles.json) et de la calibration.

Méthode (décisions du 2026-11-25, voir HANDOFF) :
  * shoulder_lift, elbow_flex, wrist_flex : moyenne des poses D et E (placées au fil à plomb, repères = axes) ;
    les poses A, B, C (reproduites à l'œil) répartissaient mal l'angle entre ces articulations ;
  * shoulder_pan : moyenne de A, B, D, E (C : base placée à 30° au lieu de 45°) ;
  * wrist_roll : moyenne des 5 poses ;
  * pince : linéaire [0, 100] -> limites sim [-10, 100]° (vérifié : fermée ≈ 0, ouverte ≈ 95-100).
Incertitude = demi-écart max entre les poses utilisées.

    ..\..\alexis\.venv-learn\Scripts\python.exe correspondance\generer_correspondance.py
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path

import numpy as np

import mesure

HERE = Path(__file__).resolve().parent
MESURES = HERE.parent / "mesures"
POSES_PAR_ARTICULATION = {
    "shoulder_pan": "ABDE",
    "shoulder_lift": "DE",
    "elbow_flex": "DE",
    "wrist_flex": "DE",
    "wrist_roll": "ABCDE",
}


def calculer(mesures: dict, calib: dict[str, mesure.Calib]) -> dict:
    par_lettre = {k[0]: v for k, v in mesures.items()}
    decal, incert, detail = {}, {}, {}
    for j, lettres in POSES_PAR_ARTICULATION.items():
        manque = [l for l in lettres if l not in par_lettre]
        if manque:
            raise ValueError(f"{j} : poses manquantes {manque}")
        d = np.array([par_lettre[l]["target_sim_deg"][j] - par_lettre[l]["lerobot"][j] for l in lettres])
        decal[j] = round(float(d.mean()), 2)
        incert[j] = round(float(np.ptp(d) / 2), 2)
        detail[j] = {l: round(float(v), 2) for l, v in zip(lettres, d)}
    return {
        "description": "Correspondance SO-101 réel (LeRobot so101_follower, use_degrees=True) <-> sim Isaac (URDF). "
                       "Bras : sim_deg = lerobot_deg + decalage_deg. Pince : sim_deg linéaire de pince_sim_deg[0] "
                       "(valeur 0) à pince_sim_deg[1] (valeur 100).",
        "date": date.today().isoformat(),
        "robot_id": "Greg",
        "decalage_deg": decal,
        "incertitude_deg": incert,
        "decalage_par_pose_deg": detail,
        "pince_sim_deg": list(mesure.SIM_LIMITS_DEG["gripper"]),
        "calibration": {j: {"homing_offset": c.homing_offset, "range_min": c.range_min, "range_max": c.range_max}
                        for j, c in calib.items()},
        "methode": {j: f"moyenne des poses {'+'.join(l)}" for j, l in POSES_PAR_ARTICULATION.items()},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mesures", default=str(MESURES / "poses_reelles.json"))
    p.add_argument("--calib", default=str(Path(os.environ["USERPROFILE"])
                                          / ".cache/huggingface/lerobot/calibration/robots/so_follower/Greg.json"))
    p.add_argument("--out", default=str(HERE / "correspondance_greg.json"))
    args = p.parse_args()
    corr = calculer(json.loads(Path(args.mesures).read_text(encoding="utf-8")), mesure.load_calibration(args.calib))
    Path(args.out).write_text(json.dumps(corr, indent=1, ensure_ascii=False), encoding="utf-8")
    for j in POSES_PAR_ARTICULATION:
        print(f"{j:14s} décalage {corr['decalage_deg'][j]:+6.2f}°  ± {corr['incertitude_deg'][j]:.2f}°  "
              f"({corr['methode'][j]})")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
