"""Chemins de la passerelle : chaque script fait ``import chemins`` pour pouvoir importer les modules des autres
dossiers (mesure, conversion, mouvement, cameras…) et trouver les données sans chemin écrit en dur."""

import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent
DOSSIERS = ("correspondance", "trajectoire", "securite", "rejeu", "vision", "alignement", "smolvla")
for _d in DOSSIERS:
    if str(RACINE / _d) not in sys.path:
        sys.path.insert(0, str(RACINE / _d))

MESURES = RACINE / "mesures"
POSES = RACINE / "correspondance" / "poses" / "poses.json"
CORRESPONDANCE = RACINE / "correspondance" / "correspondance_greg.json"
VISION = RACINE / "vision"  # calibrations des caméras, damier, vues de calibration
ALIGNEMENT = RACINE / "alignement"  # vues de référence de la sim, captures d'alignement
ALEXIS = RACINE.parents[1] / "alexis"
DEMOS_P4 = ALEXIS / "data" / "raw" / "p4_pickplace_real"
