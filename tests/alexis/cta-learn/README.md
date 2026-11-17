# cta-learn

Pipeline d'apprentissage robotique du projet CTA (MEC8370, équipe 12) : démonstration vidéo humaine (FPV + TPV) → spécification de tâche → démonstrations robot générées en simulation (Isaac Lab) → politique (LeRobot) → robot réel.

Architecture retenue : **C, hybride centrée sur l'objet** (voir `docs/decisions/ADR-001-architecture-c.md`).
Plateforme de développement : SO-101. Cible finale : robot bimanuel.

Ce projet est indépendant de l'ancien entraînement SO-101 : il n'en reprend ni code, ni checkpoints, ni datasets.

## Environnements

| Rôle | Environnement | Comment |
| --- | --- | --- |
| Simulation (Isaac Sim 6.1 / Isaac Lab 3.0) | `..\isaacsim-env` existant, **non modifié** | le code est chargé par `PYTHONPATH=src`, rien n'y est installé |
| Apprentissage, perception, tests | `.venv-learn` (dans ce repo, ignoré par Git) | `pip install -e .[dev,train]` |

Versions exactes : `docs/versions.md`.

## Démarrage rapide (PowerShell)

```powershell
# Tests unitaires (sans GPU)
.\.venv-learn\Scripts\python.exe -m pytest

# Vérifier que l'environnement de simulation voit le paquet
$env:PYTHONPATH = "$PWD\src"; & ..\isaacsim-env\Scripts\python.exe scripts\check_sim_env.py
```

## Structure

```text
configs/embodiments/   description des robots (articulations, limites, caméras)
src/cta_learn/core/    types partagés et unique couche de conversion d'unités
tests/                 tests pytest (marqueurs : sim, gpu, robot)
scripts/               points d'entrée CLI
docs/                  versions, décisions (ADR), page de résultats
data/, outputs/        artefacts locaux, ignorés par Git
```

## Règle des unités

Tout le code manipule les articulations en **radians** (unités SI). Les autres formats (degrés, plage normalisée LeRobot [-100, 100] / [0, 100]) ne sont produits ou lus que par `cta_learn.core.units`, qui valide les plages et refuse les valeurs incohérentes.
