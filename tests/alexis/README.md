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

## Scènes Isaac Lab

| ID | Fichier | Contenu |
| --- | --- | --- |
| `CTA-SO101-PickPlace-v0` | `src/cta_learn/sim/tasks/pick_place.py` | P0 : cube rouge → zone verte (démo du 20 nov.) |
| `CTA-SO101-PickPlaceReal-v0` | `src/cta_learn/sim/tasks/pick_place_real.py` | **P4 (nouveau)** : même tâche, aspect du montage réel |
| `CTA-SO101-ProfileStack-v0` | `src/cta_learn/sim/tasks/profile_stack.py` | P3 : profilés (démo CTA 113656) |
| `CTA-SO101-TaskSpec-v0` | `src/cta_learn/sim/tasks/taskspec_task.py` | scène générique créée depuis une TaskSpec (`CTA_LAYOUT`) |

### P4 : scène « comme le réel » (premier essai sur le vrai SO-101)

- Support de caméra poignet imprimé (`assets/props/camera_mount_v2.usd`, voir `assets/props/NOTICE.md`) posé sur la
  seule paire de trous M3 de la pince (`scripts/sim/fit_wrist_mount.py`) ; caméra poignet au centre du support
  (`src/cta_learn/sim/wrist_mount.py`), pose légèrement variée à chaque épisode.
- Robot blanc, cube orange, zone blanche (papier), table en bois ; caméra externe rapprochée
  (œil (0,40 ; 0 ; 0,38) m, visée (0,22 ; 0,02 ; 0), 90° horizontal).
- Les vraies caméras (Waveshare IMX335, fisheye 175°) seront redressées vers ces caméras sans distorsion.

```powershell
# Images de contrôle (avant, poignet, gros plans du support) -> outputs\p4_1\scene_check\
$env:PYTHONPATH = "$PWD\src"; & ..\isaacsim-env\Scripts\python.exe scripts\sim\check_real_scene.py --headless --enable_cameras
# Expert scripté, 20 essais + 3 vidéos (--variant real ; sans l'option : scène P0)
$env:PYTHONPATH = "$PWD\src"; & ..\isaacsim-env\Scripts\python.exe scripts\sim\gen_demos.py --headless --enable_cameras --variant real --attempts 20 --debug_videos 3 --report outputs\p4_1\gate.json
# Chaîne complète (Git Bash) : 200 démos -> dataset LeRobot -> ACT 100K pas -> évaluation sur 50 essais
bash scripts/pipelines/p4_real_chain.sh 200
```

## Interface locale

`Lancer_interface.bat` ouvre l'interface Streamlit (`app/`, guide `app/README.md`, environnement `.venv-app`).

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
