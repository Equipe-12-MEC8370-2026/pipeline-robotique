# Versions et dépendances

Relevé le 2026-11-11 sur la station de développement (lecture seule des installations existantes).

## Matériel et système

| Élément | Valeur |
| --- | --- |
| GPU | NVIDIA RTX 4070 SUPER, 12 Go, pilote 610.88 (CUDA UMD 13.3) |
| CPU | Intel i5-12600K |
| RAM | 15,8 Go (sous les 32 Go recommandés pour Isaac Sim) |
| OS | Windows 11 Pro 10.0.26200 ; pas de WSL, Docker ni ROS 2 |
| Outils | Git 2.55, ffmpeg 9.0.1 (winget) |

## Runtime simulation — `..\isaacsim-env` (existant, non modifié)

| Paquet | Version |
| --- | --- |
| Python | 3.12.10 |
| isaacsim | 6.1.0.0 |
| Isaac Lab | dépôt `..\IsaacLab`, branche `release/3.0.0-beta2`, commit `99f1423e5` (paquet pip `isaaclab` 6.1.17) |
| torch | 2.10.0+cu128 |
| numpy | 2.3.1 |
| rsl-rl-lib / skrl | 5.0.1 / 2.1.0 |
| h5py | 3.16.0 |

`cta_learn` y est chargé par `PYTHONPATH=src` (vérifié par `scripts/check_sim_env.py`). Rien n'est installé dans cet environnement.

## Runtime apprentissage — `.venv-learn` (propre au projet)

| Paquet | Version |
| --- | --- |
| Python | 3.12.10 |
| torch / torchvision | 2.11.0 / 0.26.0 (index cu128) |
| lerobot | 0.6.1 |
| transformers / tokenizers | 5.5.4 / 0.22.2 (extra `smolvla`, ajouté le 2026-11-12, P1.1) |
| num2words | 0.5.14 (extra `smolvla`) |
| pytest | voir `pip freeze` |

Recréer : voir le README.

## Runtime perception — `.venv-perception` (propre au projet, créé le 2026-11-13, P2.4a)

Séparé de `.venv-learn` : MediaPipe impose `opencv-contrib-python`, qui entrerait en conflit avec `opencv-python-headless` de LeRobot.

| Paquet | Version |
| --- | --- |
| Python | 3.12.10 |
| torch / torchvision | 2.11.0 / 0.26.0 (index cu128, mêmes versions que `.venv-learn`, inchangées après les autres installations) |
| SAM 2 (`SAM-2`) | dépôt officiel `facebookresearch/sam2`, commit `2b90b9f5ceec907a1c18123530e92e794ad901a4`, installé sans l'extension CUDA (`SAM2_BUILD_CUDA=0` : seul le bouchage des trous des masques est sauté). **Ne pas utiliser le paquet PyPI `sam2`** : republication par un tiers. |
| mediapipe | 1.2.3 (Google) ; pas de GPU sous Windows (CPU, ~20–25 ms par image) |
| opencv-contrib-python / numpy | 5.0.0 / 2.5.4 |
| hydra-core / iopath / av / pyyaml / pytest | 1.3.8 / 0.1.10 / 19.0.2 / 6.0.3 / 9.2.0 |
| `cta-learn` | installé en éditable sans dépendances (`pip install -e . --no-deps`) |

`pip check` : OK. Modèles (`data/pretrained/perception/`, `scripts/perception/fetch_perception_models.py`, sha256 figés dans le script, enregistrés au premier téléchargement faute de sommes publiées) :

| Fichier | Source | sha256 |
| --- | --- | --- |
| `hand_landmarker.task` (7,8 Mo) | Google, `storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/` | `fbc2a300…3607cde1` |
| `sam2.1_hiera_small.pt` (184 Mo) | Meta, `dl.fbaipublicfiles.com/segment_anything_2/092824/` | `6d1aa6f3…5dc4d38` |

## Sources d'assets (lecture seule)

| Asset | Origine |
| --- | --- |
| Limites articulaires SO-101 | LeIsaac 0.4.0, `leisaac/assets/robots/lerobot.py` (`SO101_FOLLOWER_USD_JOINT_LIMLITS`) |
| USD SO-101 | LeIsaac 0.4.0 `assets/robots/so101_follower.usd` (à copier dans `assets/` avec sa licence à l'étape 2) |
