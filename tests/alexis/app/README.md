# Interface CTA — robot par démonstration

Application locale : elle tourne sur cet ordinateur, s'ouvre dans le navigateur à l'adresse `localhost:8501`
et n'est accessible à personne d'autre. Pas de serveur permanent, pas de nom de domaine.

## Lancer

Double-cliquer sur `Lancer_interface.bat` (dossier `cta-learn`). Fermer la fenêtre noire arrête l'interface.
Les calculs déjà lancés (démonstrations, entraînement, évaluation) **continuent** : on peut fermer
l'interface et la rouvrir plus tard pour voir où ils en sont.

## Les étapes (menu de gauche)

En haut à gauche : **＋ Nouvelle tâche**, et le menu des tâches. Chaque tâche garde sur le disque ses pièces, vidéos,
démonstrations, modèles et évaluations ; la tâche choisie reste dans l'adresse de la page (`?task=…`).

1. **Pièces** (jusqu'à 5, bouton « ＋ Ajouter une pièce ») : forme donnée par la **CAO** (recommandé, le plus précis),
   un **scan 3D au téléphone** (OBJ/GLB), une **saisie manuelle** des dimensions, ou la **mesure sur la vidéo**
   (Temporaire). Masse = volume × densité du matériau, modifiable. Alerte si la pièce dépasse la pince ou la charge du
   robot (`app/robot_limits.yaml`).
2. **Tâche** : dépôt des vidéos FPV + TPV (jusqu'à 5 démonstrations ; seule la 1re sert pour l'instant),
   synchronisation par le son et mains ; pièces encadrées à la souris au départ et à l'arrivée (Temporaire) ;
   installation de la caméra (Temporaire) ; suivi des pièces et description ; consigne ; « J'approuve ».
3. **Adaptation robot** : placement de la tâche dans la cellule du robot, écarts avec la démonstration ; « J'approuve ».
4. **Démonstrations** : temps estimé, barre de progression, taux de réussite, vidéos. « N° de tirage » dans les
   options avancées.
5. **Entraînement** : temps estimé, barre de progression, temps restant, courbe d'apprentissage.
6. **Évaluation** : 5, 20 ou 50 essais, temps estimé, taux de réussite, causes d'échec, vidéos.

Les étiquettes orange **Temporaire** marquent ce qui n'existera plus dans le pipeline final.
Un seul calcul à la fois (une carte graphique) : l'interface prévient si un calcul est déjà en cours.

## Installation (une fois)

    py -3.12 -m venv .venv-app
    .venv-app\Scripts\python.exe -m pip install -r app\requirements.txt

L'interface pilote les scripts du projet avec `.venv-learn` (données, entraînement) et `..\isaacsim-env`
(simulation). États des calculs et approbations : `app_data/` (non versionné).

## Limites actuelles

- Une **nouvelle installation** de caméra (autre position) doit encore être calibrée hors de l'interface.
- Plusieurs démonstrations sont gardées, mais seule la 1re sert à la description (fusion pas encore codée).
- La CAO donne la taille réelle et la masse ; elle ne sert pas encore à trouver les pièces dans la vidéo.
