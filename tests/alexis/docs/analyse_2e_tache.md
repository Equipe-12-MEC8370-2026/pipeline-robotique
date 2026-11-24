# Analyse de la 2e tâche (étape 1 du plan D13) — 2026-11-18

But : voir ce qui manque pour faire tourner une 2e tâche CTA **sans code propre à la tâche**.
Méthode : lecture du code actuel, planches d'images `outputs/p2_2/sheet_*.jpg`, images extraites des clips
synchronisés, rapport `outputs/p2_5/segmentation_report.json`. Aucun code modifié, aucun calcul lancé.

## 1. Ce que montrent les deux vidéos d'équerre

| | 113952 | 114110 |
| --- | --- | --- |
| Installation caméra | A (comme la tâche des profilés) | B (calibrée directement) |
| Objets | « équerre » + foret | « équerre » + petite vis |
| Déroulé (étiquettes brouillon, ± 1–2 s) | prendre l'équerre au râtelier, la poser à l'avant (6–10 s) ; la tenir en main et y passer le foret (11–28 s) ; reposer (29–33 s) | prendre l'équerre, la poser à l'avant (9–13 s) ; prendre la vis (14–17 s) ; la visser dans l'équerre posée (17–24 s) ; soulever l'équerre à deux mains, la contrôler, la reposer (24–28 s) |
| Gestes début/fin détectés | oui (début et fin) | début oui ; **fin manquée** (cas limite connu) |
| Partie faisable par le robot aujourd'hui | prise-dépose de l'équerre | prise-dépose de l'équerre |
| Partie hors de portée | perçage à la main, pièce tenue en l'air (bimanuel, contact) | vissage d'une petite vis (insertion fine) |

**Constat important** : l'« équerre » n'est pas une cornière mince. C'est un **bloc d'aluminium usiné**
(gros, plein, avec une face en biais et des trous). Sa boîte englobante a trois côtés différents et la pièce
n'est **pas symétrique** : son orientation sur la table compte (quelle face en bas, de quel côté la face en biais).

## 2. Ce qui manque, étape par étape du pipeline

| Étape | Déjà générique | Manque pour l'équerre |
| --- | --- | --- |
| Perception des mains | oui (mains déjà calculées pour les 6 vidéos) | rien |
| Perception des objets (SAM 2) | `track_objects.py` lit un fichier de consignes | écrire les cadres de départ de l'équerre à la main (comme pour les profilés) ; le foret et la vis sont probablement trop petits en vue TPV (à vérifier) |
| Pose des objets sur la table | calibration, repos et déplacements | `object_trajectories.py` contient les noms `profile_short` / `profile_long` (hauteurs de base, contrôle final) ; l'ajustement de boîte suppose une **section carrée** et deux postures (couché/debout) : il ne sait pas dire quelle face du bloc est en bas ni son orientation sur 360° |
| Dimensions | — | mesurer le bloc sur l'image calibrée (pas encore fait) |
| Description de la tâche (TaskSpec) | formes « boîte », relations « sur la table » et « sur une autre pièce » | posture limitée à couché/debout ; pas de relation « inséré dans » (vis) ni d'étape « tenir pendant qu'on travaille » (foret) ; `build_taskspec.py` a la vidéo, les dimensions et la consigne écrites en dur |
| Adaptation au robot | — | `retarget/profile_stack.py` est écrit pour 2 profilés (noms, mise à l'échelle D10, départ couché D11, zones) |
| Scène de simulation | — | `sim/tasks/profile_stack.py` déclare 2 pièces nommées, leurs observations, leur placement aléatoire et le critère |
| Bibliothèque de gestes | 1 geste : saisir-poser une boîte | la prise suppose une barre couchée ou une section carrée (symétrie de 180° ou 90°) ; pas de pose avec une orientation imposée sur 360° ; pas de retournement si la face du bas change entre le départ et l'arrivée |
| Critère de réussite | — | `stack_success` et `stack_metrics.py` sont écrits pour les 2 profilés |
| Gestes absents | — | percer en tenant la pièce, visser : gestes de contact, prévus plus tard (D13, point 5) |

## 3. Conséquences pour la suite

1. **Ce qu'on peut viser maintenant** : la partie prise-dépose de l'équerre (prendre le bloc au râtelier,
   le poser à l'avant dans l'orientation montrée). C'est une tâche à **1 étape**, donc plus simple que les profilés,
   mais elle teste ce que la 1re tâche ne testait pas : une autre forme, trois dimensions différentes et une
   orientation qui compte.
2. **Vidéo conseillée : 114110.** Le bloc reste posé sur la table pendant le vissage, ses moments de repos sont
   donc longs et nets. Dans 113952, le bloc est tenu en l'air pendant le perçage et le code actuel prendrait ce long
   déplacement pour la manipulation principale. Le geste de fin manqué de 114110 ne touche que la phase de retrait.
3. **Travail de généralisation, dans l'ordre** :
   1. pose d'objet : orientation complète (face du bas + angle sur 360°) et boîte à trois côtés quelconques ;
      retirer les noms en dur ;
   2. TaskSpec : posture remplacée par « face du bas + angle », version du schéma relevée, la tâche des profilés
      convertie et revérifiée ;
   3. adaptation, scène et critère de réussite lus depuis la TaskSpec, pour un nombre de pièces quelconque ;
   4. bibliothèque : saisir et poser une boîte quelconque avec une orientation imposée ;
   5. preuve : refaire les profilés avec le nouveau code (viser environ 44/50), puis l'équerre sans code ajouté.
4. **Hors de portée pour l'instant** : vis et foret (insertion, contact, bimanuel).

## 4. Résultats de la généralisation de la pose (2026-11-18)

- La pose des objets accepte une boîte à trois côtés quelconques ; les profilés redonnent exactement les mêmes poses.
- Équerre de 114110 : SAM 2 la suit sur toute la vidéo (469/469 images) ; taille mesurée 71 × 69 × 62 mm ;
  positions stables à moins de 3 mm.
- Limite : le bloc est presque cubique, la silhouette ne dit pas quelle face est en bas. L'orientation demandera
  sa forme réelle (CAO) ou une autre vue.
- Taille pour le robot : réduite à une taille raisonnable pour le SO-101 à l'étape d'adaptation.

## 5. Points à vérifier avant de coder (état initial)

- Taille du bloc et ouverture de la pince du SO-101 : en simulation on réduit l'échelle (D10), mais il faut le noter.
- Masse : bloc plein, lourd pour le SO-101 ; masse allégée en simulation comme pour les profilés (D9).
- SAM 2 suit-il le bloc à travers les occultations par les mains (pendant le vissage) ?
