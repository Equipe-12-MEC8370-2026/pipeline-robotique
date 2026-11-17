# ADR-001 — Architecture C : hybride centrée sur l'objet

- Statut : **acceptée** (2026-11-11)
- Rapport complet : document « Projet CTA — Diagnostic, état de l'art et architecture du pipeline vidéo → robot » (Claude Docs)

## Contexte

Le CTA veut apprendre de nouvelles tâches à un robot à partir d'une paire de vidéos humaines (FPV + TPV). L'ancien entraînement SO-101 (SmolVLA) a échoué : unités incompatibles (degrés contre [-100, 100]), caméras inversées, écart réel→sim non traité, entraînement sous-dimensionné.

## Décision

1. Les vidéos servent à produire une spécification de tâche indépendante du robot (`TaskSpec` : objets, phases, contacts, trajectoires 6D d'objets).
2. Un retargeting propre au robot et un générateur de démonstrations en simulation (Isaac Lab) produisent des centaines d'épisodes robot randomisés au format LeRobot v3.
3. Une politique LeRobot (ACT en baseline, SmolVLA comme VLA ; GR00T N1.7 si un GPU ≥ 32 Go est disponible) est entraînée, validée en sim, puis co-entraînée avec quelques démos réelles.
4. Des démos téléopérées au même format restent possibles comme filet de sécurité (architecture B).

## Conséquences

- Une seule couche de conversion d'unités (`cta_learn.core.units`), avec radians partout ailleurs.
- La même interface `Policy` en simulation et sur le robot.
- Le passage au bimanuel ne change que la configuration d'embodiment et l'assignation des phases aux bras.
- P0 (démo du 20 novembre) construit la moitié « robot » : la scène, le générateur (expert scripté), l'entraînement ACT et l'évaluateur.
