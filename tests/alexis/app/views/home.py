"""Overview: where the task stands, step by step."""
from __future__ import annotations

import streamlit as st

import core
from views.demos import episodes
from views.evaluation import checkpoints


def render(T: core.Task) -> None:
    st.title(T.cfg.get("instruction", {}).get("fr", T.id))
    st.caption(f"Tâche « {T.id} »" + (f" · démonstration {T.pair}" if T.pair else " · aucune vidéo pour l'instant"))
    ap = T.approvals()
    objs = T.cfg.get("objects") or {}
    parts_ok = bool(objs) and all(o.get("size_m") and o.get("material") for o in objs.values())
    parts_txt = ("Aucune pièce" if not objs else f"{len(objs)} pièce(s) prête(s)" if parts_ok
                 else f"{len(objs)} pièce(s), forme ou matériau à compléter")
    n_eps = len(episodes(T))
    cks = checkpoints(T)
    runs = sorted(T.eval_dir.glob("*/report.json"), key=lambda p: p.stat().st_mtime) if T.eval_dir.exists() else []
    runs = [p for p in runs if core.read_json(p, {}).get("info", {}).get("agent") == "policy"]   # trained models only
    best = None
    if runs:
        r = core.read_json(runs[-1], {})
        recs = r.get("records", [])
        best = f"{sum(x['outcome'] == 'success' for x in recs)} / {len(recs)} réussis"
    steps = [
        ("🧩", "Pièces", parts_txt, parts_ok),
        ("🎬", "Tâche", "Description approuvée" if ap.get("taskspec", {}).get("approved") else "Description à vérifier",
         ap.get("taskspec", {}).get("approved")),
        ("🦾", "Adaptation robot", "Approuvée" if ap.get("layout", {}).get("approved") else "À vérifier",
         ap.get("layout", {}).get("approved")),
        ("🎞️", "Démonstrations", f"{n_eps} gardées" if n_eps else "À produire", n_eps >= 50),
        ("🧠", "Entraînement", f"{len(cks)} modèle(s)" if cks else "À lancer", bool(cks)),
        ("✅", "Évaluation", best or "À lancer", bool(best)),
    ]
    cols = st.columns(3)
    for i, (icon, name, txt, ok) in enumerate(steps):
        with cols[i % 3]:
            cls = "cta-ok" if ok else "cta-warn"
            st.markdown(f"<div class='cta-card'><div style='font-size:1.4rem'>{icon}</div><b>{i + 1} · {name}</b><br>"
                        f"<span class='{cls}'>● </span>{txt}</div>", unsafe_allow_html=True)
    st.caption("Suivez les étapes dans l'ordre avec le menu de gauche. Les calculs longs continuent même si vous fermez cette page.")
