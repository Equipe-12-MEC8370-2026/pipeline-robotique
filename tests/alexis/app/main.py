"""CTA interface — entry point (streamlit run app/main.py, see Lancer_interface.bat).

One page per step of the pipeline, in order: parts, task (videos, TaskSpec, review), robot adaptation (review),
demonstrations, training, evaluation. The task being worked on is chosen (or created) in the sidebar.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
import core  # noqa: E402
from views import adaptation, demos, evaluation, home, parts, task, training  # noqa: E402

st.set_page_config(page_title="CTA · GBG", page_icon="🤖", layout="wide")

st.markdown("""
<style>
  .block-container {padding-top: 1.6rem; max-width: 1250px;}
  [data-testid="stSidebar"] {border-right: 1px solid #232a38;}
  .cta-brand {font-size: 2.6rem; font-weight: 800; letter-spacing: .06em; line-height: 1; margin: .2rem 0 .1rem 0;}
  .cta-sub {color: #8b95a9; font-size: .85rem; margin-bottom: 1rem;}
  .cta-step {font-size: 0.8rem; letter-spacing: .08em; text-transform: uppercase; color: #7f8aa3; margin-bottom: -.4rem;}
  .cta-card {background: #161B26; border: 1px solid #232a38; border-radius: 10px; padding: .9rem 1.1rem; margin-bottom: .6rem;}
  .cta-ok {color: #3fb950;} .cta-warn {color: #d29922;} .cta-bad {color: #f85149;}
  div[data-testid="stMetricValue"] {font-size: 1.6rem;}
</style>
""", unsafe_allow_html=True)


@st.dialog("Nouvelle tâche")
def new_task_dialog() -> None:
    st.caption("Décrivez la tâche en une phrase. Les pièces et les vidéos s'ajoutent ensuite (étapes 1 et 2).")
    fr = st.text_input("Consigne (français)", placeholder="Ex. : Pose le bloc à l'avant de la table.")
    en = st.text_input("Consigne (anglais, donnée au modèle)", placeholder="Ex. : Place the block at the front of the table.")
    if st.button("Créer", type="primary", disabled=not fr.strip()):
        st.session_state["task_id"] = core.create_task(fr.strip(), en.strip())
        st.rerun()


tasks = core.task_configs()
with st.sidebar:
    st.markdown("<div class='cta-brand'>CTA</div><div class='cta-sub'>Génération par bibliothèque de gestes</div>",
                unsafe_allow_html=True)
    if st.button("＋ Nouvelle tâche", use_container_width=True):
        new_task_dialog()
    if not tasks:
        st.info("Aucune tâche : créez-en une.")
if not tasks:
    st.title("Bienvenue")
    st.markdown("Pour commencer, cliquez sur **＋ Nouvelle tâche** dans la barre de gauche et décrivez la tâche en une phrase. "
                "Vous ajouterez ensuite les pièces (étape 1) et les vidéos de la démonstration (étape 2).")
    st.stop()
with st.sidebar:
    ids = list(tasks)
    # the chosen task is kept in the page address (?task=...): it survives a reload or a new tab
    default = st.session_state.get("task_id") or st.query_params.get("task") or ids[0]
    tid = st.selectbox("Tâche", ids, index=ids.index(default) if default in ids else 0,
                       format_func=lambda t: (core.load_yaml(tasks[t]).get("instruction", {}).get("fr") or t)[:60],
                       help="Chaque tâche garde ses propres pièces, vidéos, démonstrations, modèles et évaluations, "
                            "sur le disque : changer de tâche ne perd rien.")
    st.session_state["task_id"] = tid
    if st.query_params.get("task") != tid:
        st.query_params["task"] = tid
    running = core.running_jobs()
    if running:
        st.markdown("**En cours**")
        for t, k in running:
            st.markdown(f"<span class='cta-warn'>●</span> {k} · {t}", unsafe_allow_html=True)
    else:
        st.caption("Aucun calcul en cours.")
    st.caption("Fermer cette page n'arrête pas les calculs lancés.")

T = core.Task(tid, tasks[tid])

pages = [
    st.Page(lambda: home.render(T), title="Vue d'ensemble", icon="🏠", url_path="accueil", default=True),
    st.Page(lambda: parts.render(T), title="1 · Pièces", icon="🧩", url_path="pieces"),
    st.Page(lambda: task.render(T), title="2 · Tâche", icon="🎬", url_path="tache"),
    st.Page(lambda: adaptation.render(T), title="3 · Adaptation robot", icon="🦾", url_path="adaptation"),
    st.Page(lambda: demos.render(T), title="4 · Démonstrations", icon="🎞️", url_path="demos"),
    st.Page(lambda: training.render(T), title="5 · Entraînement", icon="🧠", url_path="entrainement"),
    st.Page(lambda: evaluation.render(T), title="6 · Évaluation", icon="✅", url_path="evaluation"),
]
st.navigation(pages).run()
