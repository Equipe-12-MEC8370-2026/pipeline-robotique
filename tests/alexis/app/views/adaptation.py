"""3 · Robot adaptation: the task projected onto the SO-101 (sizes, masses, places), its deviations, human review."""
from __future__ import annotations

import streamlit as st
import yaml

import core
from views import widgets

POSTURE = {"lying": "couchée", "on_edge": "sur la tranche", "upright": "debout", "unknown": "inconnue"}
_DEV = [  # the layout writes its deviations in English (retarget.layout): shown in French here
    (r"^(\w+): scaled x([\d.]+) to (.+?) \(demo (.+?)\) so the SO-101 grasps it$",
     r"\1 : réduite ×\2 à \3 (réelle \4) pour que la pince du SO-101 la saisisse"),
    (r"^(\w+): size (.+?) \(demo (.+?)\), set in the adaptation config$", r"\1 : taille \2 (réelle \3), fixée à la main"),
    (r"^(\w+): start posture unknown in the video -> lying \(most stable\)$",
     r"\1 : posture de départ inconnue dans la vidéo → couchée (la plus stable)"),
    (r"^(\w+): goal posture unknown in the video -> (\w+) \(as it starts\)$",
     r"\1 : posture d'arrivée inconnue dans la vidéo → identique au départ"),
    (r"^(\w+): starts (\w+) as at arrival \(demo: (\w+)\), no turning over$",
     r"\1 : part \2 comme à l'arrivée (démonstration : \3), car la bibliothèque n'a pas de geste pour retourner une pièce"),
    (r"^(\w+): starts (\w+) \(demo: (\w+)\)(.*)$", r"\1 : part \2 (démonstration : \3)\4"),
]


def deviation_fr(d: str) -> str:
    import re
    for pat, rep in _DEV:
        if re.match(pat, d):
            out = re.sub(pat, rep, d)
            for en, fr in POSTURE.items():
                out = out.replace(f" {en}", f" {fr}")
            return out
    return d


def render(T: core.Task) -> None:
    st.markdown("<div class='cta-step'>Étape 3</div>", unsafe_allow_html=True)
    st.title("Adaptation au robot")
    st.caption("Comment la tâche montrée est placée dans la cellule du robot : où les pièces arrivent, où elles doivent "
               "aller, et ce qui change par rapport à la démonstration. Vérifiez puis approuvez.")
    if not T.taskspec.exists():
        st.info("Construisez d'abord la description de la tâche (étape 2).")
        return
    lay = yaml.safe_load(T.layout.read_text(encoding="utf-8")) if T.layout.exists() else None
    stale = lay is None or T.layout.stat().st_mtime < T.taskspec.stat().st_mtime
    with st.container(border=True):
        if stale:
            st.info("La disposition doit être (re)calculée depuis la description de la tâche.")
        if st.button("Calculer la disposition" if stale else "Recalculer la disposition", type="primary" if stale else "secondary"):
            core.start_job(T.id, "layout", "Disposition robot", [
                {"name": "Disposition SO-101", "cmd": [str(core.PY_LEARN), "scripts/sim/make_robot_layout.py",
                                                       "--taskspec", str(T.taskspec)] + (["--adapt", str(T.adapt)] if T.adapt.exists() else []),
                 "cwd": str(core.REPO)}])
            st.rerun()
        layout_status(T)
    if not lay:
        return
    with st.container(border=True):
        st.subheader("Ce que fera le robot")
        widgets.temp_note("Robot SO-101 en simulation : pièces réduites et allégées pour sa petite pince. Avec le robot cible "
                          "et des pièces à sa taille, ces réductions disparaissent ; le placement dans la cellule reste.")
        rows = []
        for pid in lay["order"]:
            o, gl = lay["objects"][pid], lay["goals"][pid]
            where = (f"zone ({gl['xyyaw'][0]:.2f} ; {gl['xyyaw'][1]:.2f}) m" if gl["relation"] == "on_table"
                     else f"sur {gl['ref']}, décalage {gl['offset_xy_m'][0] * 1000:.0f} / {gl['offset_xy_m'][1] * 1000:.0f} mm")
            rows.append({"Pièce": pid, "Taille robot (mm)": " × ".join(f"{v * 1000:.0f}" for v in o["box_m"]),
                         "Taille réelle (mm)": " × ".join(f"{v * 1000:.0f}" for v in o["demo_size_m"]),
                         "Masse sim. (g)": round(o["mass_kg"] * 1000),
                         "Départ": f"{POSTURE.get(o['start_posture'], o['start_posture'])}, zone ({o['start_xyyaw'][0]:.2f} ; {o['start_xyyaw'][1]:.2f}) m",
                         "Arrivée": f"{POSTURE.get(gl['posture'], gl['posture'])}, {where}"})
        st.dataframe(rows, hide_index=True, use_container_width=True)
        st.markdown("**Écarts avec la démonstration**")
        for d in lay["deviations"]:
            st.markdown(f"- {deviation_fr(d)}")
        img = core.REPO / "outputs" / "taskspec_scene" / T.robot_id / "front_goal_state.png"
        if img.exists():
            st.image(str(img), caption="État final visé, en simulation", width=480)
        st.divider()
        widgets.approval(T, "layout", "J'approuve l'adaptation au robot")


@st.fragment(run_every="2s")
def layout_status(T: core.Task) -> None:
    st_ = widgets.job_header(T, "layout")
    widgets.rerun_when_finished(T, "layout", st_)
    if st_ and st_["status"] == "failed":
        widgets.log_box(core.job_log(T.id, "layout"))
