"""4 · Demonstrations: the gesture library plays the task in simulation; the successes are kept (progress, videos)."""
from __future__ import annotations

import math
import time

import streamlit as st

import core
from views import widgets


def episodes(T: core.Task) -> list:
    return sorted(p for p in T.raw.glob("episode_*") if p.is_dir() and (p / "front.mp4").exists()) if T.raw.exists() else []


def demo_speed(T: core.Task) -> tuple[float, float, bool]:
    """(seconds per attempt, success rate, measured?) from this task's last generation report, else defaults:
    ~16 s per pick-and-place step (measured 2026-11-18: block 16.8 s for 1 step, profiles 31.4 s for 2)."""
    for p in sorted((core.REPO / "outputs" / "taskspec_demos" / T.robot_id).glob("gate*.json"),
                    key=lambda p: p.stat().st_mtime, reverse=True):
        d = core.read_json(p, {})
        if d.get("attempts"):
            return d["elapsed_s"] / d["attempts"], max(d["success_rate"], 0.05), True
    n_steps = max(1, len(T.cfg.get("objects") or {}))
    return 16.0 * n_steps, 0.95, False


def render(T: core.Task) -> None:
    st.markdown("<div class='cta-step'>Étape 4</div>", unsafe_allow_html=True)
    st.title("Démonstrations")
    st.caption("Le robot simulé fait la tâche avec la bibliothèque de gestes, à partir de positions de départ tirées au "
               "hasard. Seules les réussites sont gardées : elles serviront à l'entraînement.")
    ok_spec = T.approvals().get("taskspec", {}).get("approved")
    ok_lay = T.approvals().get("layout", {}).get("approved")
    if not (ok_spec and ok_lay):
        st.warning("Approuvez d'abord la description de la tâche (étape 2) et l'adaptation au robot (étape 3).")
    have = len(episodes(T))

    with st.container(border=True):
        c1, c2, c3 = st.columns([1, 1, 1])
        n = c1.number_input("Démonstrations à produire", 10, 1000, 300, 10,
                            help="300 à 500 pour un bon entraînement ; 10 à 20 pour un essai rapide.")
        per_demo, rate, measured = demo_speed(T)
        est = 60 + n / max(rate, 0.05) * per_demo
        c2.metric("Temps estimé", widgets.fmt_dur(est),
                  help=(f"Mesuré sur cette tâche : {per_demo:.0f} s par tentative, {rate:.0%} de réussite, "
                        if measured else f"Estimation par défaut : {per_demo:.0f} s par tentative, ") + "+ 1 min de démarrage d'Isaac Sim.")
        c3.metric("Démonstrations déjà gardées", have)
        with st.expander("Options avancées"):
            seed = st.number_input(
                "N° de tirage", 0, 9999, 1,
                help="Fixe les tirages au hasard : où les pièces apparaissent au départ et les petites variations du geste. "
                     "Même numéro = exactement les mêmes démonstrations (pour refaire un essai à l'identique) ; autre "
                     "numéro = un autre lot. La valeur elle-même ne veut rien dire (1 n'est pas meilleur que 42). "
                     "Évitez 7 : l'évaluation l'utilise, le modèle serait testé sur des situations déjà vues.")
            st.caption("Ajouter des démonstrations à un lot existant : prenez un autre n° de tirage.")
        st_ = core.job_state(T.id, "demos")
        running = bool(st_ and st_["status"] == "running")
        if not running and not widgets.busy_warning(T, "demos"):
            if st.button("Lancer la génération", type="primary", disabled=not (ok_spec and ok_lay)):
                core.start_job(T.id, "demos", "Génération des démonstrations", [
                    {"name": "Génération (Isaac Sim)", "cwd": str(core.REPO), "env": T.env(),
                     "cmd": [str(core.PY_SIM), "scripts/sim/gen_taskspec_demos.py", "--headless", "--enable_cameras",
                             "--record", "--successes", str(int(n)), "--attempts", str(int(math.ceil(n * 1.3)) + 5),
                             "--seed", str(int(seed)), "--out", str(T.raw), "--tag", "app"]}])
                core.write_json(T.data / "demos_target.json", {"target": int(n), "start_count": have})
                st.rerun()
        progress(T)

    eps = episodes(T)
    if eps:
        with st.container(border=True):
            st.subheader("Voir une démonstration")
            i = st.select_slider("Démonstration", options=list(range(len(eps))), value=len(eps) - 1,
                                 format_func=lambda k: eps[k].name.replace("episode_", "n° "))
            a, b = st.columns(2)
            a.markdown("**Caméra avant**")
            a.video(str(eps[i] / "front.mp4"))
            b.markdown("**Caméra poignet**")
            b.video(str(eps[i] / "wrist.mp4"))


@st.fragment(run_every="3s")
def progress(T: core.Task) -> None:
    st_ = widgets.job_header(T, "demos")
    widgets.rerun_when_finished(T, "demos", st_)
    if not st_:
        st.caption("Aucune génération lancée depuis l'interface.")
        return
    log = core.job_log(T.id, "demos")
    g = core.parse_generation(log)
    target = core.read_json(T.data / "demos_target.json", {}).get("target", 0) or 1
    st.progress(min(g["successes"] / target, 1.0), text=f"{g['successes']} / {target} démonstrations réussies "
                                                        f"({g['attempts']} tentatives)")
    if g["attempts"]:
        rate = g["successes"] / g["attempts"]
        if st_["status"] == "running" and g["successes"] < target:
            el = time.time() - st_.get("started", time.time())
            per = el / g["attempts"]
            st.caption(f"Reste environ {widgets.fmt_dur((target - g['successes']) / max(rate, 0.05) * per)}")
        cls = "cta-ok" if rate >= 0.9 else "cta-warn" if rate >= 0.7 else "cta-bad"
        st.markdown(f"Taux de réussite du geste : <span class='{cls}'>**{rate:.0%}**</span>", unsafe_allow_html=True)
    elif st_["status"] == "running":
        st.caption("Démarrage d'Isaac Sim (environ une minute)…")
    if g["failures"]:
        st.markdown("**Derniers échecs** (écartés automatiquement)")
        for a, o in g["failures"][-5:]:
            st.markdown(f"- tentative {a} : `{o.strip()}`")
    if st_["status"] == "done":
        st.success(f"Terminé : {g['successes']} démonstrations gardées. Passez à l'entraînement (étape 5).")
    widgets.stop_button(T, "demos", st_)
    widgets.log_box(log, keep=lambda ln: ln.startswith("[") or "Error" in ln or "Traceback" in ln)
