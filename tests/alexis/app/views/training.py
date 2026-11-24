"""5 · Training: demonstrations -> LeRobot dataset -> ACT (progress bar, loss curve, time left)."""
from __future__ import annotations

import re
import time

import plotly.graph_objects as go
import streamlit as st
import yaml

import core
from views import widgets
from views.demos import episodes


CONVERT_S_PER_DEMO = 4.0   # raw -> LeRobot dataset, video re-encoding (JOURNAL: ~10 min for 150 episodes)
DEFAULT_RATE = 6.0         # ACT steps/s on this GPU (P3: 200 000 steps in 8 h 50)


def step_rate() -> float:
    """ACT training speed (steps/s) read from the last tqdm line of the latest training log, else the default."""
    logs = sorted((core.REPO / "outputs" / "train").glob("act_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    for lp in logs[:3]:
        try:
            with open(lp, encoding="utf-8", errors="replace") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 20000))
                tail = f.read()
        except OSError:
            continue
        rates = re.findall(r"([\d.]+)step/s\]", tail)
        if rates:
            r = float(rates[-1])
            if 0.5 < r < 50:
                return r
    return DEFAULT_RATE


def write_train_config(T: core.Task, steps: int, out_dir: str) -> str:
    tpl = yaml.safe_load((core.CONFIGS / "pipelines" / "train_act_p3.yaml").read_text(encoding="utf-8"))
    tpl["dataset"]["repo_id"] = f"cta/{T.id}_app"
    tpl["dataset"]["root"] = str(T.dataset.relative_to(core.REPO).as_posix())
    tpl.update({"output_dir": out_dir, "job_name": T.id, "steps": int(steps), "save_freq": max(1000, int(steps) // 10)})
    path = T.data / "train_act.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# Written by the CTA interface from configs/pipelines/train_act_p3.yaml (same ACT settings).\n"
                    + yaml.safe_dump(tpl, sort_keys=False), encoding="utf-8")
    return str(path)


def render(T: core.Task) -> None:
    st.markdown("<div class='cta-step'>Étape 5</div>", unsafe_allow_html=True)
    st.title("Entraînement")
    st.caption("Le modèle (ACT) apprend à refaire la tâche à partir des images des caméras et des démonstrations.")
    n_eps = len(episodes(T))
    with st.container(border=True):
        c1, c2, c3 = st.columns([1.4, 1, 1])
        steps = c1.select_slider("Durée d'entraînement (pas)", options=[2000, 10000, 50000, 100000, 200000], value=100000,
                                 help="Un pas = le modèle corrige ses erreurs sur un petit paquet d'images. 2 000 pas = essai "
                                      "rapide ; 100 000 à 200 000 pour un modèle utilisable (200 000 pour la tâche des profilés).")
        rate = step_rate()
        est = 60 + n_eps * CONVERT_S_PER_DEMO + steps / rate
        c2.metric("Temps estimé", widgets.fmt_dur(est),
                  help=f"Préparation du jeu de données (~{CONVERT_S_PER_DEMO:.0f} s par démonstration) + entraînement "
                       f"(~{rate:.1f} pas/s mesurés sur cette carte) + 1 min de démarrage.")
        c3.metric("Démonstrations disponibles", n_eps)
        st_ = core.job_state(T.id, "train")
        running = bool(st_ and st_["status"] == "running")
        if n_eps < 10:
            st.info("Produisez d'abord des démonstrations (étape 4).")
        elif not running and not widgets.busy_warning(T, "train") and st.button("Lancer l'entraînement", type="primary"):
            out_dir = f"outputs/train/act_{T.id}_{time.strftime('%Y%m%d_%H%M')}"
            cfg = write_train_config(T, steps, out_dir)
            core.write_json(T.data / "train_current.json", {"output_dir": out_dir, "steps": int(steps), "episodes": n_eps})
            core.start_job(T.id, "train", "Entraînement ACT", [
                {"name": "Préparation du jeu de données", "cwd": str(core.REPO),
                 "cmd": [str(core.PY_LEARN), "scripts/data/raw_to_lerobot.py", "--raw", str(T.raw), "--out", str(T.dataset),
                         "--repo_id", f"cta/{T.id}_app", "--overwrite"]},
                {"name": "Entraînement ACT", "cwd": str(core.REPO),
                 "cmd": [str(core.PY_LEARN), "scripts/train/train_act.py", "--config", cfg]}])
            st.rerun()
        progress(T)


@st.fragment(run_every="5s")
def progress(T: core.Task) -> None:
    st_ = widgets.job_header(T, "train")
    widgets.rerun_when_finished(T, "train", st_)
    if not st_:
        st.caption("Aucun entraînement lancé depuis l'interface.")
        return
    log = core.job_log(T.id, "train")
    tr = core.parse_training(log)
    cur = core.read_json(T.data / "train_current.json", {})
    total = tr["total"] or cur.get("steps", 1)
    if st_.get("step_name", "").startswith("Préparation") and st_["status"] == "running":
        st.progress(0.0, text="Préparation du jeu de données (encodage des vidéos)…")
    else:
        txt = f"{tr['step']:,} / {total:,} pas".replace(",", " ")
        if tr["remaining"]:
            txt += f" · écoulé {tr['elapsed']} · reste {tr['remaining']}"
        st.progress(min(tr["step"] / max(total, 1), 1.0), text=txt)
    if tr["losses"]:
        xs, ys = zip(*tr["losses"])
        fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines", line={"color": "#4F8EF7"}))
        fig.update_layout(height=260, margin={"l": 10, "r": 10, "t": 30, "b": 10}, title="Erreur d'apprentissage (perte)",
                          paper_bgcolor="#0E1117", plot_bgcolor="#161B26", font={"color": "#E6E9EF"},
                          xaxis={"title": "pas", "gridcolor": "#232a38"}, yaxis={"type": "log", "gridcolor": "#232a38"})
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Perte actuelle {ys[-1]:.3f} (départ {ys[0]:.3f}) : elle doit baisser puis se stabiliser.")
    if st_["status"] == "done":
        st.success(f"Modèle prêt : {cur.get('output_dir', '')}. Passez à l'évaluation (étape 6).")
    widgets.stop_button(T, "train", st_)
    widgets.log_box(log, keep=lambda ln: "step:" in ln or "INFO" in ln or "Error" in ln or "[preflight]" in ln)
