"""Small shared widgets: job status, approval box."""
from __future__ import annotations

import time

import streamlit as st

import core

STATUS = {"running": ("En cours", "cta-warn"), "done": ("Terminé", "cta-ok"), "failed": ("Échec", "cta-bad"),
          "stopped": ("Arrêté", "cta-bad"), "starting": ("Démarrage", "cta-warn")}


def job_header(T: core.Task, kind: str) -> dict | None:
    st_ = core.job_state(T.id, kind)
    if not st_:
        return None
    label, cls = STATUS.get(st_["status"], (st_["status"], ""))
    dur = (st_.get("ended") or time.time()) - st_.get("started", time.time())
    step = st_.get("step_name") or st_.get("title", "")
    st.markdown(f"<span class='{cls}'>●</span> **{label}** · {step} · {int(dur // 3600)} h {int(dur % 3600 // 60):02d} min "
                f"{int(dur % 60):02d} s", unsafe_allow_html=True)
    if st_["status"] == "failed":
        st.error(f"L'étape « {st_.get('failed_step', step)} » a échoué. Voir le journal ci-dessous.")
    return st_


def stop_button(T: core.Task, kind: str, st_: dict | None) -> None:
    if st_ and st_["status"] == "running":
        if st.button("Arrêter", key=f"stop_{kind}"):
            core.stop_job(T.id, kind)
            st.rerun()


def busy_warning(T: core.Task | None = None, kind: str | None = None) -> bool:
    """True (and a warning) when another computation already uses the GPU (this page's own job excluded:
    its status is shown by the page)."""
    run = [(t, k) for t, k in core.running_jobs() if not (T and t == T.id and k == kind)]
    if run:
        st.warning("Un calcul est déjà en cours (" + ", ".join(f"{k} · {t}" for t, k in run) + "). "
                   "Attendez sa fin : la carte graphique ne peut pas en faire deux à la fois.")
        return True
    return False


def approval(T: core.Task, key: str, label: str) -> None:
    a = T.approvals().get(key)
    if a and a.get("approved"):
        st.markdown(f"<span class='cta-ok'>✔</span> Approuvé le {a['at']}", unsafe_allow_html=True)
        if st.button("Retirer l'approbation", key=f"unapprove_{key}"):
            T.set_approval(key, False)
            st.rerun()
    else:
        if st.button(label, key=f"approve_{key}", type="primary"):
            T.set_approval(key, True)
            st.rerun()


def log_box(log: str, keep=None, n: int = 14) -> None:
    with st.expander("Journal (dernières lignes)"):
        st.code(core.tail_lines(log, n, keep) or "—", language=None)


def rerun_when_finished(T: core.Task, kind: str, st_: dict | None) -> None:
    """Inside a refreshing fragment: re-run the whole page once when the job ends (its results then show)."""
    key = f"seen_end_{T.id}_{kind}"
    if st_ and st_["status"] == "running":
        st.session_state[key] = None          # watched while running: re-run once it ends
    elif st_ and st_.get("ended"):
        if key in st.session_state and st.session_state[key] != st_["ended"]:
            st.session_state[key] = st_["ended"]
            st.rerun(scope="app")
        st.session_state[key] = st_["ended"]


TEMP_HELP = "Temporaire : présent dans cette version, absent du pipeline final."


def temp(text: str = "Temporaire", help: str = TEMP_HELP) -> str:
    """Inline orange tag (HTML) marking what exists only in this version of the pipeline."""
    return (f"<span title=\"{help}\" style='background:#3d2e0f;color:#e3b341;border:1px solid #6b5418;"
            f"border-radius:6px;padding:1px 7px;font-size:.72rem;font-weight:600;vertical-align:middle'>{text}</span>")


def temp_note(text: str) -> None:
    st.markdown(f"{temp()} <span style='color:#9aa4b8;font-size:.88rem'>{text}</span>", unsafe_allow_html=True)


def fmt_dur(s: float) -> str:
    s = max(0, int(s))
    h, m = s // 3600, (s % 3600) // 60
    return f"{h} h {m:02d} min" if h else f"{max(m, 1)} min"
