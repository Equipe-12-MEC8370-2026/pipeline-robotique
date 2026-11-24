"""6 · Evaluation: the trained model plays 50 trials in simulation (progress, success rate, videos)."""
from __future__ import annotations

import time

import streamlit as st

import core
from views import widgets

OUTCOME = {"success": "réussi", "not_released": "pièce non lâchée", "not_at_rest": "pièces pas immobiles"}


SECONDS_PER_TRIAL = 45   # measured 2026-11-18: ~53 s for a failed trial (time limit), ~17 s for a success


def checkpoints(T: core.Task) -> list:
    root = core.REPO / "outputs" / "train"
    out = [p for p in root.glob(f"act_{T.id}_*/checkpoints/last/pretrained_model") if p.exists()]
    return sorted(out, key=lambda p: p.stat().st_mtime, reverse=True)


def outcome_fr(o: str) -> str:
    if o in OUTCOME:
        return OUTCOME[o]
    for suf, txt in (("_dropped", "tombée de la table"), ("_never_lifted", "jamais soulevée"), ("_in_zone", "hors de la zone"),
                     ("_posture", "mauvaise posture"), ("_on_table", "pas posée à plat"), ("_at_offset", "mal placée sur l'autre pièce")):
        if o.endswith(suf):
            return f"{o[: -len(suf)]} : {txt}"
    return o


def render(T: core.Task) -> None:
    st.markdown("<div class='cta-step'>Étape 6</div>", unsafe_allow_html=True)
    st.title("Évaluation")
    st.caption("Le modèle entraîné fait la tâche seul, avec seulement ses caméras, à partir de positions de départ qu'il n'a "
               "pas vues. Les vidéos montrent la caméra avant et la caméra poignet.")
    cks = checkpoints(T)
    with st.container(border=True):
        if not cks:
            st.info("Aucun modèle entraîné pour cette tâche (étape 5).")
        else:
            ck = st.selectbox("Modèle", cks, format_func=lambda p: p.parents[2].name)
            n = st.select_slider("Nombre d'essais", options=[5, 20, 50], value=50,
                                 help="Indépendant du nombre de démonstrations. 5 = vérification rapide ; 50 = mesure fiable "
                                      "(c'est le protocole du projet).")
            if n == 50:
                st.caption("50 essais : 40 dans la zone de départ de l'entraînement, 10 un peu en dehors (la pièce "
                           "commence plus loin ou plus tournée), pour voir si le modèle généralise.")
            else:
                st.caption(f"{n} essais, tous dans la zone de départ de l'entraînement. Pour une mesure fiable et le test "
                           "hors zone, choisissez 50.")
            st.caption(f"Temps estimé : {widgets.fmt_dur(90 + n * SECONDS_PER_TRIAL)} (au plus ~55 s par essai ; un essai "
                       "réussi s'arrête plus tôt).")
            st_ = core.job_state(T.id, "eval")
            if not (st_ and st_["status"] == "running") and not widgets.busy_warning(T, "eval"):
                if st.button("Lancer l'évaluation", type="primary"):
                    tag = f"{ck.parents[2].name}_{time.strftime('%H%M')}"
                    core.write_json(T.data / "eval_current.json", {"tag": tag, "model": ck.parents[2].name})
                    n_train, n_ood = (40, 10) if n == 50 else (n, 0)
                    core.start_job(T.id, "eval", "Évaluation", [
                        {"name": "Serveur du modèle", "cwd": str(core.REPO), "background": True, "ready": "[server] READY",
                         "cmd": [str(core.PY_LEARN), "scripts/eval/policy_server.py", "--ckpt", str(ck)]},
                        {"name": "Essais en simulation (Isaac Sim)", "cwd": str(core.REPO), "env": T.env(),
                         "cmd": [str(core.PY_SIM), "scripts/sim/eval_taskspec_policy.py", "--headless", "--enable_cameras",
                                 "--agent", "policy", "--tag", tag, "--n_train", str(n_train), "--n_ood", str(n_ood),
                                 "--shutdown_server"]}])
                    st.rerun()
        progress(T)
    results(T)


@st.fragment(run_every="3s")
def progress(T: core.Task) -> None:
    st_ = widgets.job_header(T, "eval")
    widgets.rerun_when_finished(T, "eval", st_)
    if not st_:
        return
    log = core.job_log(T.id, "eval")
    ev = core.parse_eval(log)
    if ev["total"]:
        st.progress(ev["done"] / ev["total"], text=f"{ev['done']} / {ev['total']} essais · {ev['successes']} réussis")
    elif st_["status"] == "running":
        st.caption("Chargement du modèle et démarrage d'Isaac Sim…")
    widgets.stop_button(T, "eval", st_)
    widgets.log_box(log, keep=lambda ln: ln.startswith("[") or "Error" in ln)


def results(T: core.Task) -> None:
    runs = sorted((p.parent for p in T.eval_dir.glob("*/report.json")), key=lambda p: p.stat().st_mtime, reverse=True) \
        if T.eval_dir.exists() else []
    if not runs:
        return
    with st.container(border=True):
        st.subheader("Résultats")
        def label(p):
            ag = core.read_json(p / "report.json", {}).get("info", {}).get("agent", "")
            return p.name + ("  (référence : gestes programmés)" if ag.startswith("expert") else "")
        run = st.selectbox("Évaluation", runs, format_func=label)
        rep = core.read_json(run / "report.json", {})
        recs = rep.get("records", [])
        n_ok = sum(r["outcome"] == "success" for r in recs)
        tr = [r for r in recs if r["split"] == "train_range"]
        od = [r for r in recs if r["split"] != "train_range"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Réussite", f"{n_ok} / {len(recs)}  ({n_ok / max(len(recs), 1):.0%})")
        c2.metric("Dans la zone d'entraînement", f"{sum(r['outcome'] == 'success' for r in tr)} / {len(tr)}")
        c3.metric("Hors zone", f"{sum(r['outcome'] == 'success' for r in od)} / {len(od)}" if od else "—")
        fails: dict[str, int] = {}
        for r in recs:
            if r["outcome"] != "success":
                fails[outcome_fr(r["outcome"])] = fails.get(outcome_fr(r["outcome"]), 0) + 1
        if fails:
            st.markdown("**Échecs** : " + " · ".join(f"{k} ({v})" for k, v in sorted(fails.items(), key=lambda kv: -kv[1])))
        vids = sorted((run / "videos").glob("trial_*.mp4"))
        if vids:
            by = {int(v.stem.split("_")[1]): v for v in vids}
            lab = {r["trial"]: f"essai {r['trial']} · {outcome_fr(r['outcome'])}" for r in recs}
            k = st.selectbox("Vidéo", sorted(by), format_func=lambda t: lab.get(t, f"essai {t}"))
            st.video(str(by[k]))
