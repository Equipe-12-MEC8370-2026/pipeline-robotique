"""2 · Task: demonstration videos (upload, sync, hands), parts located in the video, TaskSpec, human review."""
from __future__ import annotations

import io
import json
import subprocess
import time

import streamlit as st
from PIL import Image, ImageDraw

import core
from views import widgets

POSTURE = {"lying": "couchée", "on_edge": "sur la tranche", "upright": "debout", "unknown": "inconnue (la vidéo ne le dit pas)"}
MAX_DEMOS = 5
CALIB = core.REPO / "data" / "calib" / "tpv.json"
NEW_SETUP = "Nouvelle installation"


def clips(pair: str | None):
    d = core.REPO / "data" / "clips" / pair if pair else None
    return d if d and (d / "tpv.mp4").exists() and (d / "fpv.mp4").exists() else None


@st.cache_data(show_spinner=False)
def frame(pair: str, t: float, width: int = 960) -> Image.Image:
    raw = subprocess.run([core.ffmpeg(), "-v", "error", "-ss", f"{t:.2f}", "-i", str(core.REPO / "data" / "clips" / pair / "tpv.mp4"),
                          "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True, check=True).stdout
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return img.resize((width, int(img.height * width / img.width)))


def duration(pair: str) -> float:
    d = core.read_json(core.REPO / "data" / "clips" / pair / "sync.json", {})
    return float(d.get("duration_s") or 30.0)


def render(T: core.Task) -> None:
    st.markdown("<div class='cta-step'>Étape 2</div>", unsafe_allow_html=True)
    st.title("Tâche")
    st.caption("La démonstration filmée (vue de la personne FPV + vue de côté TPV), où sont les pièces, puis ce que le "
               "pipeline a compris de la vidéo. Vérifiez puis approuvez.")
    videos(T)
    locate(T)
    instruction(T)
    description(T)


# ---------------------------------------------------------------- videos

def videos(T: core.Task) -> None:
    demos = T.demos()
    for k, d in enumerate(demos):
        with st.container(border=True):
            h1, h2 = st.columns([5, 1])
            h1.subheader(f"Démonstration {k + 1}")
            if h2.button("Retirer", key=f"deldemo_{k}"):
                demos.pop(k)
                T.set_demos(demos)
                st.rerun()
            if k > 0:
                st.markdown(widgets.temp("Limite actuelle", "La fusion de plusieurs démonstrations n'est pas encore codée.")
                            + " <span style='color:#9aa4b8;font-size:.88rem'>Gardée pour plus tard : seule la démonstration 1 "
                              "sert à construire la description pour l'instant.</span>", unsafe_allow_html=True)
            c = clips(d.get("pair"))
            if c:
                a, b = st.columns(2)
                a.markdown("**Vue de la personne (FPV)**")
                a.video(str(c / "fpv.mp4"))
                b.markdown("**Vue de côté (TPV)**")
                b.video(str(c / "tpv.mp4"))
                if (core.REPO / "data" / "perception" / d["pair"] / "hands_fpv.npz").exists():
                    st.markdown("<span class='cta-ok'>●</span> Synchronisée, mains analysées", unsafe_allow_html=True)
                elif k == 0:
                    process_button(T, d)
                continue
            a, b = st.columns(2)
            ups = {}
            for col, view, lab in ((a, "fpv", "Vidéo vue de la personne (FPV)"), (b, "tpv", "Vidéo vue de côté (TPV)")):
                with col:
                    ups[view] = st.file_uploader(lab, type=["mp4", "mov", "mkv", "avi"], key=f"vid_{k}_{view}")
                    if d.get(view):
                        st.caption(f"Déposée : {core.REPO.joinpath(d[view]).name}")
            if ups["fpv"] and ups["tpv"] and st.session_state.get(f"vids_saved_{k}") != (ups["fpv"].file_id, ups["tpv"].file_id):
                st.session_state[f"vids_saved_{k}"] = (ups["fpv"].file_id, ups["tpv"].file_id)
                pair = d.get("pair") or f"pair_app{time.strftime('%Y%m%d%H%M%S')}"
                folder = T.upload_dir / pair
                folder.mkdir(parents=True, exist_ok=True)
                for view, up in ups.items():
                    dst = folder / f"{view}.{up.name.rsplit('.', 1)[-1].lower()}"
                    dst.write_bytes(up.getvalue())
                    d[view] = dst.relative_to(core.REPO).as_posix()
                d["pair"] = pair
                demos[k] = d
                T.set_demos(demos)
                st.rerun()
            if d.get("fpv") and d.get("tpv"):
                if k == 0:
                    process_button(T, d)
            else:
                st.caption("Déposez les deux vidéos d'une même démonstration, filmées en même temps (le son sert à les synchroniser).")
    if len(demos) < MAX_DEMOS and st.button("＋ Ajouter une démonstration"):
        demos.append({})
        T.set_demos(demos)
        st.rerun()
    job_status(T, "videos", keep=lambda ln: ln.startswith("[sync]") or ln.startswith("[hands]") or "Error" in ln)


def process_button(T: core.Task, d: dict) -> None:
    st_ = core.job_state(T.id, "videos")
    if st_ and st_["status"] == "running":
        return
    if not widgets.busy_warning(T, "videos") and st.button("Synchroniser et analyser les mains", type="primary", key=f"proc_{d['pair']}"):
        steps = []
        if d.get("fpv") and d.get("tpv") and not clips(d["pair"]):
            steps.append({"name": "Synchronisation par le son", "cwd": str(core.REPO),
                          "cmd": [str(core.PY_LEARN), "scripts/video/ingest_one.py", "--fpv", str(core.REPO / d["fpv"]),
                                  "--tpv", str(core.REPO / d["tpv"]), "--pair", d["pair"]]})
        steps.append({"name": "Mains (MediaPipe)", "cwd": str(core.REPO),
                      "cmd": [str(core.PY_PERC), "scripts/perception/track_hands.py", "--pairs", d["pair"], "--overlay"]})
        core.start_job(T.id, "videos", "Vidéos", steps)
        st.rerun()


@st.fragment(run_every="3s")
def job_status(T: core.Task, kind: str, keep=None) -> None:
    st_ = widgets.job_header(T, kind)
    widgets.rerun_when_finished(T, kind, st_)
    if st_:
        widgets.stop_button(T, kind, st_)
        widgets.log_box(core.job_log(T.id, kind), keep=keep)


# ---------------------------------------------------------------- parts in the video (SAM 2 prompts)

def objects_cfg_path(pair: str):
    return core.CONFIGS / "perception" / f"objects_{core.pair_suffix(pair)}.yaml"


def locate(T: core.Task) -> None:
    pair = T.pair
    if not pair or not clips(pair):
        return
    parts = list(T.cfg["objects"])
    with st.container(border=True):
        st.subheader("Où sont les pièces dans la vidéo")
        widgets.temp_note("Dans le pipeline final, les pièces seront trouvées automatiquement (CAO + caméra RGB-D).")
        if not parts:
            st.info("Ajoutez d'abord les pièces (étape 1).")
            return
        cfg_path = objects_cfg_path(pair)
        boxes = st.session_state.setdefault(f"boxes_{pair}", {})
        if cfg_path.exists() and not boxes:
            for pid, o in (core.load_yaml(cfg_path).get("objects") or {}).items():
                if o.get("check_start"):
                    boxes[(pid, "start")] = {"t": float(o["check_start"]["t"]), "box": list(o["check_start"]["box"])}
                if o.get("prompts"):
                    boxes[(pid, "end")] = {"t": float(o["prompts"][0]["t"]), "box": list(o["prompts"][0]["box"])}
        st.caption("Pour chaque pièce, encadrez-la en glissant la souris sur l'image : une fois **à sa place de départ**, une "
                   "fois **à sa place d'arrivée**, à des instants où aucune main ne la touche.")
        dur = duration(pair)
        c1, c2 = st.columns(2)
        pid = c1.selectbox("Pièce", parts, key=f"loc_part_{pair}")
        which = c2.radio("Moment", ["start", "end"], horizontal=True, key=f"loc_which_{pair}",
                         format_func=lambda w: "Place de départ" if w == "start" else "Place d'arrivée")
        cur = boxes.get((pid, which))
        default_t = float(cur["t"]) if cur else (0.5 if which == "start" else round(dur - 1.0, 1))
        t = st.slider("Instant (s)", 0.0, round(dur - 0.1, 1), default_t, 0.1, key=f"loc_t_{pair}_{pid}_{which}")
        from streamlit_image_coordinates import streamlit_image_coordinates
        img = frame(pair, t).copy()
        sx = 1920 / img.width
        if cur and abs(cur["t"] - t) < 0.05:
            x0, y0, x1, y1 = (v / sx for v in cur["box"])
            ImageDraw.Draw(img).rectangle([x0, y0, x1, y1], outline=(79, 142, 247), width=3)
        val = streamlit_image_coordinates(img, key=f"img_{pair}_{pid}_{which}_{t}", click_and_drag=True)
        if val and val.get("x1") is not None and abs(val["x2"] - val["x1"]) > 4 and abs(val["y2"] - val["y1"]) > 4:
            k = img.width / (val.get("width") or img.width)
            box = [round(min(val["x1"], val["x2"]) * k * sx), round(min(val["y1"], val["y2"]) * k * sx),
                   round(max(val["x1"], val["x2"]) * k * sx), round(max(val["y1"], val["y2"]) * k * sx)]
            if not cur or cur.get("box") != box or abs(cur["t"] - t) > 0.05:
                boxes[(pid, which)] = {"t": round(t, 1), "box": box}
                st.rerun()
        done = {p: [w for w in ("start", "end") if (p, w) in boxes] for p in parts}
        st.markdown(" · ".join(f"{p} : {'✔' if len(w) == 2 else f'{len(w)}/2'}" for p, w in done.items()))
        if all(len(w) == 2 for w in done.values()) and st.button("Enregistrer les repères", key=f"save_boxes_{pair}"):
            data = {"pair": pair, "view": "tpv", "fps": 15, "height": 540, "chunk": 150, "objects": {}}
            for p in parts:
                s_, e_ = boxes[(p, "start")], boxes[(p, "end")]
                data["objects"][p] = {"prompts": [{"t": e_["t"], "box": e_["box"]}, {"t": s_["t"], "box": s_["box"]}],
                                      "check_start": {"t": s_["t"], "box": s_["box"]}}
            cfg_path.write_text("# Written by the CTA interface: boxes drawn on the TPV view (full-resolution pixels).\n"
                                + json.dumps(data, indent=2), encoding="utf-8")   # JSON is valid YAML
            st.success(f"Repères enregistrés ({cfg_path.name}).")
        setup = camera_setup(pair)
        if not cfg_path.exists():
            return
        if setup == NEW_SETUP:
            st.warning("Caméra non calibrée : la calibration d'une nouvelle installation se fait encore hors de l'interface.")
            return
        run_perception(T, pair, cfg_path)


def camera_setup(pair: str) -> str:
    cal = core.read_json(CALIB, {})
    setups = sorted(cal.get("setups", {}))
    cur = cal.get("pair_to_setup", {}).get(pair)
    opts = [f"Installation {s}" for s in setups] + [NEW_SETUP]
    st.markdown("**Installation de la caméra de côté** " + widgets.temp(), unsafe_allow_html=True)
    choice = st.selectbox("Installation de la caméra", opts, index=opts.index(f"Installation {cur}") if cur else len(opts) - 1,
                          label_visibility="collapsed", key=f"setup_{pair}",
                          help="La caméra de côté a-t-elle filmé depuis une installation déjà calibrée (même position par "
                               "rapport à la table) ? Dans le pipeline final, chaque poste est calibré une fois avec un repère imprimé.")
    if choice != NEW_SETUP and choice != f"Installation {cur}":
        cal.setdefault("pair_to_setup", {})[pair] = choice.split()[-1]
        core.write_json(CALIB, cal)
    return choice


def run_perception(T: core.Task, pair: str, cfg_path) -> None:
    st_ = core.job_state(T.id, "perception")
    running = bool(st_ and st_["status"] == "running")
    if not running and not widgets.busy_warning(T, "perception") and st.button("Suivre les pièces et construire la description",
                                                                              type="primary"):
        objs = core.load_yaml(cfg_path)["objects"]
        steps = [{"name": "Suivi des pièces (SAM 2)", "cwd": str(core.REPO),
                  "cmd": [str(core.PY_PERC), "scripts/perception/track_objects.py", "--config", str(cfg_path)]}]
        for pid, o in T.cfg["objects"].items():
            if pid in objs and (not o.get("size_m") or o.get("method", "video") == "video"):
                times = [str(objs[pid]["check_start"]["t"]), str(objs[pid]["prompts"][0]["t"])]
                steps.append({"name": f"Mesure de {pid} sur la vidéo", "cwd": str(core.REPO),
                              "cmd": [str(core.PY_LEARN), "scripts/perception/measure_part.py", "--objects_cfg", str(cfg_path),
                                      "--object", pid, "--times", *times]})
        steps += [{"name": "Tailles mesurées", "cwd": str(core.REPO),
                   "cmd": [str(core.PY_APP), "app/apply_measures.py", str(T.cfg_path), pair]},
                  {"name": "Poses des pièces", "cwd": str(core.REPO),
                   "cmd": [str(core.PY_LEARN), "scripts/perception/object_trajectories.py", "--objects_cfg", str(cfg_path),
                           "--task_cfg", str(T.cfg_path)]},
                  {"name": "Description de la tâche", "cwd": str(core.REPO),
                   "cmd": [str(core.PY_LEARN), "scripts/task/build_taskspec.py", "--task_cfg", str(T.cfg_path)]}]
        core.start_job(T.id, "perception", "Perception et description", steps)
        st.rerun()
    job_status(T, "perception", keep=lambda ln: ln.startswith("[") or "Error" in ln)


# ---------------------------------------------------------------- instruction and description

def instruction(T: core.Task) -> None:
    with st.container(border=True):
        st.subheader("Consigne")
        ins = T.cfg.get("instruction", {})
        fr = st.text_input("Français", value=str(ins.get("fr", "")))
        en = st.text_input("Anglais (donnée au modèle)", value=str(ins.get("en", "")))
        if st.button("Enregistrer la consigne"):
            T.cfg["instruction"]["fr"], T.cfg["instruction"]["en"] = fr, en
            T.save()
            st.success("Consigne enregistrée. Reconstruisez la description pour l'y inclure.")


def description(T: core.Task) -> None:
    with st.container(border=True):
        st.subheader("Description de la tâche (lue dans la vidéo)")
        spec = core.read_json(T.taskspec)
        if not spec:
            st.info("Pas encore de description : déposez la vidéo, repérez les pièces, puis lancez le suivi.")
            return
        st.markdown(f"**Consigne :** {spec['instruction'].get('fr', '')}")
        rows = []
        for s in spec["steps"]:
            g = s["goal"]
            rel = "sur la table" if g["relation"]["type"] == "on_table" else f"sur {g['relation']['ref']}"
            rows.append({"Étape": s["index"] + 1, "Pièce": s["object"], "Arrivée": rel,
                         "Posture": POSTURE.get(g["pose"]["posture"], g["pose"]["posture"]),
                         "Prise → dépose (s)": f"{s['t_grasp']} → {s['t_release']}", "Main": s["hand"]})
        st.dataframe(rows, hide_index=True, use_container_width=True)
        sizes = {o["id"]: " × ".join(f"{v * 1000:.0f}" for v in o["size_m"]) + " mm" for o in spec["objects"]}
        st.caption("Pièces : " + " · ".join(f"{k} {v}" for k, v in sizes.items()) + f" · fichier {T.taskspec.name}")
        img = core.REPO / "outputs" / "p2_4" / f"poses_{T.pair}.jpg"
        if img.exists():
            st.image(str(img), caption="Pièces repérées au début et à la fin de la vidéo (boîtes ajustées)")
        if (core.REPO / "data" / "perception" / T.pair / "objects.json").exists():
            rebuild(T)
        st.divider()
        widgets.approval(T, "taskspec", "J'approuve cette description")


@st.fragment(run_every="2s")
def rebuild(T: core.Task) -> None:
    st_ = widgets.job_header(T, "taskspec")
    widgets.rerun_when_finished(T, "taskspec", st_)
    running = bool(st_ and st_["status"] == "running")
    if not running and not widgets.busy_warning(T, "taskspec") and st.button("Reconstruire la description",
                                                                             help="Après un changement de pièce ou de consigne"):
        core.start_job(T.id, "taskspec", "Description de la tâche", [
            {"name": "Construction de la TaskSpec", "cmd": [str(core.PY_LEARN), "scripts/task/build_taskspec.py",
                                                             "--task_cfg", str(T.cfg_path)], "cwd": str(core.REPO)}])
        st.rerun()
    if st_:
        if st_["status"] == "done" and time.time() - st_.get("ended", 0) < 120:
            st.success("Description reconstruite.")
        widgets.log_box(core.job_log(T.id, "taskspec"), keep=lambda ln: ln.startswith("[spec]"))
