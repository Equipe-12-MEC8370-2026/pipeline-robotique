"""1 · Parts (up to 5): size and mass from a CAD file, a phone 3D scan, typed dimensions or the video; robot limits."""
from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import yaml

import core
from views import widgets

MAX_PARTS = 5
MATERIALS = {  # density, kg/m3
    "Aluminium": 2700, "Acier": 7850, "Acier inoxydable": 8000, "Laiton": 8500, "Cuivre": 8960,
    "Plastique ABS": 1050, "Plastique POM (Delrin)": 1410, "Nylon (PA)": 1140, "Bois": 600,
}
UNITS = {"mm": 1e-3, "cm": 1e-2, "m": 1.0, "po (inch)": 0.0254}
METHODS = {  # code: (label, file types or None)
    "cad": ("CAO (recommandé · le plus précis)", ["step", "stp", "stl", "obj", "ply", "glb"]),
    "scan": ("Scan 3D au téléphone (photogrammétrie)", ["obj", "glb", "ply", "stl"]),
    "manual": ("Saisie manuelle des dimensions", None),
    "video": ("Mesure sur la vidéo", None),
}
HELP = {
    "cad": "Le fichier de conception de la pièce : forme exacte, dimensions et volume exacts.",
    "scan": "Une application de scan 3D au téléphone (photogrammétrie) exporte un fichier OBJ ou GLB. "
            "Précis à quelques millimètres ; vérifiez les unités (souvent en mètres).",
    "manual": "Longueur, largeur, hauteur au pied à coulisse : la pièce est vue comme une boîte.",
    "video": "Mesurée automatiquement sur la vidéo de démonstration : précise à quelques millimètres, mais la face "
             "posée sur la table peut rester incertaine.",
}


def limits() -> dict:
    return yaml.safe_load((core.REPO / "app" / "robot_limits.yaml").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_mesh(path: str, mtime: float, unit: float):
    import trimesh
    m = trimesh.load(path, force="mesh")
    m.apply_scale(unit)
    try:
        obb = m.bounding_box_oriented.extents   # smallest box around the part, whatever its axes in the file
    except Exception:  # noqa: BLE001  (degenerate hull): box along the file axes
        obb = m.extents
    return {"vertices": np.asarray(m.vertices), "faces": np.asarray(m.faces),
            "extents_m": sorted((float(v) for v in obb), reverse=True),
            "volume_m3": float(abs(m.volume)) if m.is_watertight else None, "watertight": bool(m.is_watertight)}


def mesh_figure(mesh: dict) -> go.Figure:
    v, f = mesh["vertices"], mesh["faces"]
    fig = go.Figure(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2], color="#9aa7bd",
                              flatshading=True, lighting={"ambient": 0.45, "diffuse": 0.8, "specular": 0.3}))
    ax = {"visible": False}
    fig.update_layout(scene={"xaxis": ax, "yaxis": ax, "zaxis": ax, "aspectmode": "data", "bgcolor": "#0E1117"},
                      paper_bgcolor="#0E1117", margin={"l": 0, "r": 0, "t": 0, "b": 0}, height=300)
    return fig


def box_figure(size) -> go.Figure:
    a, b, c = size
    x = [0, a, a, 0, 0, a, a, 0]
    y = [0, 0, b, b, 0, 0, b, b]
    z = [0, 0, 0, 0, c, c, c, c]
    i = [0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3]
    j = [1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 0, 4]
    k = [2, 3, 6, 7, 5, 4, 6, 5, 7, 6, 4, 7]
    return mesh_figure({"vertices": np.c_[x, y, z], "faces": np.c_[i, j, k]})


def limit_check(size, mass_kg) -> None:
    lim = limits()
    width = sorted(size)[0]   # narrowest side: what the gripper closes on (it can choose the narrowest face)
    msgs = []
    if width > lim["gripper_max_width_m"]:
        msgs.append(f"plus petit côté {width * 1000:.0f} mm > ouverture de la pince {lim['gripper_max_width_m'] * 1000:.0f} mm")
    if lim.get("payload_kg") and mass_kg > lim["payload_kg"]:
        msgs.append(f"masse {mass_kg * 1000:.0f} g > charge max {lim['payload_kg'] * 1000:.0f} g")
    if msgs:
        st.markdown(f"<span class='cta-bad'>●</span> **Trop grande pour le robot {lim['robot']}** : " + " ; ".join(msgs) + ". "
                    + widgets.temp() + " En simulation, la pièce sera réduite et allégée (étape 3).", unsafe_allow_html=True)
    else:
        st.markdown(f"<span class='cta-ok'>●</span> Compatible avec le robot {lim['robot']}"
                    + ("" if lim.get("payload_kg") else " (charge max du robot non renseignée : masse non vérifiée)"),
                    unsafe_allow_html=True)


def video_measure(T: core.Task, pid: str):
    """(size, path) of the last measurement of this part on the task's video, else (None, None)."""
    if not T.pair:
        return None, None
    p = core.REPO / "outputs" / "p2_4" / f"size_{T.pair}_{pid}.json"
    d = core.read_json(p)
    return (d["size_m_longest_first"], p) if d else (None, None)


def part_block(T: core.Task, pid: str) -> None:
    o = T.cfg["objects"][pid]
    with st.container(border=True):
        h1, h2 = st.columns([5, 1])
        h1.subheader(pid.replace("_", " ").capitalize())
        if h2.button("Retirer", key=f"del_{pid}", help="Retirer cette pièce de la tâche"):
            del T.cfg["objects"][pid]
            T.save()
            st.rerun()
        o["description"] = st.text_input("Description", value=str(o.get("description", "")), key=f"desc_{pid}")
        codes = list(METHODS)
        cur = o.get("method") or ("cad" if o.get("cad") else "video" if o.get("size_m") else "cad")
        method = st.radio("Comment donner la forme de la pièce ?", codes, index=codes.index(cur), key=f"meth_{pid}",
                          format_func=lambda c: METHODS[c][0], horizontal=True)
        st.caption(HELP[method])
        if method == "video":
            st.markdown(widgets.temp() + " <span style='color:#9aa4b8;font-size:.88rem'>Remplacée par la CAO ou la "
                        "caméra RGB-D dans le pipeline final.</span>", unsafe_allow_html=True)
        left, right = st.columns([1, 1.2])
        size, volume, fig, sig = None, None, None, method
        with left:
            if METHODS[method][1]:
                up = st.file_uploader("Fichier " + ", ".join(e.upper() for e in METHODS[method][1]), type=METHODS[method][1],
                                      key=f"file_{pid}_{method}")
                if up is not None and st.session_state.get(f"saved_{pid}") != up.file_id:
                    st.session_state[f"saved_{pid}"] = up.file_id
                    T.cad_dir.mkdir(parents=True, exist_ok=True)
                    for old in T.cad_dir.glob(f"{pid}.*"):
                        old.unlink()
                    dst = T.cad_dir / f"{pid}.{up.name.rsplit('.', 1)[-1].lower()}"
                    dst.write_bytes(up.getvalue())
                    o["cad"] = dst.relative_to(core.REPO).as_posix()
                    T.save()
                default_unit = o.get("cad_units", "mm" if method == "cad" else "m")
                unit = st.selectbox("Unités du fichier", list(UNITS), key=f"unit_{pid}_{method}",
                                    index=list(UNITS).index(default_unit) if default_unit in UNITS else 0)
                cad = core.REPO / o["cad"] if o.get("cad") else None
                if cad and cad.exists():
                    try:
                        mesh = load_mesh(str(cad), cad.stat().st_mtime, UNITS[unit])
                        size, volume, fig = mesh["extents_m"], mesh["volume_m3"], mesh_figure(mesh)
                        sig = f"{method}-{cad.stat().st_mtime}-{unit}"
                        if not mesh["watertight"]:
                            st.warning("Maillage non fermé : volume inconnu, saisissez la masse.")
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Lecture impossible : {exc}")
            elif method == "manual":
                base = [min(2000.0, float(v) * 1000) for v in (o.get("size_m") or [0.03, 0.03, 0.03])][:3]
                c1, c2, c3 = st.columns(3)
                dims = [c1.number_input("Longueur (mm)", 1.0, 2000.0, base[0], 1.0, key=f"L_{pid}"),
                        c2.number_input("Largeur (mm)", 1.0, 2000.0, base[1], 1.0, key=f"W_{pid}"),
                        c3.number_input("Hauteur (mm)", 1.0, 2000.0, base[2], 1.0, key=f"H_{pid}")]
                size = sorted((d / 1000 for d in dims), reverse=True)
                volume, fig, sig = math.prod(size), box_figure(size), f"manual-{size}"
            else:  # video
                vs, vp = video_measure(T, pid)
                if vs:
                    size, volume, fig, sig = vs, math.prod(vs), box_figure(vs), f"video-{vp.stat().st_mtime}"
                elif o.get("size_m") and o.get("method", "video") == "video":
                    size = sorted((float(v) for v in o["size_m"]), reverse=True)
                    volume, fig, sig = math.prod(size), box_figure(size), f"video-cfg-{size}"
                else:
                    st.info("Pas encore de mesure : elle se fait à l'étape 2, une fois les pièces repérées dans la vidéo.")
            mats = list(MATERIALS)
            mat = st.selectbox("Matériau", mats, key=f"mat_{pid}", index=mats.index(o["material"]) if o.get("material") in mats else 0)
        with right:
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True, key=f"fig_{pid}_{sig}")
            else:
                st.markdown("<div class='cta-card' style='height:280px;display:flex;align-items:center;justify-content:center;"
                            "color:#7f8aa3'>Aperçu 3D</div>", unsafe_allow_html=True)
        if size is None:
            return
        rho = MATERIALS[mat]
        auto = volume * rho if volume else None
        box_note = method in ("manual", "video")
        st.markdown(f"**Dimensions** : {' × '.join(f'{v * 1000:.1f}' for v in size)} mm")
        default_g = round((auto if auto else float(o.get("real_mass_kg", 0.1))) * 1000, 1)
        # the key follows the shape, the file and the material: the computed mass refreshes when any of them changes
        mass_g = st.number_input("Masse réelle (g)", min_value=0.1, value=default_g, step=1.0, key=f"mass_{pid}_{sig}_{mat}",
                                 help="Calculée depuis le volume et la densité du matériau ; modifiable.")
        if auto:
            st.caption(f"Volume {volume * 1e6:.1f} cm³ × {rho} kg/m³ = {auto * 1000:.1f} g"
                       + (" (boîte pleine : surestime une pièce creuse ou usinée)" if box_note else ""))
        limit_check(size, mass_g / 1000)
        if st.button("Enregistrer la pièce", key=f"save_{pid}", type="primary"):
            o["method"], o["material"], o["real_mass_kg"] = method, mat, round(mass_g / 1000, 4)
            o["size_m"] = [round(v, 4) for v in size]
            if METHODS[method][1]:
                o["cad_units"] = unit
            T.save()
            st.success("Pièce enregistrée. Reconstruisez la description de la tâche (étape 2) pour en tenir compte.")


def render(T: core.Task) -> None:
    st.markdown("<div class='cta-step'>Étape 1</div>", unsafe_allow_html=True)
    st.title("Pièces")
    st.caption("Les pièces manipulées dans la tâche (jusqu'à 5). Leur forme donne la taille et la masse ; la page vérifie "
               "que le robot peut les prendre.")
    for pid in list(T.cfg["objects"].keys()):
        part_block(T, pid)
    n = len(T.cfg["objects"])
    if n < MAX_PARTS:
        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            name = c1.text_input("Nom de la nouvelle pièce", key="new_part_name", placeholder="Ex. : bloc, profilé court…",
                                 label_visibility="collapsed")
            if c2.button("＋ Ajouter une pièce", use_container_width=True, disabled=not name.strip()):
                pid = core.slug(name)
                if pid in T.cfg["objects"]:
                    st.error("Une pièce porte déjà ce nom.")
                else:
                    T.cfg["objects"][pid] = {"description": name.strip(), "shape": "box", "mass_kg": 0.02}
                    T.save()
                    st.rerun()
    else:
        st.caption(f"Maximum de {MAX_PARTS} pièces atteint.")
