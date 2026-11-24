"""Place the wrist camera mount (isaac/Pièces/CameraMount_v2.stl) on the SO-101 gripper link, from geometry only.

Run with isaacsim-env (pxr + trimesh):
    ..\\isaacsim-env\\Scripts\\python.exe scripts\\sim\\fit_wrist_mount.py

How (2026-11-24, no photo of the mounted part available):
  1. the gripper link of the robot USD (meshes ``wrist_roll_follower_so101_v1`` + ``sts3215_03a_v1``) is exported in
     the link frame;
  2. M3 holes are searched on ``wrist_roll_follower`` (small circular loops in plane sections): exactly ONE pair has
     the 8.1 mm spacing of the mount tab (Thingiverse thing 7390547: "M3 nuts added to the arm", 2 M3 bolts);
  3. the mount tab holes are found the same way; the tab face is laid on the outer wall face, holes on holes;
  4. of the 4 candidates (2 tab faces x 2 hole orders) we keep the ones that do not enter the gripper, then the one
     whose camera looks toward the fingertips (-Z of the gripper link), as on the photos of the mount's author.
Output: ``assets/props/camera_mount_v2_fit.json`` (4x4 transform mount(m) -> gripper link, hole centres, checks).
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import trimesh
from pxr import Usd, UsdGeom

REPO = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--usd", default=str(REPO / "assets/robots/so101/so101_follower.usd"))
ap.add_argument("--stl", default=str(REPO.parent / "Pièces" / "CameraMount_v2.stl"))
ap.add_argument("--out", default=str(REPO / "assets/props/camera_mount_v2_fit.json"))
args = ap.parse_args()

M3_SPACING = 0.0081  # mount tab, CameraMount_v2_cadquery.py / HANDOFF


def gripper_meshes(usd_path: str) -> dict[str, trimesh.Trimesh]:
    st = Usd.Stage.Open(usd_path)
    cache = UsdGeom.XformCache()
    link = st.GetPrimAtPath("/so101_new_calib/gripper")
    link_w = np.array(cache.GetLocalToWorldTransform(link)).T
    out = {}
    for p in Usd.PrimRange(link, Usd.TraverseInstanceProxies()):
        if p.GetTypeName() == "Mesh" and "/visuals/" in str(p.GetPath()):
            m = UsdGeom.Mesh(p)
            pts = np.array(m.GetPointsAttr().Get(), dtype=float)
            faces = np.array(m.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
            rel = np.linalg.inv(link_w) @ np.array(cache.GetLocalToWorldTransform(p)).T
            out[p.GetParent().GetName()] = trimesh.Trimesh((rel[:3, :3] @ pts.T).T + rel[:3, 3], faces, process=True)
    return out


def circle_loops(mesh, normal, origins, r_range=(0.0012, 0.0022)):
    found = []
    for o in origins:
        s = mesh.section(plane_origin=o, plane_normal=normal)
        if s is None:
            continue
        for e in s.discrete:
            c = e.mean(0)
            r = np.linalg.norm(e - c, axis=1)
            if r_range[0] < r.mean() < r_range[1] and r.std() < 2e-4:
                found.append(np.r_[c, r.mean()])
    return np.array(found)


def cluster_axis(loops, axis, tol=5e-4):
    groups = []
    for h in loops:
        for g in groups:
            if np.linalg.norm(np.cross(g[0][:3] - h[:3], axis)) < tol:
                g.append(h)
                break
        else:
            groups.append([h])
    return [np.mean(g, 0) for g in groups]


meshes = gripper_meshes(args.usd)
grip, servo = meshes["wrist_roll_follower_so101_v1"], meshes["sts3215_03a_v1"]

# --- 2. holes on the gripper body (search along the 3 link axes)
pairs = []
for ax in range(3):
    n = np.eye(3)[ax]
    lo, hi = grip.bounds[0][ax], grip.bounds[1][ax]
    holes = cluster_axis(circle_loops(grip, n, [n * t for t in np.arange(lo + 5e-4, hi, 1e-3)]), n)
    for a, b in itertools.combinations(holes, 2):
        d = np.linalg.norm(np.delete(a[:3] - b[:3], ax))
        if abs(d - M3_SPACING) < 4e-4:
            pairs.append((ax, a, b, d))
assert len(pairs) == 1, f"expected one M3 pair on the gripper, found {len(pairs)}"
ax, ha, hb, gap = pairs[0]
wall_dir = np.eye(3)[ax]
# outer wall face: farthest hit of a ray coming from outside along -axis, 3 mm beside the first hole (outside the pair)
u = (ha[:3] - hb[:3]) / np.linalg.norm(ha[:3] - hb[:3])
locs, _, _ = grip.ray.intersects_location([ha[:3] + u * 0.003 + wall_dir * 0.05], [-wall_dir])
assert len(locs), "no wall found beside the M3 holes"
wall = float(np.max(locs @ wall_dir))

# --- 3. tab of the mount: its two large faces tilted from the plate are the tab faces
mount = trimesh.load(args.stl)
mount.apply_scale(0.001)
fa = mount.facets_area
tab_faces = [i for i in np.argsort(-fa)[:12] if 0.2 < abs(mount.facets_normal[i][2]) < 0.8]
assert len(tab_faces) >= 2
n1 = mount.facets_normal[tab_faces[0]]
c1 = mount.vertices[np.unique(mount.faces[mount.facets[tab_faces[0]]])].mean(0)
opp = [i for i in tab_faces if np.dot(mount.facets_normal[i], n1) < -0.99][0]
c2 = mount.vertices[np.unique(mount.faces[mount.facets[opp]])].mean(0)
thick = float(abs(np.dot(c1 - c2, n1)))
tab_holes = cluster_axis(circle_loops(mount, n1, [c2 + n1 * t for t in np.linspace(5e-4, thick - 5e-4, 5)]), n1)
assert len(tab_holes) == 2, f"expected 2 tab holes, found {len(tab_holes)}"


def frame(p0, p1, nz):
    x = p1 - p0
    x -= np.dot(x, nz) * nz
    x /= np.linalg.norm(x)
    return np.c_[x, np.cross(nz, x), nz]


def onplane(p, c, n):
    return p - np.dot(p - c, n) * n


cands = []
for face_c, face_out in ((c1, n1), (c2, -n1)):
    A = [onplane(h[:3], face_c, face_out) for h in tab_holes]
    for swap in (False, True):
        a0, a1 = (A[1], A[0]) if swap else (A[0], A[1])
        R = frame(ha[:3], hb[:3], -wall_dir) @ frame(a0, a1, face_out).T
        T = np.eye(4)
        T[:3, :3] = R
        tgt = (ha[:3] + hb[:3]) / 2
        tgt = tgt - np.dot(tgt, wall_dir) * wall_dir + wall * wall_dir
        T[:3, 3] = tgt - R @ ((a0 + a1) / 2)
        mm = mount.copy()
        mm.apply_transform(T)
        pts = mm.vertices[np.random.default_rng(0).choice(len(mm.vertices), 3000, replace=False)]
        depth = 0.0
        for g in (grip, servo):
            ins = g.contains(pts)
            if ins.any():
                depth = max(depth, float(g.nearest.on_surface(pts[ins])[1].max()))
        axis = R @ np.array([0.0, 0.0, 1.0])  # lens looks out of the counterbored face (+Z of the STL)
        cands.append(dict(face_sign=1 if face_out is n1 else -1, swap=swap, max_depth_m=depth,
                          cam_axis_link=axis.tolist(), T=T.tolist(), bounds_link=mm.bounds.tolist()))

ok = [c for c in cands if c["max_depth_m"] < 5e-4]
assert ok, "every candidate enters the gripper"
best = min(ok, key=lambda c: c["cam_axis_link"][2])  # looks most toward the fingertips (-Z)
res = dict(
    source_stl=str(Path(args.stl).name), robot_usd=str(Path(args.usd).name), link="gripper",
    T_mount_to_link=best["T"], gripper_holes_link=[ha[:3].tolist(), hb[:3].tolist()], gripper_hole_dia_m=[2 * ha[3], 2 * hb[3]],
    gripper_hole_spacing_m=gap, wall_axis="xyz"[ax], wall_outer_face_m=wall,
    tab_holes_mount=[h[:3].tolist() for h in tab_holes], tab_hole_dia_m=[2 * h[3] for h in tab_holes], tab_thickness_m=thick,
    chosen=dict(face_sign=best["face_sign"], swap=best["swap"], max_depth_m=best["max_depth_m"],
                cam_axis_link=best["cam_axis_link"], bounds_link=best["bounds_link"]),
    candidates=[{k: v for k, v in c.items() if k != "T"} for c in cands],
)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if k != "candidates"}, indent=1))
