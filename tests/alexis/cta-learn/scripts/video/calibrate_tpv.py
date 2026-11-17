"""P2.3 — metric calibration of the TPV camera on the table plane (both camera set-ups).

    .venv-learn\\Scripts\\python.exe scripts\\video\\calibrate_tpv.py
Reads configs/calib/tpv_annotations.yaml and the synced clips; writes data/calib/tpv.json (per set-up:
K, image<->table homographies, checks) and outputs/p2_3/grid_<setup>.jpg (10 cm table grid drawn on the view).
Focal: scanned; cost = line-direction residuals + equal spacing of the bolts of the two identical plates.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import yaml

from cta_learn import CONFIGS_DIR, REPO_ROOT
from cta_learn.video.calib import (apply_h, fit_orientation, img_to_plane_fn, intrinsics, plane_homography,
                                   plane_pose, refine_line, vanishing_point)
from cta_learn.video.ingest import ffbin

OUT = REPO_ROOT / "outputs" / "p2_3"
W, H = 1920, 1080


def frame(pair: str, t: float) -> np.ndarray:
    raw = subprocess.run([ffbin("ffmpeg"), "-v", "error", "-ss", str(t), "-i",
                          str(REPO_ROOT / "data" / "clips" / pair / "tpv.mp4"), "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(H, W, 3).copy()


def calibrate_reference(img: np.ndarray, cfg: dict, search: dict) -> dict:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lines, named = {"x": [], "y": [], "z": []}, {}
    for name, s in cfg["segments"].items():
        l, rms = refine_line(g, s["p0"], s["p1"], band=s["band"])
        m = (np.array(s["p0"], float) + np.array(s["p1"], float)) / 2
        m = m - (l[:2] @ m + l[2]) * l[:2]
        lines[name[0]].append((l, m))
        named[name] = {"line": l, "rms_px": rms}
    c = np.array([W / 2, H / 2])
    vx = vanishing_point([l for l, _ in lines["x"]])
    vz = vanishing_point([l for l, _ in lines["z"]])
    right = named["x:right_edge"]["line"]
    front = named["y:front_edge"]["line"]
    corner = np.cross(right, front)
    corner = corner[:2] / corner[2]
    fdir = np.array([front[1], -front[0]])
    left_pt = corner - 200 * fdir * np.sign(fdir[0])  # a point along the front edge, to the left of the corner
    tape = cfg["tape"]
    hook, case = np.array(tape["hook_px"], float), np.array(tape["case_px"], float)
    chk = cfg["checks"]

    def solve(f):
        K = intrinsics(f, W, H)
        Ki = np.linalg.inv(K)
        rx = Ki @ vx; rx /= np.linalg.norm(rx)
        rz = Ki @ vz; rz /= np.linalg.norm(rz)
        rz = rz - (rz @ rx) * rx; rz /= np.linalg.norm(rz)
        _, Rf, rdeg = fit_orientation(lines, c, f, np.column_stack([rx, np.cross(rz, rx), rz]), fix_f=True)
        R, t1 = plane_pose(K, K @ Rf[:, 0], K @ Rf[:, 1], corner, case, left_pt)
        unit = img_to_plane_fn(K, R, t1, 1.0)
        scale = tape["length_m"] / float(np.linalg.norm(np.diff(unit(np.array([hook, case])), axis=0)))
        P = img_to_plane_fn(K, R, t1, scale)
        b1 = float(np.linalg.norm(np.diff(P(chk["bolts_plate_near"]), axis=0)))
        b2 = float(np.linalg.norm(np.diff(P(chk["bolts_plate_far"]), axis=0)))
        cost = (np.sqrt(np.mean(rdeg ** 2)) / search["line_tol_deg"]) ** 2 + ((b1 - b2) / ((b1 + b2) / 2) / search["bolt_rel_tol"]) ** 2
        return dict(f=f, K=K, R=R, t1=t1, scale=scale, P=P, rdeg=rdeg, b=(b1, b2), cost=float(cost))

    scan = [solve(f) for f in np.arange(search["min"], search["max"] + 1, search["step"])]
    best = min(scan, key=lambda s: s["cost"])
    # f range where the cost stays within +1 of the minimum (rough 1-sigma band)
    ok = [s["f"] for s in scan if s["cost"] <= best["cost"] + 1]
    P = best["P"]
    h0, hc, ho = P(np.array([hook, case, chk["hole_px"]]))
    u = (hc - h0) / np.linalg.norm(hc - h0)
    Hm = plane_homography(best["K"], best["R"], best["t1"], best["scale"])  # metres -> px
    return {
        "f_px": float(best["f"]), "f_range_px": [float(min(ok)), float(max(ok))], "hfov_deg": float(np.degrees(2 * np.arctan(W / 2 / best["f"]))),
        "K": best["K"], "H_table_to_img": Hm, "corner_px": corner,
        "line_residuals_deg": {n: float(r) for n, r in zip([k for a in "xyz" for k in cfg["segments"] if k[0] == a], best["rdeg"])},
        "edge_rms_px": {n: v["rms_px"] for n, v in named.items()},
        "checks": {
            "hole_along_tape_m": float((ho - h0) @ u), "hole_expected_m": chk["hole_along_tape_m"],
            "hole_error_m": float((ho - h0) @ u - chk["hole_along_tape_m"]),
            "bolt_spacing_m": list(best["b"]),
            "tape_hook_xy_m": h0.tolist(), "tape_angle_to_x_deg": float(np.degrees(np.arctan2(u[1], u[0]))),
        },
    }


def plane_h_between(img_a: np.ndarray, img_b: np.ndarray, table_poly_b: np.ndarray) -> tuple[np.ndarray, dict]:
    """Homography A->B induced by the table plane: ORB matches whose B point lies on the table, RANSAC."""
    orb = cv2.ORB_create(6000)
    ga, gb = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    ka, da = orb.detectAndCompute(ga, None)
    kb, db = orb.detectAndCompute(gb, None)
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    pa = np.float32([ka[x.queryIdx].pt for x in m])
    pb = np.float32([kb[x.trainIdx].pt for x in m])
    on = np.array([cv2.pointPolygonTest(table_poly_b.astype(np.float32), tuple(map(float, p)), False) > 0 for p in pb])
    Hab, inl = cv2.findHomography(pa[on], pb[on], cv2.RANSAC, 2.0)
    inl = inl.ravel().astype(bool)
    err = np.linalg.norm(apply_h(Hab, pa[on][inl]) - pb[on][inl], axis=1)
    return Hab, {"matches_on_table": int(on.sum()), "inliers": int(inl.sum()), "reproj_rms_px": float(np.sqrt(np.mean(err ** 2)))}


def draw_grid(img: np.ndarray, Hm: np.ndarray, title: str) -> np.ndarray:
    out = img.copy()
    xs, ys = np.arange(0, 0.81, 0.1), np.arange(0, 1.21, 0.1)
    for x in xs:
        p = apply_h(Hm, np.c_[np.full(50, x), np.linspace(0, ys[-1], 50)])
        cv2.polylines(out, [p.astype(np.int32)], False, (0, 255, 255) if x else (0, 0, 255), 1, cv2.LINE_AA)
    for y in ys:
        p = apply_h(Hm, np.c_[np.linspace(0, xs[-1], 50), np.full(50, y)])
        cv2.polylines(out, [p.astype(np.int32)], False, (0, 255, 255) if y else (255, 0, 0), 1, cv2.LINE_AA)
    o, ex, ey = apply_h(Hm, [[0, 0], [0.1, 0], [0, 0.1]])
    cv2.arrowedLine(out, tuple(o.astype(int)), tuple(ex.astype(int)), (0, 0, 255), 3)
    cv2.arrowedLine(out, tuple(o.astype(int)), tuple(ey.astype(int)), (255, 0, 0), 3)
    cv2.putText(out, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main() -> None:
    cfg = yaml.safe_load((CONFIGS_DIR / "calib" / "tpv_annotations.yaml").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    ref = cfg["reference_setup"]
    sb = cfg["setups"][ref]
    img_b = frame(sb["frame"]["pair"], sb["frame"]["t"])
    cal = calibrate_reference(img_b, sb, cfg["focal_search"])
    Hb = cal["H_table_to_img"]
    table_poly_b = apply_h(Hb, [[-0.02, -0.02], [0.75, -0.02], [0.75, 1.3], [-0.02, 1.3]])
    table_poly_b = np.clip(table_poly_b, [0, 0], [W - 1, H - 1])
    result = {"convention": "table frame: origin = front-right corner, x = depth toward the back, y = along the front "
                            "edge toward the left, z = up; metres. H_table_to_img maps (x, y, 1) -> pixel; invert for "
                            "pixel -> table.",
              "image_size": [W, H], "setups": {}, "pair_to_setup": {}}
    result["setups"][ref] = {"K": cal["K"].tolist(), "H_table_to_img": Hb.tolist(), "f_px": cal["f_px"],
                             "f_range_px": cal["f_range_px"], "hfov_deg": cal["hfov_deg"],
                             "line_residuals_deg": cal["line_residuals_deg"], "edge_rms_px": cal["edge_rms_px"],
                             "checks": cal["checks"], "frame": sb["frame"]}
    cv2.imwrite(str(OUT / f"grid_{ref}.jpg"), draw_grid(img_b, Hb, f"set-up {ref}: 10 cm grid, origin front-right corner"))
    print(f"[calib] set-up {ref}: f = {cal['f_px']:.0f} px (range {cal['f_range_px']}), HFOV {cal['hfov_deg']:.1f} deg")
    print(f"        line residuals (deg): { {k: round(v, 2) for k, v in cal['line_residuals_deg'].items()} }")
    ch = cal["checks"]
    print(f"        hole along tape {ch['hole_along_tape_m'] * 100:.1f} cm (FPV {ch['hole_expected_m'] * 100:.1f}) -> "
          f"error {ch['hole_error_m'] * 100:+.1f} cm; bolt spacings {[round(b * 1000, 1) for b in ch['bolt_spacing_m']]} mm")
    for name, s in cfg["setups"].items():
        for p in s.get("pairs", []):
            result["pair_to_setup"][p] = name
        if name == ref:
            continue
        img_a = frame(s["frame"]["pair"], s["frame"]["t"])
        Hab, st = plane_h_between(img_a, img_b, table_poly_b)
        Ha = np.linalg.inv(Hab) @ Hb  # table -> img_b -> img_a
        result["setups"][name] = {"K": cal["K"].tolist(), "H_table_to_img": Ha.tolist(), "H_img_to_ref_img": Hab.tolist(),
                                  "plane_homography": st, "frame": s["frame"]}
        cv2.imwrite(str(OUT / f"grid_{name}.jpg"), draw_grid(img_a, Ha, f"set-up {name}: 10 cm grid (via plane homography to {ref})"))
        print(f"[calib] set-up {name}: table-plane homography to {ref}: {st}")
    dst = REPO_ROOT / "data" / "calib" / "tpv.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[calib] -> {dst}")


if __name__ == "__main__":
    main()
