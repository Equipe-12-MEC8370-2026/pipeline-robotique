"""P2.4d — metric object trajectories on the table from the TPV masks + hands + calibration.

    .venv-learn\\Scripts\\python.exe scripts\\perception\\object_trajectories.py   (numpy/opencv only)
-> data/perception/<pair>/objects.json, outputs/p2_4/poses_<pair>.jpg (fitted boxes drawn at each rest)
Per object: rest segments (pose x, y, yaw, posture, base height; stability over several frames) and moves
between rests (grasp = first hand contact before the motion, release = last contact after it). During a move
only the image path is kept: one fixed view gives no reliable height.
Gate: final short-piece position vs the P2.3 measurement (1-2 cm), rest stability (sigma < 5 mm), order.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from cta_learn import CONFIGS_DIR, REPO_ROOT
from cta_learn.perception.hands import HandTrack
from cta_learn.perception.objects import load_masks
from cta_learn.perception.poses import (box_corners, box_extents, contact_frames, fit_box_pose, inside_footprint,
                                       rest_segments, same_pose)
from cta_learn.video.calib import intrinsics, pose_from_homography, project, ray_to_height

OUT = REPO_ROOT / "outputs" / "p2_4"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects_cfg", type=Path, default=CONFIGS_DIR / "perception" / "objects_113656.yaml")
    ap.add_argument("--task_cfg", type=Path, default=CONFIGS_DIR / "tasks" / "cta_profile_upright.yaml")
    ap.add_argument("--fits_per_rest", type=int, default=6)
    ap.add_argument("--max_rest_contact", type=float, default=0.5, help="max fraction of frames with hand contact in a rest")
    ap.add_argument("--min_rest_s", type=float, default=1.0)
    ap.add_argument("--min_rest_iou", type=float, default=0.7, help="box-fit IoU below this = occluded, not a reliable rest")
    ap.add_argument("--zb_iou_tol", type=float, default=0.03, help="raised base kept if its IoU is within this of the table fit")
    a = ap.parse_args()
    ocfg = yaml.safe_load(a.objects_cfg.read_text(encoding="utf-8"))
    task = yaml.safe_load(a.task_cfg.read_text(encoding="utf-8"))
    pair, H = ocfg["pair"], ocfg["height"]
    d = REPO_ROOT / "data" / "perception" / pair
    cal = json.loads((REPO_ROOT / "data" / "calib" / "tpv.json").read_text(encoding="utf-8"))
    setup = cal["pair_to_setup"][pair]
    K_full = np.array(cal["setups"][setup]["K"])
    R, t = pose_from_homography(K_full, np.array(cal["setups"][setup]["H_table_to_img"]))
    sc = H / 1080.0
    K = intrinsics(K_full[0, 0] * sc, int(1920 * sc), H)

    hands = HandTrack.load(d / "hands_tpv.npz")
    masks = {n: load_masks(d / f"masks_{n}.npz") for n in ocfg["objects"]}
    fps = next(iter(masks.values()))[1]
    T = len(next(iter(masks.values()))[0])
    # hands: 30 fps full-res -> nearest frame at the mask rate, scaled to the mask resolution
    hidx = np.clip(np.searchsorted(hands.t_s, np.arange(T) / fps - 1e-4), 0, len(hands.t_s) - 1)
    hk = hands.kp_px[hidx] * sc

    result = {"pair": pair, "setup": setup, "fps": fps, "frame": "table frame of data/calib/tpv.json (m, deg)",
              "objects": {}}
    drawn, supports = {}, []
    for name, (m, _) in masks.items():
        dims = task["objects"][name]["size_m"]
        section, length = float(dims[1]), float(max(dims))
        z_bases = (0.0, section) if name == "profile_long" else (0.0,)
        area = m.reshape(T, -1).sum(1)
        cen = np.array([np.argwhere(mk).mean(0)[::-1] if a_ else [np.nan, np.nan] for mk, a_ in zip(m, area)])
        contact = contact_frames(m, hk)
        still = rest_segments(cen, fps, speed_px=1.0, min_s=a.min_rest_s)
        # a part held still in the hand is not at rest: keep segments where the hand mostly does not touch it
        rests = [(ra, rb) for ra, rb in still if contact[ra:rb + 1].mean() <= a.max_rest_contact]
        held_still = [(ra, rb) for ra, rb in still if (ra, rb) not in rests]
        rest_out, uncertain = [], []
        for ra, rb in rests:
            idx = np.arange(ra, rb + 1)
            free = idx[~contact[idx]]
            use = free if free.size >= 3 else idx
            pick = use[np.linspace(0, len(use) - 1, min(a.fits_per_rest, len(use))).round().astype(int)]
            by_zb = {zb: [] for zb in z_bases}
            for i in pick:
                ys, xs = np.nonzero(m[i])
                low = np.argmax(ys)
                init = ray_to_height(K, R, t, (xs[low], ys[low]), 0.0)[:2]
                for zb in z_bases:
                    by_zb[zb].append(fit_box_pose(m[i], K, R, t, section, length, init, z_bases=(zb,)))
            # base height: one fixed view barely separates "on a support at height h" from "on the table, further
            # back". Prefer a raised base only if the fit there rests on a known support (another part lying at
            # that time, top at h); otherwise the table.
            zb = 0.0
            for h in z_bases:
                if h <= 0:
                    continue
                fh = by_zb[h]
                cx, cy = np.median([f.x for f in fh]), np.median([f.y for f in fh])
                sup = [o for o in supports if o["t"][0] <= rb / fps and o["t"][1] >= ra / fps and abs(o["top"] - h) < 1e-6
                       and inside_footprint(cx, cy, o, margin=0.01)]
                iou_h, iou_0 = np.median([f.iou for f in fh]), np.median([f.iou for f in by_zb[0.0]])
                if sup and iou_h >= iou_0 - a.zb_iou_tol:
                    zb = h
            fits = by_zb[zb]
            ups = [f.upright for f in fits]
            up = max(set(ups), key=ups.count)
            sel = [f for f in fits if f.upright == up]
            xy = np.array([[f.x, f.y] for f in sel])
            per = 90.0 if up else 180.0
            yaws = np.array([f.yaw_deg for f in sel])
            ref = yaws[0]
            yaws = ref + ((yaws - ref + per / 2) % per - per / 2)  # unwrap around the first
            pose = {"x": float(np.median(xy[:, 0])), "y": float(np.median(xy[:, 1])), "yaw_deg": float(np.median(yaws) % per),
                    "upright": bool(up), "z_base": float(zb), "iou_median": float(np.median([f.iou for f in sel]))}
            if pose["iou_median"] < a.min_rest_iou:  # mask too partial (occluded part): not a reliable rest
                uncertain.append({"t": [round(ra / fps, 2), round(rb / fps, 2)], "pose": pose})
                continue
            rest_out.append({"_f": (ra, rb), "t": [round(ra / fps, 2), round(rb / fps, 2)], "pose": pose,
                             "xy_std_mm": float(np.linalg.norm(xy.std(0)) * 1000), "yaw_std_deg": float(yaws.std()),
                             "n_fits": len(sel), "n_fits_disagree": len(fits) - len(sel),
                             "contact_frac": float(contact[ra:rb + 1].mean()),
                             "iou_by_base": {str(k): float(np.median([f.iou for f in v])) for k, v in by_zb.items()}})
            drawn.setdefault(name, []).append((rb, pose, section, length))
            if not up:  # a lying part is a support for what may be put on it
                supports.append({"t": [ra / fps, rb / fps], "top": section, "x": pose["x"], "y": pose["y"],
                                 "yaw_deg": pose["yaw_deg"], "ext": box_extents(section, length, False)[:2].tolist()})
        # consecutive rests with the same pose = the part was only touched in between, not moved: merge them
        merged = []
        for r_ in rest_out:
            if merged and same_pose(merged[-1]["pose"], r_["pose"]):
                prev = merged[-1]
                prev["_f"] = (prev["_f"][0], r_["_f"][1])
                prev["t"] = [prev["t"][0], r_["t"][1]]
                prev["xy_std_mm"] = max(prev["xy_std_mm"], r_["xy_std_mm"])
                prev["yaw_std_deg"] = max(prev["yaw_std_deg"], r_["yaw_std_deg"])
                prev["merged_from"] = prev.get("merged_from", 1) + 1
            else:
                merged.append(r_)
        rest_out = merged
        rests = [r_.pop("_f") for r_ in rest_out]
        moves = []
        for (a0, b0), (a1, b1) in zip(rests[:-1], rests[1:]):
            g = b0 + 1
            while g - 1 > a0 and contact[g - 1]:
                g -= 1
            r = a1 - 1
            while r + 1 < b1 and contact[r + 1]:
                r += 1
            path = cen[b0:a1 + 1] / sc
            moves.append({"t_grasp": round(g / fps, 2), "t_motion": [round((b0 + 1) / fps, 2), round((a1 - 1) / fps, 2)],
                          "t_release": round(r / fps, 2), "contact_frac_during_motion": float(contact[b0 + 1:a1].mean()) if a1 > b0 + 1 else 0.0,
                          "image_path_px": np.round(path[::3], 1).tolist()})
        result["objects"][name] = {"rests": rest_out, "moves": moves,
                                   "held_still_s": [[round(x / fps, 2), round(y / fps, 2)] for x, y in held_still],
                                   "uncertain_still": uncertain,
                                   "contact_frames": int(contact.sum()), "frames": T}
        print(f"[traj] {name}: {len(rest_out)} rests, {len(moves)} moves")
        for r_ in rest_out:
            p = r_["pose"]
            print(f"   rest {r_['t'][0]:5.1f}-{r_['t'][1]:5.1f} s: x {p['x']:.3f} y {p['y']:.3f} yaw {p['yaw_deg']:5.1f} "
                  f"{'UPRIGHT' if p['upright'] else 'lying'} base {p['z_base']:.2f} | IoU {p['iou_median']:.2f} "
                  f"| sigma {r_['xy_std_mm']:.1f} mm, {r_['yaw_std_deg']:.1f} deg | fits {r_['n_fits']}/{r_['n_fits'] + r_['n_fits_disagree']}")
        for mv in moves:
            print(f"   move: grasp {mv['t_grasp']} s, motion {mv['t_motion']} s, release {mv['t_release']} s, "
                  f"contact during motion {mv['contact_frac_during_motion']:.0%}")
    # gate
    gate = {}
    sh = result["objects"]["profile_short"]["rests"][-1]["pose"]
    ref = task["objects"]["profile_short"]["final_pose"]["xy_m"]  # P2.3: centre of the front bottom edge
    c = box_corners(box_extents(0.04, 0.08, sh["upright"]), sh["x"], sh["y"], sh["yaw_deg"], 0.0)[:4]
    C = -R.T @ t
    # front bottom edge = the long bottom edge nearest the camera
    edges = [(c[0], c[2]), (c[1], c[3]), (c[0], c[1]), (c[2], c[3])]
    long_edges = sorted(edges, key=lambda e: -np.linalg.norm(e[0][:2] - e[1][:2]))[:2]
    front = min(long_edges, key=lambda e: np.linalg.norm((e[0] + e[1])[:2] / 2 - C[:2]))
    fm = (front[0] + front[1])[:2] / 2
    gate["short_final_front_edge_xy"] = fm.round(4).tolist()
    gate["short_final_vs_p23_mm"] = float(np.linalg.norm(fm - ref) * 1000)
    gate["max_rest_sigma_mm"] = max(r_["xy_std_mm"] for o in result["objects"].values() for r_ in o["rests"])
    tg = {n: o["moves"][0]["t_grasp"] if o["moves"] else None for n, o in result["objects"].items()}
    gate["order_short_then_long"] = bool(tg["profile_short"] is not None and tg["profile_long"] is not None
                                         and tg["profile_short"] < tg["profile_long"])
    lg = result["objects"]["profile_long"]["rests"][-1]["pose"]
    gate["long_final_upright_on_short"] = bool(lg["upright"] and lg["z_base"] > 0.02
                                               and np.hypot(lg["x"] - sh["x"], lg["y"] - sh["y"]) < 0.05)
    gate["pass"] = bool(gate["short_final_vs_p23_mm"] <= 20 and gate["max_rest_sigma_mm"] < 5 and gate["order_short_then_long"])
    result["gate"] = gate
    (d / "objects.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[traj] gate: {gate}")

    # control image: first and last frames with the fitted boxes of every rest
    frames = sorted((d / f"frames_tpv_{fps}fps").glob("*.jpg"))
    tiles = []
    for fi in (0, T - 1):
        img = cv2.imread(str(frames[fi]))
        for name, lst in drawn.items():
            col = (0, 140, 255) if name == "profile_short" else (255, 200, 0)
            for rb, p, s_, L_ in lst:
                if (fi == 0 and rb != lst[0][0]) or (fi == T - 1 and rb != lst[-1][0]):
                    continue
                cs = project(K, R, t, box_corners(box_extents(s_, L_, p["upright"]), p["x"], p["y"], p["yaw_deg"], p["z_base"]))
                for i, j in [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
                    cv2.line(img, tuple(cs[i].astype(int)), tuple(cs[j].astype(int)), col, 2, cv2.LINE_AA)
        cv2.putText(img, f"{fi / fps:.1f} s: fitted boxes", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        tiles.append(img)
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / f"poses_{pair}.jpg"), np.hstack(tiles))


if __name__ == "__main__":
    main()
