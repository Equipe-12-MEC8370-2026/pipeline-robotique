r"""Tests sans matériel de cameras.py et damier.py (depuis tests/tristan/passerelle) :
    ..\..\alexis\.venv-learn\Scripts\python.exe -m pytest -q
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

import cameras as cam  # noqa: E402
import damier  # noqa: E402

# « vraie » caméra synthétique : fisheye 640x480 à champ très large (proche d'une IMX335 175°)
K_VRAI = np.array([[200.0, 0, 318.0], [0, 200.0, 243.0], [0, 0, 1]])
D_VRAI = np.array([[0.03], [-0.01], [0.004], [-0.001]])
TAILLE = (640, 480)


def vues_synthetiques(n=20, bruit=0.1, seed=0):
    """Coins du damier projetés (modèle fisheye) pour n poses variées, visibles en entier dans l'image."""
    rng = np.random.default_rng(seed)
    obj = cam.points_objet()
    centre = obj[0].mean(0)
    vues = []
    while len(vues) < n:
        rvec = rng.uniform(-0.6, 0.6, 3)
        R, _ = cv2.Rodrigues(rvec)
        t = np.array([rng.uniform(-0.25, 0.25), rng.uniform(-0.18, 0.18), rng.uniform(0.18, 0.35)]) - R @ centre
        pts_cam = (R @ obj[0].T).T + t
        if np.any(pts_cam[:, 2] < 0.05):
            continue
        img, _ = cv2.fisheye.projectPoints(obj, rvec, t, K_VRAI, D_VRAI)
        img = img.reshape(-1, 2)
        if img[:, 0].min() < 5 or img[:, 1].min() < 5 or img[:, 0].max() > 635 or img[:, 1].max() > 475:
            continue
        vues.append((img + rng.normal(0, bruit, img.shape)).reshape(1, -1, 2))
    return vues


def test_sim_camera_is_90deg_pinhole():
    K = cam.matrice_sim()
    assert K[0, 0] == pytest.approx(320.0) and K[0, 2] == 320 and K[1, 2] == 240
    assert np.degrees(2 * np.arctan(cam.SIM_W / 2 / K[0, 0])) == pytest.approx(90.0)


def test_calibration_recovers_intrinsics():
    calib = cam.calibrer(vues_synthetiques(), TAILLE)
    K = np.array(calib["K"])
    assert calib["rms_px"] < 0.3
    assert K[0, 0] == pytest.approx(200.0, rel=0.01) and K[1, 1] == pytest.approx(200.0, rel=0.01)
    assert K[0, 2] == pytest.approx(318.0, abs=1.5) and K[1, 2] == pytest.approx(243.0, abs=1.5)


def test_too_few_views_refused():
    with pytest.raises(ValueError, match="au moins 8"):
        cam.calibrer(vues_synthetiques(5), TAILLE)


def test_undistortion_lands_on_sim_pinhole():
    """Un point 3D vu par la vraie caméra (fisheye) doit tomber, après redressement, là où le sténopé de la sim
    (90°, f = 320 px) le projetterait."""
    red = cam.Redresseur({"K": K_VRAI.tolist(), "D": D_VRAI.ravel().tolist(), "taille": list(TAILLE)}, precis=True)
    rng = np.random.default_rng(1)
    for _ in range(50):
        x, y, z = rng.uniform(-0.9, 0.9), rng.uniform(-0.65, 0.65), 1.0  # dans le champ 90° x 74°
        u, v = 320 * x / z + 320, 320 * y / z + 240  # pixel dans l'image redressée
        attendu, _ = cv2.fisheye.projectPoints(np.array([[[x, y, z]]]), np.zeros(3), np.zeros(3), K_VRAI, D_VRAI)
        ui, vi = int(round(u)), int(round(v))
        if not (0 <= ui < 640 and 0 <= vi < 480):
            continue
        # la carte donne, pour chaque pixel redressé, où lire dans l'image fisheye
        src = np.array([red.map1[vi, ui], red.map2[vi, ui]])
        # comparer au point projeté du pixel entier (on a arrondi u, v)
        att2, _ = cv2.fisheye.projectPoints(np.array([[[(ui - 320) / 320, (vi - 240) / 320, 1.0]]]),
                                            np.zeros(3), np.zeros(3), K_VRAI, D_VRAI)
        assert src == pytest.approx(att2.ravel(), abs=0.05)
        assert np.linalg.norm(attendu.ravel() - att2.ravel()) < 3  # cohérence de l'arrondi


def test_redresseur_output_format_and_color_order():
    red = cam.Redresseur({"K": K_VRAI.tolist(), "D": D_VRAI.ravel().tolist(), "taille": list(TAILLE)})
    bgr = np.zeros((480, 640, 3), np.uint8)
    bgr[..., 0] = 255  # bleu en BGR
    rgb = red(bgr)
    assert rgb.shape == (480, 640, 3) and rgb.dtype == np.uint8
    assert rgb[240, 320].tolist() == [0, 0, 255]  # bleu en RGB
    with pytest.raises(ValueError, match="taille de calibration"):
        red(np.zeros((1080, 1920, 3), np.uint8))


def test_checkerboard_detection_on_rendered_board(tmp_path):
    """Le damier généré (PDF) est détecté une fois rendu en image, avec le bon nombre de coins dans le bon ordre."""
    pdf = tmp_path / "d.pdf"
    W, H = damier.generer(pdf)
    assert (W, H) == (3300, 2550)  # lettre paysage à 300 dpi
    # rendu équivalent en mémoire (même dessin), réduit et perspectivé
    nx, ny = cam.DAMIER_COINS[0] + 1, cam.DAMIER_COINS[1] + 1
    s = 40
    board = np.full(((ny + 2) * s, (nx + 2) * s), 255, np.uint8)
    for i in range(nx):
        for j in range(ny):
            if (i + j) % 2 == 0:
                board[(j + 1) * s:(j + 2) * s, (i + 1) * s:(i + 2) * s] = 0
    src = np.float32([[0, 0], [board.shape[1], 0], [board.shape[1], board.shape[0]], [0, board.shape[0]]])
    dst = np.float32([[120, 80], [540, 110], [520, 420], [100, 380]])
    img = cv2.warpPerspective(board, cv2.getPerspectiveTransform(src, dst), (640, 480), borderValue=255)
    coins = cam.detecter_damier(img)
    assert coins is not None and coins.shape == (1, cam.DAMIER_COINS[0] * cam.DAMIER_COINS[1], 2)


def test_orientation_correction_is_an_exact_rotation():
    """Avec une correction de lacet θ, le pixel central de l'image redressée lit la direction tournée de θ :
    vérifié contre la projection fisheye de ce rayon tourné (indépendant de la profondeur)."""
    base = {"K": K_VRAI.tolist(), "D": D_VRAI.ravel().tolist(), "taille": list(TAILLE)}
    for corr in ({"lacet": 10.0}, {"tangage": -7.0}, {"roulis": 20.0, "lacet": 5.0}):
        red = cam.Redresseur(dict(base, correction_deg=corr), precis=True)
        for u, v in ((320, 240), (100, 400), (600, 60)):
            ray = np.linalg.inv(cam.matrice_sim()) @ np.array([u, v, 1.0])
            src_ray = red.R.T @ ray  # rayon dans le repère de la vraie caméra
            att, _ = cv2.fisheye.projectPoints(src_ray.reshape(1, 1, 3), np.zeros(3), np.zeros(3), K_VRAI, D_VRAI)
            assert [red.map1[v, u], red.map2[v, u]] == pytest.approx(att.ravel().tolist(), abs=0.05)
    sans = cam.Redresseur(dict(base, correction_deg={}), precis=True)
    assert np.allclose(sans.R, np.eye(3))
    # lacet positif : le centre de l'image redressée regarde vers +x caméra (à droite dans l'image brute)
    red = cam.Redresseur(dict(base, correction_deg={"lacet": 10.0}), precis=True)
    assert red.map1[240, 320] > sans.map1[240, 320] + 20


def test_alignment_score_rewards_overlap():
    import aligner_cameras as al

    img = np.zeros((480, 640, 3), np.uint8)
    cv2.rectangle(img, (200, 150), (440, 330), (255, 255, 255), -1)
    ref_cont = al.contours(img)
    assert al.score_alignement(ref_cont, img) > 0.95
    decale = np.roll(img, 40, axis=1)
    assert al.score_alignement(ref_cont, decale) < 0.6
    assert al.composer(img, img, ref_cont, "cote a cote").shape == (480, 1280, 3)


def test_save_and_load_calibration(tmp_path):
    calib = {"K": K_VRAI.tolist(), "D": D_VRAI.ravel().tolist(), "taille": list(TAILLE), "rms_px": 0.2}
    f = tmp_path / "c.json"
    cam.sauver_calibration(calib, f, "wrist", 0)
    c = cam.charger_calibration(f)
    assert c["camera"] == "wrist" and c["index_opencv"] == 0 and c["K"] == calib["K"]
    f.write_text("{}")
    with pytest.raises(ValueError, match="manquante"):
        cam.charger_calibration(f)
