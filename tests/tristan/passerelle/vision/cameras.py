"""Caméras réelles (Waveshare IMX335 fisheye ~175°) -> images « comme en sim » pour la politique.

Les caméras de la sim (scène P4 d'Alexis) sont des sténopés 640 x 480, 90° de champ horizontal : focale 320 px,
centre (320, 240), sans distorsion. Chaque caméra réelle est calibrée (modèle fisheye d'OpenCV, damier), puis son
image est redressée vers EXACTEMENT ce sténopé. Sortie : RGB uint8 (480, 640, 3), le format envoyé à policy_server.

Mode vidéo retenu : 640 x 480 MJPG (20 img/s mesurées ; 2592x1944 et 1280x960 : 1-2,5 img/s, inutilisables).
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

SIM_W, SIM_H = 640, 480
SIM_F = SIM_W / 2 / np.tan(np.radians(90.0) / 2)  # = 320 px : 90° horizontal
DAMIER_COINS = (8, 6)  # coins INTÉRIEURS (damier de 9 x 7 cases)
DAMIER_CARRE_M = 0.025


def matrice_sim() -> np.ndarray:
    return np.array([[SIM_F, 0.0, SIM_W / 2], [0.0, SIM_F, SIM_H / 2], [0.0, 0.0, 1.0]])


# ---------------------------------------------------------------------------------------------- acquisition
def ouvrir(index: int, largeur: int = 640, hauteur: int = 480, chauffe: int = 15) -> cv2.VideoCapture:
    """Ouvre une caméra USB (DirectShow, MJPG) au format demandé ; lit quelques images (exposition automatique)."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"caméra {index} introuvable")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, largeur)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hauteur)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    got = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if got != (largeur, hauteur):
        cap.release()
        raise RuntimeError(f"caméra {index} : format {got} au lieu de {(largeur, hauteur)}")
    for _ in range(chauffe):
        cap.read()
    return cap


def lire(cap: cv2.VideoCapture) -> np.ndarray:
    ok, bgr = cap.read()
    if not ok or bgr is None:
        raise RuntimeError("lecture caméra impossible")
    return bgr


# ---------------------------------------------------------------------------------------------- damier
def points_objet(coins=DAMIER_COINS, carre: float = DAMIER_CARRE_M) -> np.ndarray:
    """Coins intérieurs du damier dans son plan (z = 0), (1, N, 3) float64, même ordre que la détection."""
    cx, cy = coins
    g = np.zeros((cx * cy, 3), np.float64)
    g[:, :2] = np.mgrid[0:cx, 0:cy].T.reshape(-1, 2) * carre
    return g[None]


def detecter_damier(bgr_ou_gris: np.ndarray, coins=DAMIER_COINS, precis: bool = True) -> np.ndarray | None:
    """Coins intérieurs détectés (1, N, 2) float64, ou None. Détecteur « SB » (robuste à la distorsion fisheye).
    ``precis=False`` : plus rapide, pour l'aperçu en direct ; ``True`` pour les vues gardées."""
    gris = bgr_ou_gris if bgr_ou_gris.ndim == 2 else cv2.cvtColor(bgr_ou_gris, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY if precis else cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, c = cv2.findChessboardCornersSB(gris, coins, flags=flags)
    return c.reshape(1, -1, 2).astype(np.float64) if ok else None


# ---------------------------------------------------------------------------------------------- calibration
def calibrer(vues: list[np.ndarray], taille: tuple[int, int], coins=DAMIER_COINS,
             carre: float = DAMIER_CARRE_M) -> dict:
    """Calibration fisheye (K, D) à partir des coins détectés dans plusieurs vues. ``taille`` = (largeur, hauteur)."""
    if len(vues) < 8:
        raise ValueError(f"{len(vues)} vues : il en faut au moins 8 (idéalement 20, bords et coins compris)")
    obj = [points_objet(coins, carre)] * len(vues)
    # estimation initiale INDISPENSABLE : sans elle, l'initialisation interne d'OpenCV diverge sur un fisheye
    # aussi large (RMS ~150 px sur données synthétiques). f ~ largeur / 3 ≈ modèle équidistant à ~175° diagonal.
    K0 = np.array([[taille[0] / 3, 0, taille[0] / 2], [0, taille[0] / 3, taille[1] / 2], [0, 0, 1]], np.float64)
    crit = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 200, 1e-9)
    base = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW | cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
    try:
        rms, K, D, _, _ = cv2.fisheye.calibrate(obj, list(vues), taille, K0.copy(), np.zeros((4, 1)),
                                               flags=base | cv2.fisheye.CALIB_CHECK_COND, criteria=crit)
    except cv2.error:  # une vue mal conditionnée : on recommence sans le contrôle
        rms, K, D, _, _ = cv2.fisheye.calibrate(obj, list(vues), taille, K0.copy(), np.zeros((4, 1)),
                                               flags=base, criteria=crit)
    return {"K": K.tolist(), "D": D.reshape(-1).tolist(), "rms_px": float(rms), "taille": list(taille),
            "n_vues": len(vues), "damier_coins": list(coins), "damier_carre_m": carre}


def sauver_calibration(calib: dict, path: str | Path, nom: str, index: int) -> None:
    calib = dict(calib, camera=nom, index_opencv=index, date=datetime.now().isoformat(timespec="seconds"))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(calib, indent=1), encoding="utf-8")


def charger_calibration(path: str | Path) -> dict:
    c = json.loads(Path(path).read_text(encoding="utf-8"))
    for k in ("K", "D", "taille"):
        if k not in c:
            raise ValueError(f"{path} : clé {k} manquante")
    return c


# ---------------------------------------------------------------------------------------------- redressement
def rotation_correction(lacet: float = 0.0, tangage: float = 0.0, roulis: float = 0.0) -> np.ndarray:
    """Rotation (degrés) de la caméra VIRTUELLE par rapport à la vraie, axes caméra OpenCV : lacet autour de y
    (vers le bas de l'image -> décale l'image horizontalement), tangage autour de x (décalage vertical), roulis
    autour de z (axe optique). Une rotation pure se corrige exactement, quelle que soit la profondeur."""
    a, b, c = np.radians([lacet, tangage, roulis])
    ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return rz @ rx @ ry


class Redresseur:
    """Image fisheye (BGR, taille de calibration) -> sténopé de la sim (RGB uint8 480 x 640).
    ``calib["correction_deg"]`` (optionnel) = {"lacet", "tangage", "roulis"} : la caméra virtuelle est tournée
    d'autant (support de caméra non réglable : on corrige l'orientation par logiciel)."""

    def __init__(self, calib: dict, precis: bool = False):
        self.taille = tuple(calib["taille"])
        K, D = np.array(calib["K"], np.float64), np.array(calib["D"], np.float64).reshape(4, 1)
        corr = calib.get("correction_deg") or {}
        # OpenCV attend la rotation repère réel -> repère virtuel : transposée de l'orientation de la caméra virtuelle
        self.R = rotation_correction(**{k: float(corr.get(k, 0.0)) for k in ("lacet", "tangage", "roulis")}).T
        kind = cv2.CV_32FC1 if precis else cv2.CV_16SC2
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(K, D, self.R, matrice_sim(),
                                                                   (SIM_W, SIM_H), kind)

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        if (bgr.shape[1], bgr.shape[0]) != self.taille:
            raise ValueError(f"image {bgr.shape[1]}x{bgr.shape[0]} != taille de calibration {self.taille}")
        out = cv2.remap(bgr, self.map1, self.map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


class FluxCamera:
    """Lit une caméra en continu dans un fil d'exécution : ``derniere()`` renvoie toujours l'image la plus récente
    (sinon OpenCV/DirectShow peut rendre des images en retard, gardées dans son tampon)."""

    def __init__(self, index: int, largeur: int = 640, hauteur: int = 480):
        import threading

        self.cap = ouvrir(index, largeur, hauteur)
        self._img, self._t, self._n = None, 0.0, 0
        self._verrou = threading.Lock()
        self._actif = True
        self._fil = threading.Thread(target=self._lire, daemon=True)
        self._fil.start()
        t0 = time.perf_counter()
        while self._img is None and time.perf_counter() - t0 < 3.0:
            time.sleep(0.01)
        if self._img is None:
            raise RuntimeError(f"caméra {index} : aucune image")

    def _lire(self):
        while self._actif:
            ok, img = self.cap.read()
            if ok:
                with self._verrou:
                    self._img, self._t, self._n = img, time.perf_counter(), self._n + 1

    def derniere(self) -> tuple[np.ndarray, float]:
        """(image BGR, âge en secondes)."""
        with self._verrou:
            return self._img.copy(), time.perf_counter() - self._t

    def fermer(self):
        self._actif = False
        self._fil.join(timeout=1.0)
        self.cap.release()


def mesurer_cadence(cap: cv2.VideoCapture, duree: float = 2.0) -> float:
    t, n = time.perf_counter(), 0
    while time.perf_counter() - t < duree:
        n += cap.read()[0]
    return n / (time.perf_counter() - t)
