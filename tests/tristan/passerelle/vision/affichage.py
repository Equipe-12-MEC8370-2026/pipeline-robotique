"""Petite fenêtre d'affichage + clavier en tkinter.

``.venv-learn`` contient opencv-python-HEADLESS (exigé par LeRobot) : ``cv2.imshow`` n'y existe pas. On ne change pas
de paquet (risque de casser LeRobot) : tkinter (bibliothèque standard) affiche les images et lit le clavier.
"""

from __future__ import annotations

import tkinter as tk
from collections import deque

import cv2
import numpy as np
from PIL import Image, ImageTk


class Fenetre:
    def __init__(self, titre: str):
        self.root = tk.Tk()
        self.root.title(titre)
        self.label = tk.Label(self.root)
        self.label.pack()
        self._touches: deque[str] = deque()
        self._tk_img = None
        self.ouverte = True
        self.root.bind("<Key>", self._clavier)
        self.root.protocol("WM_DELETE_WINDOW", self.fermer)

    def _clavier(self, e) -> None:
        self._touches.append(" " if e.keysym == "space" else (e.char or e.keysym).lower())

    def montrer(self, bgr: np.ndarray) -> None:
        """Affiche une image BGR (convention OpenCV) et traite les événements de la fenêtre."""
        if not self.ouverte:
            return
        self._tk_img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        self.label.configure(image=self._tk_img)
        self.root.update()

    def touche(self) -> str | None:
        """Dernière touche appuyée (une à la fois), « q » si la fenêtre a été fermée, sinon None."""
        if not self.ouverte:
            return "q"
        self.root.update()
        return self._touches.popleft() if self._touches else None

    def fermer(self) -> None:
        if self.ouverte:
            self.ouverte = False
            self.root.destroy()
