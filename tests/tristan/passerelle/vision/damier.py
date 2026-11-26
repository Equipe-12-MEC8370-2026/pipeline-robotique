r"""Génère le damier de calibration à imprimer (PDF, format lettre paysage) : 9 x 7 cases de 25 mm
(8 x 6 coins intérieurs). Imprimer à 100 % (« taille réelle », sans ajustement à la page), coller sur un carton
rigide et bien plat. La taille exacte des cases n'influence pas la correction de distorsion.

    ..\..\alexis\.venv-learn\Scripts\python.exe vision\damier.py
"""

from pathlib import Path

from PIL import Image, ImageDraw

from cameras import DAMIER_CARRE_M, DAMIER_COINS

DPI = 300
PAGE_MM = (279.4, 215.9)  # lettre, paysage


def mm(v: float) -> int:
    return round(v / 25.4 * DPI)


def generer(path: str | Path) -> tuple[int, int]:
    W, H = mm(PAGE_MM[0]), mm(PAGE_MM[1])
    img = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(img)
    nx, ny = DAMIER_COINS[0] + 1, DAMIER_COINS[1] + 1
    c = DAMIER_CARRE_M * 1000
    x0, y0 = (PAGE_MM[0] - nx * c) / 2, (PAGE_MM[1] - ny * c) / 2
    for i in range(nx):
        for j in range(ny):
            if (i + j) % 2 == 0:
                d.rectangle([mm(x0 + i * c), mm(y0 + j * c), mm(x0 + (i + 1) * c) - 1, mm(y0 + (j + 1) * c) - 1],
                            fill=0)
    d.text((mm(10), mm(4)), f"Damier {nx}x{ny} cases de {c:.0f} mm - IMPRIMER A 100 % (taille reelle)", fill=0)
    img.save(path, "PDF", resolution=DPI)
    return W, H


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "damier_9x7_25mm_lettre.pdf"
    out.parent.mkdir(exist_ok=True)
    print(generer(out), "->", out)
