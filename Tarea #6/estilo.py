"""Estilo común para todas las figuras del informe."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AZUL, NARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TINTA, TINTA2, GRILLA = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = [AZUL, NARANJA, AQUA]

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": TINTA2, "axes.labelcolor": TINTA, "axes.titlecolor": TINTA,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRILLA, "grid.linewidth": 0.7,
    "xtick.color": TINTA2, "ytick.color": TINTA2,
    "legend.frameon": False, "lines.linewidth": 2,
})
FIG = "fig"
import os
os.makedirs(FIG, exist_ok=True)
