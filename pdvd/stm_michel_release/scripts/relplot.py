"""relplot.py -- shared matplotlib style for the release example scripts (no seaborn needed)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# fixed categorical order (never cycled), single-hue sequential map for magnitudes
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET, RED, GREEN = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#e34948", "#008300")
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
SEQ = "Blues"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "lines.linewidth": 2.0, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "legend.frameon": False, "figure.dpi": 110,
})


def finish(fig, out, note=None):
    if note:
        fig.text(0.01, 0.005, note, fontsize=8, color=INK2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.02 if note else 0, 1, 1))
    fig.savefig(out)
    print("wrote", out)
