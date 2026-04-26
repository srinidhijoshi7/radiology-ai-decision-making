"""
plotting.py
-----------
Shared matplotlib styling so every chart in figures/ has a consistent look
suitable for a Word-document report.
"""

import matplotlib.pyplot as plt

PALETTE = {
    "primary":   "#1f4e79",   # deep blue
    "secondary": "#c8102e",   # accent red
    "accent":    "#f5a623",   # warm orange
    "neutral":   "#4a4a4a",
    "muted":     "#9aa0a6",
    "grid":      "#e6e6e6",
}


def apply_style():
    plt.rcParams.update({
        "figure.dpi":          120,
        "savefig.dpi":         220,
        "savefig.bbox":        "tight",
        "font.family":         "DejaVu Sans",
        "font.size":           11,
        "axes.titlesize":      13,
        "axes.titleweight":    "bold",
        "axes.labelsize":      11,
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.grid":           True,
        "grid.color":          PALETTE["grid"],
        "grid.linewidth":      0.7,
        "legend.frameon":      False,
        "legend.fontsize":     10,
    })
