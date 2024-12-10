import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes
from matplotlib.lines import Line2D
import seaborn as sns
import os
import numpy as np
from box import Box


def figure_grid(n_row, n_col, size=None, size_ax=None):
    if size is None and size_ax is not None:
        size = np.array([n_col, n_row]) * size_ax
    fig, axs = plt.subplots(n_row, n_col, figsize=size, constrained_layout=True)
    return fig, axs


def figure_save(fig:Figure, path, dpi=100, save_svg=False, pad_inches=0.01):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=pad_inches)

    if save_svg:
        root, _ = os.path.splitext(path)
        path_svg = f"{root}.pdf"
        fig.savefig(path_svg, dpi=dpi, bbox_inches='tight', pad_inches=pad_inches)

    plt.close(fig)