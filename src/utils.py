"""Small helpers used by the training and evaluation code."""
import os
import random

import matplotlib
matplotlib.use("Agg")                      # figures are written to disk, never shown
import numpy as np
import torch

import config


def set_seed(seed=42):
    """Make Python, NumPy and PyTorch draw the same random numbers on every run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_figure(fig, name, folder=None):
    """Write a Matplotlib figure as <name>.png (150 dpi, white background) and close it."""
    import matplotlib.pyplot as plt

    folder = folder or config.FIGURES_DIR
    os.makedirs(folder, exist_ok=True)
    target = os.path.join(folder, name + ".png")
    fig.savefig(target, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved figure -> {target}")
    return target
