"""Locations and runtime settings shared by every module of the project."""
import os

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))

# input data
DATA_DIR = os.path.join(ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")            # IDRBT dataset is extracted here
TEMPLATE_DIR = os.path.join(DATA_DIR, "templates")      # blank cheque of every bank layout

# generated artefacts
OUTPUT_DIR = os.path.join(ROOT, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
SAMPLE_RESULTS_DIR = os.path.join(OUTPUT_DIR, "sample_results")

# the networks are small, so the CPU is enough; a GPU is used when one is present
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for _folder in (RAW_DATA_DIR, TEMPLATE_DIR, MODEL_DIR, FIGURES_DIR, SAMPLE_RESULTS_DIR):
    os.makedirs(_folder, exist_ok=True)
