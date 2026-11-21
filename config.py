"""Locations and runtime settings shared by every module of the project."""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

# input data
DATA_DIR = os.path.join(ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")            # IDRBT dataset is extracted here

# generated artefacts
OUTPUT_DIR = os.path.join(ROOT, "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
SAMPLE_RESULTS_DIR = os.path.join(OUTPUT_DIR, "sample_results")

# The blank bank templates and the trained models can be moved elsewhere (the web API
# ships them in backend/artifacts) by setting these environment variables.
TEMPLATE_DIR = os.environ.get("CHEQUE_TEMPLATE_DIR", os.path.join(DATA_DIR, "templates"))
MODEL_DIR = os.environ.get("CHEQUE_MODEL_DIR", os.path.join(OUTPUT_DIR, "models"))

for _folder in (RAW_DATA_DIR, FIGURES_DIR, SAMPLE_RESULTS_DIR):
    try:
        os.makedirs(_folder, exist_ok=True)
    except OSError:                  # read-only file system on some hosts: not needed for inference
        pass
for _folder in (TEMPLATE_DIR, MODEL_DIR):
    try:
        os.makedirs(_folder, exist_ok=True)
    except OSError:
        pass


def __getattr__(name):
    """`config.DEVICE` imports PyTorch only when it is first used, so the Random Forest
    path (and the web API on a small server) never has to load it."""
    if name == "DEVICE":
        import torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise AttributeError(f"module 'config' has no attribute {name!r}")
