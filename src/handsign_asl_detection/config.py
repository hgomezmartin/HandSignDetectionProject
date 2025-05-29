# src/handsign_asl_detection/config.py
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]  # ajusta niveles hasta la raíz de tu proyecto

# paths a modelos, etiquetas, assets…
MODEL_DIR = PROJECT_ROOT / "models" / "teachable_machine"
MODEL_PATH = str(MODEL_DIR / "keras_model.h5")
LABELS_PATH = str(MODEL_DIR / "labels.txt")
IMG_SIZE = 224
OFFSET = 20
