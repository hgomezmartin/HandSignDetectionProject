"""
config.py
---------
Punto único de configuración para todo el proyecto:
rutas, hiperparámetros, seeds, etc.
"""

from pathlib import Path

# RUTAS

# Directorio raíz del proyecto  (<repo>/src/handsign_asl_detection/config.py → subimos 2 niveles)
ROOT = Path(__file__).resolve().parents[2]

# Datos
DATA_DIR = ROOT / "data"
ORDERED_DATADIR = DATA_DIR / "ordered"
DISORDERED_DATADIR = DATA_DIR / "disordered"
REPRESENTATIVE_DATADIR = DATA_DIR / "representative"

# Modelos
MODELS_DIR = ROOT / "models"
AUGMENTED_DIR = MODELS_DIR / "augmented"
NOT_AUGMENTED_DIR = MODELS_DIR / "not_augmented"
IMX_READY_DIR = MODELS_DIR / "imx_ready"
RANDOM_SEARCH_DIR = MODELS_DIR / "random_search"
TEACHABLE_DIR = MODELS_DIR / "teachable_machine"
TEACHABLE_TFL_DIR = TEACHABLE_DIR / "rpi_tflite"

# Hiperparámetros globales
# De los Modelos:
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 70
SEED = 42

# De la creacion del dataset:
IMG_SIZE_DS = 300
OFFSET = 20  # margen para el ROI de la mano
MAX_IMAGES_DS = 500  # imágenes por clase en collection.py
SAMPLES_PER_CLASS = 8
