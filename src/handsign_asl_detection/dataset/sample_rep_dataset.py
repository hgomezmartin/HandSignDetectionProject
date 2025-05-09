"""
sample_rep_dataset.py
Crea una carpeta rep_data/ con N imágenes aleatorias por clase
partiendo de tu dataset completo.
"""
import random
import shutil
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]
DATASET_ROOT = PROJECT_ROOT / "data/Data_disordered"  # 26 carpetas A, B, C...
REP_ROOT = PROJECT_ROOT / "data/rep_data"
SAMPLES_PER_CLASS = 8

REP_ROOT.mkdir(parents=True, exist_ok=True)

for class_dir in DATASET_ROOT.iterdir():
    imgs = list(class_dir.glob("*.jpg"))
    random.shuffle(imgs)
    for img in imgs[:SAMPLES_PER_CLASS]:
        # Copia y conserva nombre: A_0001.jpg, etc.
        shutil.copy(img, REP_ROOT / f"{class_dir.name}_{img.name}")
print("Representative dataset creado:", len(list(REP_ROOT.glob('*.jpg'))), "imágenes")
