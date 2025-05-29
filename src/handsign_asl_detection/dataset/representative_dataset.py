"""
sample_rep_dataset_mod.py
Crea una carpeta rep_data/ con subcarpetas por clase (A, B, C...)
y N imágenes aleatorias por clase.
"""
import random
import shutil
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]
DATASET_ROOT = PROJECT_ROOT / "data/disordered"  # 26 carpetas A, B, C...
REP_ROOT = PROJECT_ROOT / "data/rep_data"
SAMPLES_PER_CLASS = 8

# Crear carpeta principal y limpiar existente (opcional)
if REP_ROOT.exists():
    shutil.rmtree(REP_ROOT)
REP_ROOT.mkdir(parents=True, exist_ok=True)

for class_dir in DATASET_ROOT.iterdir():
    if not class_dir.is_dir():
        continue

    # Crear subcarpeta para la clase (A, B, C...)
    class_rep_dir = REP_ROOT / class_dir.name
    class_rep_dir.mkdir(exist_ok=True)

    # Copiar imágenes
    imgs = list(class_dir.glob("*.jpg"))
    random.shuffle(imgs)
    for img in imgs[:SAMPLES_PER_CLASS]:
        shutil.copy(img, class_rep_dir / img.name)

# Verificación final
total_imgs = sum(1 for _ in REP_ROOT.glob("**/*.jpg"))
print(f"Dataset representativo creado en: {REP_ROOT}")
print(f"Estructura: {[d.name for d in REP_ROOT.iterdir() if d.is_dir()]}")
print(f"Total imágenes: {total_imgs} ({SAMPLES_PER_CLASS} por clase)")
