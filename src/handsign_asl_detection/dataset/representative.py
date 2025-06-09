import random
import shutil
from pathlib import Path

from handsign_asl_detection.config import (
    DISORDERED_DATADIR, REPRESENTATIVE_DATADIR, SAMPLES_PER_CLASS
)

DATA_DIR = DISORDERED_DATADIR  # 26 carpetas A, B, C…
REP_DIR = REPRESENTATIVE_DATADIR  # carpeta de salida


def build_representative():
    """Genera el dataset representativo y devuelve (ruta, total_imgs)."""
    # Limpiar (si existe) y recrear carpeta
    if REP_DIR.exists():
        shutil.rmtree(REP_DIR)
    REP_DIR.mkdir(parents=True, exist_ok=True)

    for class_dir in Path(DATA_DIR).iterdir():
        if not class_dir.is_dir():
            continue

        # Subcarpeta destino para la clase
        dst_cls = REP_DIR / class_dir.name
        dst_cls.mkdir(exist_ok=True)

        # Copiar imágenes aleatorias
        imgs = list(class_dir.glob("*.jpg"))
        random.shuffle(imgs)
        for img in imgs[:SAMPLES_PER_CLASS]:
            shutil.copy2(img, dst_cls / img.name)

    total_imgs = sum(1 for _ in REP_DIR.glob("**/*.jpg"))
    print(f"Dataset representativo creado en: {REP_DIR}")
    print(f"Estructura: {[d.name for d in REP_DIR.iterdir() if d.is_dir()]}")
    print(f"Total imágenes: {total_imgs} ({SAMPLES_PER_CLASS} por clase)")
    return REP_DIR, total_imgs


if __name__ == "__main__":
    build_representative()
