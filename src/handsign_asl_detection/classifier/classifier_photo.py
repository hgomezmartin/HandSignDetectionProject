import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from handsign_asl_detection.config import TEACHABLE_DIR, IMG_SIZE

MODEL_PATH = TEACHABLE_DIR / "keras_model.h5"
LABELS_PATH = TEACHABLE_DIR / "labels.txt"


model = load_model(MODEL_PATH)
with open(LABELS_PATH, encoding="utf-8") as f:
    labels = [line.strip() for line in f]


def _select_image_via_dialog() -> str | None:
    root = tk.Tk();
    root.withdraw()
    filetypes = [("Imágenes", "*.jpg *.jpeg *.png *.bmp"),
                 ("Todos los archivos", "*.*")]
    return filedialog.askopenfilename(
        title="Selecciona una imagen de tu dataset",
        filetypes=filetypes
    )


def main():
    img_path = _select_image_via_dialog()
    if not img_path:
        print("No se seleccionó ninguna imagen.");
        return

    label, conf = classify_image(img_path)
    print(f"Imagen: {img_path}")
    print(f"Clase:  {label}")
    print(f"Confianza: {conf:.1f} %")


def classify_image(img_path: str):
    """Devuelve (label, confidence) para una sola imagen."""
    img_path = Path(img_path)
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(img_path)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    preds = model.predict(np.expand_dims(img, 0), verbose=0)[0]
    idx = int(np.argmax(preds))
    return labels[idx], float(preds[idx] * 100)


if __name__ == "__main__":
    main()
