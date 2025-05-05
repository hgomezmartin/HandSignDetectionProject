import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = "Model/Augmented_vs_NotAugmented/Model_Augmented/my_cnn_model.h5"
LABELS_PATH = "Model/Augmented_vs_NotAugmented/Model_Augmented/class_labels.txt"

# carga modelo + etiquetas
model = load_model(MODEL_PATH)
with open(LABELS_PATH, encoding="utf-8") as f:
    labels = [l.strip() for l in f]

# abre diálogo de selección
root = tk.Tk()
root.withdraw()  # sin ventana raíz
filetypes = [("Imágenes", "*.jpg *.jpeg *.png *.bmp"), ("Todos los archivos", "*.*")]
img_path = filedialog.askopenfilename(
    title="Selecciona una imagen de tu dataset",
    filetypes=filetypes)

if not img_path:
    print("No se seleccionó ninguna imagen.")
    exit()

# lee y procesa la imagen
IMG_SIZE = 224
img = cv2.imread(img_path)  # BGR
if img is None:
    print("Error: No se pudo abrir la imagen.")
    exit()

img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR → RGB
img = img.astype(np.float32) / 255.0
input_tensor = np.expand_dims(img, 0)  # (1,224,224,3)

# inferencia
preds = model.predict(input_tensor, verbose=0)
idx = int(np.argmax(preds))
conf = preds[0][idx] * 100

print(f"Imagen:  {img_path}")
print(f"Clase:   {labels[idx]}")
print(f"Confianza: {conf:.1f} %")
