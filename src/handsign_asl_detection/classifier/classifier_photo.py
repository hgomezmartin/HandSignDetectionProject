"""
classifier_photo.py
--------------------

Envuelve toda la lógica necesaria para clasificar una imágen estática
con el modelo ASL

Se usa desde la sección photo de la app Streamlit

Autor: Hugo Gómez Martín
Contacto: hgm1001@alu.ubu.es
Fecha: 05/07/2025
"""

from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

from handsign_asl_detection.config import IMG_SIZE, TEACHABLE_TFL_DIR


# Cacheadores
@st.cache_resource
def load_tflite(model_path: str | Path):
    """
    Carga una sola ves un modelo TFLite
    """
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    inp_det = interpreter.get_input_details()
    out_det = interpreter.get_output_details()
    return interpreter, inp_det, out_det


@st.cache_resource
def load_h5(model_path: str | Path):
    """
    carga un modleo .h5 de jkeras. no se usa aún en la app, pero queda
    listo por si en un futuro migramos a servidores capaces de mover bien
    un .h5
    """
    from tensorflow.keras.models import load_model

    model = load_model(str(model_path))
    return model


# Etiquetas
with open((TEACHABLE_TFL_DIR / "labels.txt")) as f:
    labels = [ln.strip() for ln in f]


def classify_image(img_bgr: np.ndarray, model_path: str | Path) -> tuple[str, float]:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    # Pre-procesado común: resize 224×224, RGB, [0-1]
    img_resized = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    # Convertimos de BGR a RGB y a float32 normalizado
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    inp = np.expand_dims(img_rgb, 0)

    # Inferencia
    if model_path.suffix == ".tflite":
        # ruta a .tflite
        interpreter, inp_det, out_det = load_tflite(model_path)
        # aseguramos un dtype correcto
        inp = inp.astype(inp_det[0]["dtype"])
        interpreter.set_tensor(inp_det[0]["index"], inp)  # copiamos al tensor
        interpreter.invoke()  # inferencia
        preds = interpreter.get_tensor(out_det[0]["index"])[0]  # array de una dimension para predecir
    else:  # .h5 (para un futuro)
        model = load_h5(model_path)
        preds = model.predict(inp, verbose=0)[0]

    idx = int(np.argmax(preds))  # indice de la probabilidad maxima
    label = labels[idx]  # nombre de la clase
    conf = float(preds[idx] * 100)  # porcentaje de 0-100
    return label, conf
