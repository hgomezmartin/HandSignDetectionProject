

from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

from handsign_asl_detection.config import IMG_SIZE, TEACHABLE_TFL_DIR


# Cacheadores


@st.cache_resource
def load_tflite(model_path: str | Path):
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    inp_det = interpreter.get_input_details()
    out_det = interpreter.get_output_details()
    return interpreter, inp_det, out_det


@st.cache_resource
def load_h5(model_path: str | Path):
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
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    inp = np.expand_dims(img_rgb, 0)

    if model_path.suffix == ".tflite":
        interpreter, inp_det, out_det = load_tflite(model_path)
        inp = inp.astype(inp_det[0]["dtype"])
        interpreter.set_tensor(inp_det[0]["index"], inp)
        interpreter.invoke()
        preds = interpreter.get_tensor(out_det[0]["index"])[0]
    else:  # .h5
        model = load_h5(model_path)
        preds = model.predict(inp, verbose=0)[0]

    idx = int(np.argmax(preds))
    label = labels[idx]
    conf = float(preds[idx] * 100)
    return label, conf
