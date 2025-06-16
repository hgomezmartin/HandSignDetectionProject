from __future__ import annotations

import io
import pathlib
import shutil
import tempfile
import time
import zipfile

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

from handsign_asl_detection.config import IMG_SIZE, MAX_IMAGES_DS
from handsign_asl_detection.model_creation.cnn_build import build_cnn_model

TMP_PATH = pathlib.Path(tempfile.gettempdir()) / "train_v2_tmp"


# helpers 
def _save_images(cls: str, files, root: pathlib.Path, limit=MAX_IMAGES_DS):
    d = root / cls
    d.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(files[:limit]):
        arr = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            cv2.imwrite(str(d / f"{i:04d}.jpg"), img)


def _train(dir_: pathlib.Path, epochs: int, batch: int, lr: float):
    gen = ImageDataGenerator(rescale=1 / 255., validation_split=0.2)
    tr = gen.flow_from_directory(
        dir_, (IMG_SIZE, IMG_SIZE),
        batch_size=batch, class_mode="sparse",
        subset="training", seed=42)
    val = gen.flow_from_directory(
        dir_, (IMG_SIZE, IMG_SIZE),
        batch_size=batch, class_mode="sparse",
        subset="validation", seed=42)

    model = build_cnn_model((IMG_SIZE, IMG_SIZE, 3), len(tr.class_indices))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(tr, epochs=epochs, validation_data=val, verbose=1)
    return model, tr.class_indices


def trainer_v2_section():
    st.header("🛠️ Entrenar modelo (subida de imágenes)")

    n = st.number_input("Número de clases", 2, 20, 2)
    uploads: dict[str, list] = {}

    for i in range(int(n)):
        with st.expander(f"Clase {i + 1}", expanded=True):

            name = st.text_input("Nombre", key=f"name_{i}").strip()
            files = st.file_uploader("Imágenes (máx 500)",
                                     ["jpg", "jpeg", "png"],
                                     accept_multiple_files=True,
                                     key=f"files_{i}")
            if name and files:
                uploads[name] = files
                st.caption(f"{len(files)} archivo(s)")

    col1, col2, col3 = st.columns(3)
    epochs = col1.slider("Epochs", 5, 50, 10)
    batch = col2.slider("Batch size", 4, 128, 8, step=4)
    lr = col3.number_input("Learning rate", 1e-5, 1e-1, 1e-3, format="%.5f")

    if st.button("🚀 Entrenar"):
        if len(uploads) < 2:
            st.error("Necesitas al menos dos clases con imágenes.")
            return

        # preparar dataset temporal limpio
        if TMP_PATH.exists(): shutil.rmtree(TMP_PATH)
        TMP_PATH.mkdir(parents=True)
        for cls, files in uploads.items():
            _save_images(cls, files, TMP_PATH)

        st.success(f"📂 Dataset construido con clases: {', '.join(uploads.keys())}")

        # entrenamiento
        with st.spinner("Entrenando… esto puede tardar unos minutos"):
            model, idx_map = _train(TMP_PATH, epochs, batch, lr)

        # artefactos
        ts = int(time.time())
        h5_path = TMP_PATH / "model.h5"
        tfl_path = TMP_PATH / "model_fp32.tflite"
        lbl_path = TMP_PATH / "labels.txt"

        model.save(h5_path)
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        tfl_path.write_bytes(conv.convert())
        lbl_path.write_text("\n".join(sorted(idx_map, key=idx_map.get)))

        # empaquetar ZIP en memoria
        ZIP_buffer = io.BytesIO()
        with zipfile.ZipFile(ZIP_buffer, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(h5_path, "model.h5")
            z.write(tfl_path, "model_fp32.tflite")
            z.write(lbl_path, "labels.txt")
        ZIP_buffer.seek(0)

        st.success("✅ Entrenamiento finalizado")
        st.download_button("Descargar paquete (.zip)",
                           data=ZIP_buffer,
                           file_name=f"hand_sign_model_{ts}.zip",
                           mime="application/zip")
