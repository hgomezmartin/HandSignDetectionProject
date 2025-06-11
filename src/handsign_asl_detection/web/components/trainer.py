from __future__ import annotations

import io
import pathlib
import tempfile
import time
import zipfile

import streamlit as st
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

from handsign_asl_detection.config import IMG_SIZE
from handsign_asl_detection.model_creation.cnn_build import build_cnn_model

TMP = pathlib.Path(tempfile.gettempdir()) / "train_v2_tmp"


def trainer_v2_section():
    st.header("🧪 Entrenar modelo")

    # 1. Elegir carpeta dataset
    data_dir = st.text_input("Ruta a la carpeta de datos "
                             "(subcarpetas por clase). Ej:  /home/pi/dataset")
    if not data_dir:
        st.info("Introduce la ruta de tu dataset para continuar…")
        return
    data_dir = pathlib.Path(data_dir)
    if not data_dir.exists():
        st.error("La ruta no existe.")
        return

    # 2. Parámetros
    epochs = st.slider("Epochs", 5, 50, 10)
    batch = st.slider("Batch size", 4, 64, 8, step=4)
    lr = st.number_input("Learning rate", 1e-5, 1e-1, 1e-3, format="%.5f")

    if st.button("🚀 Entrenar"):
        # Generadores
        gen = ImageDataGenerator(
            rescale=1 / 255., validation_split=0.2,
            rotation_range=15, zoom_range=0.1,
            width_shift_range=0.1, height_shift_range=0.1,
            horizontal_flip=True
        )
        train = gen.flow_from_directory(
            data_dir, target_size=(IMG_SIZE, IMG_SIZE),
            subset="training", class_mode="categorical",
            batch_size=batch, shuffle=True)
        val = gen.flow_from_directory(
            data_dir, target_size=(IMG_SIZE, IMG_SIZE),
            subset="validation", class_mode="categorical",
            batch_size=batch, shuffle=False)

        st.write(f"Clases detectadas: {train.class_indices}")

        with st.spinner("Entrenando…"):
            model = build_cnn_model((IMG_SIZE, IMG_SIZE, 3), len(train.class_indices))
            model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                          loss="categorical_crossentropy",
                          metrics=["accuracy"])
            model.fit(train, epochs=epochs, validation_data=val, verbose=1)

        # Guardar artefactos en tmp
        ts = int(time.time())
        h5 = TMP / f"model_{ts}.h5"
        tfl = TMP / f"model_{ts}_fp32.tflite"
        lbl = TMP / f"labels_{ts}.txt"

        model.save(h5)
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        tfl.write_bytes(conv.convert())
        lbl.write_text("\n".join(sorted(train.class_indices, key=train.class_indices.get)))

        # Empaquetar ZIP
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(h5, "model.h5")
            z.write(tfl, "model_fp32.tflite")
            z.write(lbl, "labels.txt")
        buf.seek(0)

        st.success("✅ Entrenamiento finalizado")
        st.download_button("Descargar paquete (.zip)",
                           data=buf,
                           file_name=f"model_{ts}.zip",
                           mime="application/zip")
