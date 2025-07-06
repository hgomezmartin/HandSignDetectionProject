"""
trainer.py
-----------
Sección Streamlit que permite al usuario entrenar su propio modelo
subiendo un dataset que no supere las 26 clases.

Flujo
1. El usuario introduce el numero de clases, nombre y las imagenes de
cada clase ajustando hiperparámetros (epochs, batchsize y learning rate)
2. Se guardan las imagenes en una carpeta temporal.
3. Se crea un dataser con aumentación.
4. Se construye  entrena la CNN
5. al final se descarga un .zip con labels.txt el modelo .h5 y optimizado
.tflite FP32

Autor: Hugo Gómez Martín
Contacto: hgm1001@alu.ubu.es
Fecha: 05/07/2025
"""

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
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import layers, Sequential

from handsign_asl_detection.config import IMG_SIZE
from handsign_asl_detection.model_creation.cnn_build import build_cnn_model

# constantes
TMP_PATH = pathlib.Path(tempfile.gettempdir()) / "train_v3_tmp"  # carpeta temporal
AUTOTUNE = tf.data.AUTOTUNE  # paralelismo tf.data


# helpers
def _save_images(cls: str, files, root: pathlib.Path, limit=None):
    """Guarda imágenes subidas por el usuario en carpetas por clase convertidas a JPG"""
    d = root / cls
    d.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(files[:limit]):  # corte si hay limit
        arr = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            cv2.imwrite(str(d / f"{i:04d}.jpg"), img)  # XXXX.jpg


def tf_dataset(dir_: pathlib.Path, batch: int, seed: int = 42):
    """Crea train/val tf.data.Dataset con augmentación y normalizado."""

    # Carga de datos con "image_dataset_from_directory"
    raw_train = tf.keras.utils.image_dataset_from_directory(
        dir_,
        validation_split=0.20,  # 80/20 split de datos
        subset="training",
        seed=seed,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch,
        label_mode="int"  # Etiquetas enteras 0...N-1
    )
    raw_val = tf.keras.utils.image_dataset_from_directory(
        dir_,
        validation_split=0.20,
        subset="validation",
        seed=seed,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch,
        label_mode="int",
    )

    class_names = raw_train.class_names  # A, B, C ...
    num_classes = len(class_names)
    print(f"Clases ({num_classes}): {class_names}")

    # 1.1 Aumentación (augmentation) on the fly
    data_aug = Sequential([
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.10),
        layers.RandomTranslation(0.10, 0.10),
        layers.RandomContrast(0.2),
    ])

    # Se aplican aumentos solo a train, se cachea y prefetch
    train_ds = (raw_train
                .map(lambda x, y: (data_aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
                .cache()
                .prefetch(AUTOTUNE))

    # solo cache y prefech
    val_ds = raw_val.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, class_names


# Streamlit
def trainer_v2_section():
    """
    Sección de entrenamiento interactivo
    """
    st.header("🛠️ Entrenar modelo (subida de imágenes)")

    # 1) Carga de archivos
    n = st.number_input("Número de clases", 2, 26, 2)  # mínimo 2
    uploads: dict[str, list] = {}

    for i in range(int(n)):
        with st.expander(f"Clase {i + 1}/26", expanded=True):
            name = st.text_input("Nombre", key=f"name_{i}").strip()
            files = st.file_uploader("Imágenes",
                                     ["jpg", "jpeg", "png"],
                                     accept_multiple_files=True,
                                     key=f"files_{i}")
            if name and files:
                uploads[name] = files
                st.caption(f"{len(files)} archivo(s)")

    # Hiperparámetros básicos
    col1, col2, col3 = st.columns(3)
    epochs = col1.slider("Epochs", 5, 50, 10)
    batch = col2.slider("Batch size", 4, 128, 8, step=4)
    lr = col3.select_slider(
        "Learning rate",
        options=[1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6],
        value=1e-3
    )

    # Botón entrenar
    if st.button("🚀 Entrenar"):

        # validación minima a la hora de introducir clases
        if len(uploads) < 2:
            st.error("Necesitas al menos dos clases con imágenes.")
            return

        # Construimos dataset temporal
        if TMP_PATH.exists():
            shutil.rmtree(TMP_PATH)
        TMP_PATH.mkdir(parents=True)
        for cls, files in uploads.items():
            _save_images(cls, files, TMP_PATH)

        st.success(f"📂 Dataset listo con clases: "
                   f"{', '.join(uploads.keys())}")

        # Entrenamos
        with st.spinner("Entrenando... esto puede tardar unos minutos"):
            train_ds, val_ds, class_names = tf_dataset(TMP_PATH, batch)

            # Creamos el modleo
            input_shape = (IMG_SIZE, IMG_SIZE, 3)
            model = build_cnn_model(input_shape,
                                    len(class_names),
                                    lr=lr)
            model.summary()

            # callbacks
            earlystop_cb = EarlyStopping(
                monitor="val_accuracy",  # metrica a vigilar
                patience=10,  # epochs sin mejora
                restore_best_weights=True,  # vuelve al mejor checkpoint
                verbose=1
            )

            reduce_lr_cb = ReduceLROnPlateau(
                monitor="val_accuracy",
                factor=0.5,  # divide el LR entre 2
                patience=3,
                min_lr=1e-6,
                verbose=1
            )

            # barra y gráfico en vivo
            prog = st.progress(0.0)  # barra pocentaje
            chart_area = st.empty()  # gráfica

            class LivePlot(tf.keras.callbacks.Callback):
                def __init__(self, total_epochs):
                    super().__init__()
                    self.total = total_epochs
                    self.tr_acc, self.val_acc = [], []

                def on_epoch_end(self, epoch, logs=None):
                    logs = logs or {}
                    # guarda métricas de la época
                    self.tr_acc.append(logs.get("accuracy", 0.0))
                    self.val_acc.append(logs.get("val_accuracy", 0.0))

                    # actualiza barra de progreso
                    prog.progress((epoch + 1) / self.total)

                    # actualiza gráfico
                    with chart_area:
                        st.line_chart({
                            "train_acc": self.tr_acc,
                            "val_acc": self.val_acc,
                        })

            # Entrenamos con model.fit
            model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs,
                verbose=1,
                callbacks=[earlystop_cb, reduce_lr_cb,
                           LivePlot(epochs)]
            )

        # 4) Guardar artefactos
        ts = int(time.time())  # con una marca temporal en el nombre
        h5_path = TMP_PATH / "model.h5"
        tfl_path = TMP_PATH / "model_fp32.tflite"
        lbl_path = TMP_PATH / "labels.txt"

        model.save(h5_path)
        # Convertimos a TFLite FP32
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        tfl_path.write_bytes(conv.convert())
        # Labels
        lbl_path.write_text("\n".join(class_names))

        # empaquetar ZIP en memoria
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(h5_path, "model.h5")
            z.write(tfl_path, "model_fp32.tflite")
            z.write(lbl_path, "labels.txt")
        buf.seek(0)

        # Descargamos con el download button
        st.success("✅ Entrenamiento finalizado")
        st.download_button("Descargar paquete (.zip)",
                           data=buf,
                           file_name=f"hand_sign_model_{ts}.zip",
                           mime="application/zip")
