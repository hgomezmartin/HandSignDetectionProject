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
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import layers, Sequential

from handsign_asl_detection.config import IMG_SIZE
from handsign_asl_detection.model_creation.cnn_build import build_cnn_model

# ---- constantes ----
TMP_PATH = pathlib.Path(tempfile.gettempdir()) / "train_v3_tmp"
AUTOTUNE = tf.data.AUTOTUNE


# ---------- helpers ----------
def _save_images(cls: str, files, root: pathlib.Path, limit=None):
    """Guarda imágenes subidas por el usuario en carpetas por clase."""
    d = root / cls
    d.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(files[:limit]):
        arr = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            cv2.imwrite(str(d / f"{i:04d}.jpg"), img)


# --------------------------------


def _tf_dataset(dir_: pathlib.Path, batch: int, seed: int = 42):
    """Crea train/val tf.data.Dataset con augmentación y normalizado."""
    raw_train = tf.keras.utils.image_dataset_from_directory(
        dir_,
        validation_split=0.20,
        subset="training",
        seed=seed,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch,
        label_mode="int"
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
    class_names = raw_train.class_names
    num_classes = len(class_names)
    print(f"Clases ({num_classes}): {class_names}")

    data_aug = Sequential([
        layers.Rescaling(1 / 255.),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.10),
        layers.RandomTranslation(0.10, 0.10),
        layers.RandomContrast(0.2),
        layers.RandomFlip("horizontal"),
    ])

    train_ds = (raw_train
                .map(lambda x, y: (data_aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
                .cache()
                .prefetch(AUTOTUNE))

    val_ds = (raw_val
              .map(lambda x, y: (x / 255.0, y))
              .cache()
              .prefetch(AUTOTUNE))

    return train_ds, val_ds, class_names


# --------------------- sección Streamlit ---------------------
def trainer_v2_section():
    st.header("🛠️ Entrenar modelo (subida de imágenes)")

    # 1 · Carga de archivos ---------------------------------------------------
    n = st.number_input("Número de clases", 2, 26, 2)
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

    col1, col2, col3 = st.columns(3)
    epochs = col1.slider("Epochs", 5, 50, 10)
    batch = col2.slider("Batch size", 4, 128, 8, step=4)
    lr = col3.number_input("Learning rate", 1e-5, 1e-1, 1e-3,
                           format="%.5f")

    if st.button("🚀 Entrenar"):
        if len(uploads) < 2:
            st.error("Necesitas al menos dos clases con imágenes.")
            return

        # 2 · Construir dataset temporal -------------------------------------
        if TMP_PATH.exists():
            shutil.rmtree(TMP_PATH)
        TMP_PATH.mkdir(parents=True)
        for cls, files in uploads.items():
            _save_images(cls, files, TMP_PATH)

        st.success(f"📂 Dataset listo con clases: "
                   f"{', '.join(uploads.keys())}")

        # 3 · tf.data + modelo + callbacks -----------------------------------
        with st.spinner("Entrenando… esto puede tardar unos minutos"):
            train_ds, val_ds, class_names = _tf_dataset(TMP_PATH, batch)

            input_shape = (IMG_SIZE, IMG_SIZE, 3)
            model = build_cnn_model(input_shape, len(class_names))
            model.summary()

            model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                          loss="sparse_categorical_crossentropy",
                          metrics=["accuracy"])

            earlystop_cb = EarlyStopping(
                monitor="val_accuracy",
                patience=15,
                restore_best_weights=True,
                verbose=1
            )

            reduce_lr_cb = ReduceLROnPlateau(
                monitor="val_accuracy",
                factor=0.5,
                patience=4,
                min_lr=1e-6,
                verbose=1
            )

            # barra y gráfico en vivo
            prog = st.progress(0.0)
            chart_area = st.empty()

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

            model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs,
                verbose=1,
                callbacks=[earlystop_cb, reduce_lr_cb,
                           LivePlot(epochs)]
            )

        # 4 · Guardar artefactos ---------------------------------------------
        ts = int(time.time())
        h5_path = TMP_PATH / "model.h5"
        tfl_path = TMP_PATH / "model_fp32.tflite"
        lbl_path = TMP_PATH / "labels.txt"

        model.save(h5_path)
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        tfl_path.write_bytes(conv.convert())
        lbl_path.write_text("\n".join(class_names))

        # empaquetar ZIP en memoria
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(h5_path, "model.h5")
            z.write(tfl_path, "model_fp32.tflite")
            z.write(lbl_path, "labels.txt")
        buf.seek(0)

        st.success("✅ Entrenamiento finalizado")
        st.download_button("Descargar paquete (.zip)",
                           data=buf,
                           file_name=f"hand_sign_model_{ts}.zip",
                           mime="application/zip")
