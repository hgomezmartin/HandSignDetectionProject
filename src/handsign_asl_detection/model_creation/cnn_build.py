"""
cnn_build.py
-------------
Entrena un clasificador de signos usando EfficientNetB0 como extractor
de características congelado y una pequeña cabez densa propia

Flujo:
1. Cargamos las imagenes desde nuestro dataset desordenado "DISORDERED_DATADIR
con una division 80 train / 20 test
2. Definimos aumentos ligeros para dar robustez al modelo
3. Construimos la arquitectura de la red
4. Se entrena con "Early Stopping" y "ReduceLROnPlateau"
5. Guardamos el modelo, labels y gráficas (loss, accuracy y matriz de confusión)

Autor: Hugo Gómez Martín
Contacto: hgm1001@alu.ubu.es
Fecha: 05/07/2025
"""
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras import layers, Sequential
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import (GlobalAveragePooling2D, Dense,
                                     BatchNormalization, Dropout)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Config del proyecto
from handsign_asl_detection.config import (
    IMG_SIZE, BATCH_SIZE, EPOCHS, SEED,
    DISORDERED_DATADIR, FINAL_DIR
)

# rutas y constantes
DATA_DIR = DISORDERED_DATADIR  # Cogemos el conjunto de datos desordenado
MODEL_PATH = FINAL_DIR / "my_cnn_model.h5"  # Denominamos el modelo "my_cnn_model.h5"
LABELS_PATH = FINAL_DIR / "labels.txt"  # Establecemos el conjutno de estiquetas en un fichero de texto
PLOTS_DIR = FINAL_DIR / "plots"  # Creamos una carpeta para almacenar las gráficas

AUTOTUNE = tf.data.AUTOTUNE  # Paralelizar tf.data

# Semillas para la reproducibilidad (con el randomsearch)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# contruccion del modelo
def build_cnn_model(input_shape, num_classes, lr: float = 0.0003):
    """EfficientNet-B0 congelado + cabecera nueva."""
    base = EfficientNetB0(include_top=False,  # Quitamos cabecera
                          weights="imagenet",  # Pesos base
                          input_shape=input_shape)
    base.trainable = False  # Congelamos tod o el backbone por bug el TF-metal

    model = Sequential([
        layers.Input(shape=input_shape),
        layers.Lambda(preprocess_input),  # normalización específica EfficientNet
        base,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dropout(0.2),

        Dense(num_classes,
              activation="softmax",
              kernel_regularizer=l2(0.0001))
    ])

    # Compilamos
    model.compile(optimizer=Adam(lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


# main
def main():
    """
    entrena, guarda y genera gráficas del modelo entrenado

    """
    # 1. Carga de datos con "image_dataset_from_directory"
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.20,  # 80/20 split de datos
        subset="training",
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="int"  # Etiquetas enteras 0...N-1
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.20,
        subset="validation",
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="int"
    )

    class_names = train_ds.class_names  # A, B, C ...
    num_classes = len(class_names)
    print(f"Clases ({num_classes}): {class_names}")

    # 1.1 Aumentación (augmentation) on the fly
    aug = Sequential([
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.10),
        layers.RandomTranslation(0.10, 0.10),
        layers.RandomContrast(0.2),
    ])

    # Se aplican aumentos solo a train, se cachea y prefetch
    train_ds = (train_ds
                .map(lambda x, y: (aug(x, training=True), y),
                     num_parallel_calls=AUTOTUNE)
                .cache()
                .prefetch(AUTOTUNE))

    # solo cache y prefech
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    # 2. Modelo
    input_shape = (IMG_SIZE, IMG_SIZE, 3)  # 224, 224 y 3 canales RGB
    model = build_cnn_model(input_shape, num_classes)
    model.summary()  # impreime la arquitectura para ver el num de capas

    # 3. Callbacks
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
        min_lr=0.000001,
        verbose=1
    )

    # 4. Entrenamiento (una sola fase)
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=[earlystop_cb, reduce_lr_cb],
        verbose=1
    )

    # 5. Guardar modelo y etiquetas
    os.makedirs(os.path.dirname(FINAL_DIR), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Modelo guardado en: {MODEL_PATH}")

    with open(LABELS_PATH, "w") as f:
        f.write("\n".join(class_names))
    print("Se ha guardado labels.txt con las clases.")

    # 6. Evaluación
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"Validación: {val_acc * 100:.2f}% | Loss: {val_loss:.4f}")

    # 7. Graficas
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Gráfico de la función de pérdida (loss por epoch)
    plt.figure(figsize=(8, 6))
    plt.title("Loss")
    plt.plot(history.epoch, history.history["loss"], label="Training Loss")
    plt.plot(history.epoch, history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    loss_path = os.path.join(PLOTS_DIR, "loss.png")
    plt.savefig(loss_path, dpi=150)
    print(f"Guardado gráfico de Pérdida/Loss en: {loss_path}")
    plt.show()

    # Gráfico de la exactitud (accuracy por epoch)
    plt.figure(figsize=(8, 6))
    plt.title("Accuracy")
    plt.plot(history.epoch, history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.epoch, history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    acc_path = os.path.join(PLOTS_DIR, "accuracy.png")
    plt.savefig(acc_path, dpi=150)
    print(f"Guardado gráfico de Exactitud/Accuracy en: {acc_path}")
    plt.show()

    # Matriz de confusión
    y_true = np.concatenate([y.numpy() for _, y in val_ds])  # etiquetas reales
    y_pred_probs = model.predict(val_ds, verbose=0)  # probabilidades
    y_pred = np.argmax(y_pred_probs, axis=1)  # clases predecidas

    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=45)
    ax.set_title("Matriz de confusión")
    cm_path = PLOTS_DIR / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    print(f"Matriz de confusión guardada en: {cm_path}")
    plt.show()


# Punto de entrada
if __name__ == "__main__":
    main()
