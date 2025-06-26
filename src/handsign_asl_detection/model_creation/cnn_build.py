import os
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from handsign_asl_detection.config import IMG_SIZE, BATCH_SIZE, EPOCHS, SEED, DISORDERED_DATADIR, AUGMENTED_DIR

DATA_DIR = DISORDERED_DATADIR
MODEL_PATH = AUGMENTED_DIR / "my_cnn_model.h5"
LABELS_PATH = AUGMENTED_DIR / "labels.txt"
PLOTS_DIR = AUGMENTED_DIR / "plots"

AUTOTUNE = tf.data.AUTOTUNE

# Establecemos semillas para la reproducibilidad
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def build_cnn_model(input_shape, num_classes):
    # Crea y devuelve un modelo CNN  con Keras.
    model = Sequential()

    # Capa convolucional 1
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    # Capa convolucional 2
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    # Capa convolucional 3
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    # Capa convolucional 4
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(GlobalAveragePooling2D())

    # Capa densa
    model.add(Dense(units=512, activation='relu', kernel_regularizer=l2(0.0004)))
    model.add(Dropout(0.5))

    # Capa de salida
    model.add(Dense(num_classes, activation='softmax'))

    # Compilación
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='sparse_categorical_en nuestr',
                  metrics=['accuracy'])

    return model


def main():
    # 1. Carga de imágenes en tf.data.Dataset (80 / 20)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.20,
        subset="training",
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="int"
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

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"Clases ({num_classes}): {class_names}")

    # 1.2 capas de normalizado + augmentación (en GPU)
    data_aug = Sequential([
        layers.Rescaling(1. / 255),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.10),
        layers.RandomTranslation(0.10, 0.10),
        layers.RandomContrast(0.2),
        layers.RandomFlip("horizontal")
    ])

    # train dataset -> aug + prefetch
    train_ds_aug = (train_ds
                    .map(lambda x, y: (data_aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
                    .cache()
                    .prefetch(AUTOTUNE))

    val_ds_norm = (val_ds
                   .map(lambda x, y: (x / 255.0, y))
                   .cache()
                   .prefetch(AUTOTUNE))

    # 2. Construir el modelo
    input_shape = (IMG_SIZE, IMG_SIZE, 3)
    num_classes = num_classes
    model = build_cnn_model(input_shape, num_classes)
    model.summary()

    # 2.1 Introduccion del EarlyStopping y del ReduceLr
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
        min_lr=0.000001,
        verbose=1
    )

    # 3. Entrenar el modelo usando los generadores
    history = model.fit(
        train_ds_aug,
        epochs=EPOCHS,
        validation_data=val_ds_norm,
        callbacks=[earlystop_cb, reduce_lr_cb],
        verbose=1
    )

    # 4. Guardar el modelo
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Modelo guardado en: {MODEL_PATH}")

    # 5. Guardamos también el mapeo de clases a un archivo .txt
    with open(LABELS_PATH, "w") as f:
        f.write("\n".join(class_names))
    print("Se ha guardado en labels.txt con las clases.")

    # 6. Evaluar en validación / test si tuviera
    val_loss, val_acc = model.evaluate(val_ds_norm, verbose=0)
    print(f"Precisión en validación: {val_acc * 100:.2f}%, Pérdida: {val_loss:.4f}")

    # 7. Imprimimos las graficas de funcion de pérdida y de exactitud
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Gráfico de la función de pérdida
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

    # Gráfico de la exactitud
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


if __name__ == "__main__":
    main()
