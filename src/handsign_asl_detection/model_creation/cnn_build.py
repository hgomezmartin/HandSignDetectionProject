import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2

from handsign_asl_detection.config import IMG_SIZE, BATCH_SIZE, EPOCHS, SEED, DISORDERED_DATADIR, AUGMENTED_DIR

DATA_DIR = DISORDERED_DATADIR
MODEL_PATH = AUGMENTED_DIR / "my_cnn_model.h5"  # …/models/augmented/my_cnn_model.h5
LABELS_PATH = AUGMENTED_DIR / "class_labels.txt"  # …/models/augmented/class_labels.txt
PLOTS_DIR = AUGMENTED_DIR / "plots"

# Establecemos semillas para la reproducibilidad
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_dataset(data_dir=DATA_DIR, img_size=IMG_SIZE):
    """
    Carga las imágenes del directorio 'Data' donde cada subcarpeta es una clase (A, B, C, 0, 1, etc.)
    Devuelve arrays de numpy para X (imágenes) e y (etiquetas).
    """
    labels = []
    images = []

    # Lista de carpetas (clases)
    classes = sorted(os.listdir(data_dir))

    # Mapeamos cada clase a un índice entero
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    # Recorremos cada clase y cargamos sus imágenes
    for cls_name in classes:
        cls_folder = os.path.join(data_dir, cls_name)
        if not os.path.isdir(cls_folder):
            continue

        for img_name in os.listdir(cls_folder):
            img_path = os.path.join(cls_folder, img_name)
            # Lee la imagen
            img = cv2.imread(img_path)
            if img is None:
                continue
            # Aseguramos que la imagen sea del tamaño deseado
            img = cv2.resize(img, (img_size, img_size))
            # Convertimos a array numpy
            img = np.array(img, dtype=np.float32)
            # Normalizamos a [0,1]
            img = img / 255.0

            images.append(img)
            labels.append(class_to_idx[cls_name])

    images = np.array(images)
    labels = np.array(labels)

    return images, labels, classes


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
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def main():
    # 0. Implementacion del data augmentation
    # Solo para entrenamiento

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,  # rotación ±15°
        zoom_range=0.1,  # zoom ±10%
        width_shift_range=0.1,  # desplazamiento horizontal ±10%
        height_shift_range=0.1,  # desplazamiento vertical ±10%
        brightness_range=[0.8, 1.2],  # brillo entre 80–120%
        horizontal_flip=True,  # volteo horizontal
        fill_mode='reflect',  # rellena bordes reflejando pixeles
        validation_split=0.2

    )

    # 2. Solo rescale para validación
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2
    )


    # 1. Configurar el ImageDataGenerator con validación (20% de los datos)
    '''
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2
    )'''
    train_generator = train_datagen.flow_from_directory(
    #train_generator = datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='sparse',  # Usamos etiquetas enteras
        subset='training',
        seed = 42  # fija shuffle + aug para reproducibilidad
    )

    validation_generator = val_datagen.flow_from_directory(
    #validation_generator = datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        subset='validation',
        seed = 42  # fija shuffle + aug para reproducibilidad
    )

    num_classes = len(train_generator.class_indices)
    print(f"Clases encontradas (en total {num_classes}): {train_generator.class_indices}")

    # 2. Construir el modelo
    input_shape = (IMG_SIZE, IMG_SIZE, 3)
    num_classes = num_classes
    model = build_cnn_model(input_shape, num_classes)
    model.summary()

    # 2.1 Introduccion del EarlyStopping y del ReduceLr
    earlystop_cb = EarlyStopping(
        monitor="val_accuracy",
        patience=7,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr_cb = ReduceLROnPlateau(
        monitor="val_accuracy",
        factor=0.5,
        patience=3,
        min_lr=0.000001,
        verbose=1
    )

    # 3. Entrenar el modelo usando los generadores
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=[earlystop_cb, reduce_lr_cb],
        verbose=1
    )

    # 4. Guardar el modelo
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Modelo guardado en: {MODEL_PATH}")

    # 5. Guardamos también el mapeo de clases a un archivo .txt
    with open(LABELS_PATH, "w") as f:
        # Escribimos las clases ordenadas por su índice
        for cls_name in sorted(train_generator.class_indices, key=train_generator.class_indices.get):
            f.write(f"{cls_name}\n")
    print("Se ha guardado class_labels.txt con las clases.")

    # 6. Evaluar en validación / test si tuviera
    val_loss, val_acc = model.evaluate(validation_generator, verbose=0)
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
