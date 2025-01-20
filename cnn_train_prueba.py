import os

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Ajusta estas constantes a tu gusto
IMG_SIZE = 300
EPOCHS = 10  # Número de épocas de entrenamiento
BATCH_SIZE = 32  # Tamaño del batch
DATA_DIR = "Data"  # Directorio donde guardamos las carpetas A, B, C... 0, 1, 2...
MODEL_PATH = "Model/my_cnn_model.h5"  # Ruta donde se guardará el modelo


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
    """
    Crea y devuelve un modelo CNN básico con Keras.
    """
    model = Sequential()

    # Capa convolucional 1
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))

    # Capa convolucional 2
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    # Capa convolucional 3 (opcional, puedes añadir más)
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    # Aplanado
    model.add(Flatten())

    # Capa densa
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    # Capa de salida
    model.add(Dense(num_classes, activation='softmax'))

    # Compilación
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def main():
    # 1. Carga el dataset
    X, y, classes = load_dataset(DATA_DIR, IMG_SIZE)
    print(f"Dataset cargado. Total imágenes: {len(X)}, Clases: {classes}")

    # 2. Mezclar y dividir en train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Conjunto de entrenamiento: {X_train.shape}, Conjunto de validación: {X_val.shape}")

    # 3. Construir el modelo
    input_shape = (IMG_SIZE, IMG_SIZE, 3)
    num_classes = len(classes)
    model = build_cnn_model(input_shape, num_classes)
    model.summary()

    # 4. Entrenar el modelo
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    # 5. Guardar el modelo
    model.save(MODEL_PATH)
    print(f"Modelo guardado en: {MODEL_PATH}")

    # 6. Evaluar en validación (o test si tuviera)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Precisión en validación: {val_acc * 100:.2f}%, Pérdida: {val_loss:.4f}")

    # 7. Guardamos también el mapeo de clases a un archivo .txt
    with open("Model/class_labels.txt", "w") as f:
        for cls_name in classes:
            f.write(f"{cls_name}\n")
    print("Se ha guardado class_labels.txt con las clases.")


if __name__ == "__main__":
    main()
