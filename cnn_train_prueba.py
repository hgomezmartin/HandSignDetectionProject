import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Constantes
IMG_SIZE = 224
EPOCHS = 10  # Número de épocas de entrenamiento
BATCH_SIZE = 32  # Tamaño del batch
DATA_DIR = "Data"  # Directorio donde guardamos las carpetas A, B, C...
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

    # Aplanado
    # model.add(Flatten())

    model.add(GlobalAveragePooling2D())

    # Capa densa
    model.add(Dense(units=512, activation='relu'))
    model.add(Dropout(0.5))

    # Capa de salida
    model.add(Dense(num_classes, activation='softmax'))

    # Compilación
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def main():
    # 1. Configurar el ImageDataGenerator con validación (20% de los datos)
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.2
    )

    train_generator = datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='sparse',  # Usamos etiquetas enteras
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        subset='validation'
    )

    print(f"Clases encontradas: {train_generator.class_indices}")

    # 2. Construir el modelo
    input_shape = (IMG_SIZE, IMG_SIZE, 3)
    num_classes = len(train_generator.class_indices)
    model = build_cnn_model(input_shape, num_classes)
    model.summary()

    # 3. Entrenar el modelo usando los generadores
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        verbose=1
    )

    # 4. Guardar el modelo
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Modelo guardado en: {MODEL_PATH}")

    # 5. Evaluar en validación / test si tuviera
    val_loss, val_acc = model.evaluate(validation_generator, verbose=0)
    print(f"Precisión en validación: {val_acc * 100:.2f}%, Pérdida: {val_loss:.4f}")

    # 6. Guardamos también el mapeo de clases a un archivo .txt
    with open("Model/class_labels.txt", "w") as f:
        # Escribimos las clases ordenadas por su índice
        for cls_name in sorted(train_generator.class_indices, key=train_generator.class_indices.get):
            f.write(f"{cls_name}\n")
    print("Se ha guardado class_labels.txt con las clases.")

    # 7. Imprimimos las graficas de funcion de pérdida y de exactitud
    # Gráfico de la función de pérdida
    plt.figure(figsize=(8, 6))
    plt.title("Loss")
    plt.plot(history.epoch, history.history["loss"], label="Training Loss")
    plt.plot(history.epoch, history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
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
    plt.show()

if __name__ == "__main__":
    main()
