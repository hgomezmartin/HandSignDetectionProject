import json
import os
import random

import cv2
import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization, GlobalAveragePooling2D,
    Conv2D, MaxPooling2D, Dense, Dropout
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2

from handsign_asl_detection.config import RANDOM_SEARCH_DIR, DISORDERED_DATADIR, IMG_SIZE, SEED, EPOCHS

# --- Constantes ---
DATA_DIR = DISORDERED_DATADIR
MODEL_PATH = RANDOM_SEARCH_DIR / "my_cnn_model.h5"  # …/models/augmented/my_cnn_model.h5
LABELS_PATH = RANDOM_SEARCH_DIR / "class_labels.txt"  # …/models/augmented/class_labels.txt
PLOTS_DIR = RANDOM_SEARCH_DIR / "plots"
HPARAMS_JSON_PATH = RANDOM_SEARCH_DIR / "hparams" / "best_hparams.json"
HPARAMS_HTML_PATH = RANDOM_SEARCH_DIR / "hparams" / "best_hparams.html"

# Establecemos esras semillas para la reproducibilidad
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_dataset(data_dir=DISORDERED_DATADIR, img_size=IMG_SIZE):
    """
    Carga las imágenes del directorio 'Data' donde cada subcarpeta es una clase (A, B, C, etc.)
    Devuelve arrays de numpy para X (imágenes) e y (etiquetas).
    """
    labels = []
    images = []

    # Lista de carpetas (clases)
    classes = sorted(os.listdir(data_dir))
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    for cls_name in classes:
        cls_folder = os.path.join(data_dir, cls_name)
        if not os.path.isdir(cls_folder):
            continue

        for img_name in os.listdir(cls_folder):
            img_path = os.path.join(cls_folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (img_size, img_size))
            img = np.array(img, dtype=np.float32)
            img = img / 255.0

            images.append(img)
            labels.append(class_to_idx[cls_name])

    images = np.array(images)
    labels = np.array(labels)
    return images, labels, classes


def build_model(hp, input_shape, num_classes):
    model = Sequential()

    # Hiperparámetros para la activación #

    dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.7, step=0.1)
    lr = hp.Choice('learning_rate', values=[0.0001, 0.001])
    ks = hp.Choice('kernel_size', values=[3, 5])

    # Capa convolucional 1
    model.add(Conv2D(filters=32, kernel_size=(ks, ks), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    # Capa convolucional 2
    model.add(Conv2D(filters=64, kernel_size=(ks, ks), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    # Capa convolucional 3
    model.add(Conv2D(filters=128, kernel_size=(ks, ks), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    # Capa convolucional 4
    model.add(Conv2D(filters=256, kernel_size=(ks, ks), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    # Global Average Pooling
    model.add(GlobalAveragePooling2D())

    # Regularización l2
    l2_reg = hp.Float('l2_reg', min_value=0.000001, max_value=0.001, sampling='log')

    # Capa densa intermedia
    model.add(Dense(512, activation='relu', kernel_regularizer=l2(l2_reg)))

    # Dropout
    model.add(Dropout(dropout_rate))

    # Capa de salida
    model.add(Dense(num_classes, activation='softmax'))

    # Learning rate
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def main():
    # 1. Generadores de datos con validación (20%)
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.2
    )

    train_generator = datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32,  # Aumentar dependiendo de la GPU usada
        class_mode='sparse',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32,
        class_mode='sparse',
        subset='validation'
    )

    num_classes = len(train_generator.class_indices)
    print(f"Clases encontradas (en total {num_classes}): {train_generator.class_indices}")

    # 1.5. verificamos si esta utilizando la gpu
    print("Dispositivos GPU disponibles:", tf.config.list_physical_devices('GPU'))

    # 2. Definimos RandomSearch
    tuner = kt.RandomSearch(
        hypermodel=lambda hp: build_model(
            hp,
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            num_classes=num_classes
        ),
        objective='val_accuracy',
        max_trials=2,
        executions_per_trial=1,
        overwrite=True,
        directory='Model/random_search/tuner_dir',
        project_name='my_cnn_randomSearch'
    )

    # 3. Lanzamos la búsqueda de hiperparámetros
    tuner.search(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)]
    )

    # 4. Obtenemos los mejores hiperparámetros
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Mejores hiperparámetros encontrados:")
    for param, value in best_hp.values.items():
        print(f"  {param}: {value}")

    # Guardar hiperparámetros en JSON
    os.makedirs(os.path.dirname(HPARAMS_JSON_PATH), exist_ok=True)
    with open(HPARAMS_JSON_PATH, "w") as f:
        json.dump(best_hp.values, f, indent=4)
    print(f"Hiperparámetros guardados en: {HPARAMS_JSON_PATH}\n")

    # Guardar en HTML
    html_str = """
    <html>
    <head><meta charset="UTF-8"><title>Mejores Hiperparámetros</title></head>
    <body>
      <h2>Mejores Hiperparámetros Encontrados</h2>
      <table border="1">
        <tr><th>Hiperparámetro</th><th>Valor</th></tr>
    """

    for param, value in best_hp.values.items():
        html_str += f"<tr><td>{param}</td><td>{value}</td></tr>"

    html_str += """
      </table>
    </body>
    </html>
    """

    with open(HPARAMS_HTML_PATH, "w") as f:
        f.write(html_str)

    print(f"Hiperparámetros guardados en HTML: {HPARAMS_HTML_PATH}\n")

    # 5. Construimos y entrenamos el modelo con mejores hiperparámetros
    best_model = tuner.hypermodel.build(best_hp)
    best_model.summary()

    history = best_model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        verbose=1
    )

    # 6. Guardar el mejor modelo
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    best_model.save(MODEL_PATH)
    print(f"Modelo guardado en: {MODEL_PATH}")

    # 7. Guardar mapeo de clases
    with open(LABELS_PATH, "w") as f:
        for cls_name in sorted(train_generator.class_indices, key=train_generator.class_indices.get):
            f.write(f"{cls_name}\n")
    print("Se ha guardado class_labels.txt con las clases.")

    # 8. Evaluar
    val_loss, val_acc = best_model.evaluate(validation_generator, verbose=0)
    print(f"Precisión en validación: {val_acc * 100:.2f}%, Pérdida: {val_loss:.4f}")

    # 9. Imprimimos las graficas de funcion de pérdida y de exactitud
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
