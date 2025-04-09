import json
import os

import cv2
import keras_tuner as kt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- Constantes ---
IMG_SIZE = 224
EPOCHS = 70
DATA_DIR = "Data"
MODEL_PATH = "Model/RS/my_cnn_model_tuned.h5"
HPARAMS_JSON_PATH = "Model/RS/best_hparams.json"


# --- Función para cargar dataset si lo deseas manualmente (opcional) ---
def load_dataset(data_dir=DATA_DIR, img_size=IMG_SIZE):
    """
    Carga las imágenes del directorio 'Data' donde cada subcarpeta es una clase (A, B, C, 0, 1, etc.)
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


# --- Definimos la función que construye la CNN con hiperparámetros tunables ---
def build_model(hp, input_shape=(224, 224, 3), num_classes=5):
    model = Sequential()

    # Capa conv 1: número de filtros entre 32 y 256 (paso 32)
    filters_1 = hp.Int('filters_1', min_value=32, max_value=256, step=32)
    model.add(Conv2D(filters=filters_1,
                     kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    # Capa conv 2
    filters_2 = hp.Int('filters_2', min_value=32, max_value=256, step=32)
    model.add(Conv2D(filters=filters_2, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    # Capa conv 3
    filters_3 = hp.Int('filters_3', min_value=64, max_value=512, step=64)
    model.add(Conv2D(filters=filters_3, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    # Capa conv 4
    filters_4 = hp.Int('filters_4', min_value=64, max_value=512, step=64)
    model.add(Conv2D(filters=filters_4, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    # Global Average Pooling
    model.add(GlobalAveragePooling2D())

    # Capa densa intermedia
    dense_units = hp.Int('dense_units', min_value=128, max_value=1024, step=128)
    model.add(Dense(dense_units, activation='relu'))

    # Dropout
    dropout_rate = hp.Float('dropout_rate', min_value=0.01, max_value=0.5, step=0.05)
    model.add(Dropout(dropout_rate))

    # Capa de salida
    model.add(Dense(num_classes, activation='softmax'))

    # Learning rate
    lr = hp.Choice('learning_rate', values=[0.0001, 0.0005, 0.001])

    model.compile(optimizer=Adam(learning_rate=lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def main():
    # --- 1. Preparamos generadores de datos con validación (20%) ---
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.2
    )

    train_generator = datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32,  # Puedes tunear también el batch_size si lo deseas
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
    print("Clases encontradas:", train_generator.class_indices)

    # --- 2. Definimos el RandomSearch de Keras Tuner ---
    tuner = kt.RandomSearch(
        hypermodel=lambda hp: build_model(hp,
                                          input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                          num_classes=num_classes),
        objective='val_accuracy',
        max_trials=10,  # Número de configuraciones distintas a probar
        executions_per_trial=1,  # Cuántas veces repite la misma configuración
        overwrite=True,  # Sobrescribe resultados previos
        directory='Model/RS/tuner_dir',  # Carpeta donde guardará los resultados
        project_name='my_cnn_randomSearch'
    )

    # --- 3. Lanzamos la búsqueda de hiperparámetros ---
    # Nota: añadimos EarlyStopping para no sobreentrenar
    tuner.search(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)
        ]
    )

    # --- 4. Obtenemos los mejores hiperparámetros y entrenamos el mejor modelo ---
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Mejores hiperparámetros encontrados:")
    for param, value in best_hp.values.items():
        print(f"  {param}: {value}")

    # === Guardar los hiperparámetros en un JSON ===
    os.makedirs(os.path.dirname(HPARAMS_JSON_PATH), exist_ok=True)
    with open(HPARAMS_JSON_PATH, "w") as f:
        json.dump(best_hp.values, f, indent=4)
    print(f"Hiperparámetros guardados en: {HPARAMS_JSON_PATH}\n")

    # 5. Construimos un modelo con los mejores hiperparámetros
    best_model = tuner.hypermodel.build(best_hp)
    best_model.summary()

    # Entrenamos de nuevo (opcional) con más épocas si quieres
    history = best_model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator
    )

    # --- 6. Guardar el mejor modelo ---
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    best_model.save(MODEL_PATH)
    print(f"Modelo guardado en: {MODEL_PATH}")

    # --- 7. Evaluar ---
    val_loss, val_acc = best_model.evaluate(validation_generator, verbose=0)
    print(f"Precisión en validación: {val_acc * 100:.2f}%, Pérdida: {val_loss:.4f}")

    # --- 8. Guardar mapeo de clases ---
    with open("Model/RS/class_labels.txt", "w") as f:
        for cls_name in sorted(train_generator.class_indices, key=train_generator.class_indices.get):
            f.write(f"{cls_name}\n")
    print("Se ha guardado class_labels.txt con las clases.")


if __name__ == "__main__":
    main()
