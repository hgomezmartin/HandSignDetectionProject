import json
import os
import random

import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import (
    BatchNormalization, GlobalAveragePooling2D,
    Conv2D, MaxPooling2D, Dense, Dropout
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from handsign_asl_detection.config import RANDOM_SEARCH_DIR, DISORDERED_DATADIR, IMG_SIZE, BATCH_SIZE, SEED, EPOCHS

# Constantes
DATA_DIR = DISORDERED_DATADIR
MODEL_PATH = RANDOM_SEARCH_DIR / "my_cnn_randomsearch.h5"
LABELS_PATH = RANDOM_SEARCH_DIR / "labels.txt"
PLOTS_DIR = RANDOM_SEARCH_DIR / "plots"
HPARAMS_JSON_PATH = RANDOM_SEARCH_DIR / "hparams" / "best_hparams.json"
HPARAMS_HTML_PATH = RANDOM_SEARCH_DIR / "hparams" / "best_hparams.html"
TUNER_DIR = RANDOM_SEARCH_DIR / "tuner_dir"

AUTOTUNE = tf.data.AUTOTUNE

# Establecemos esras semillas para la reproducibilidad
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def build_model(hp, input_shape, num_classes):
    model = Sequential()

    # Hiperparámetros para la activación #

    dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.6, step=0.1)
    lr = hp.Choice('learning_rate', values=[0.0001, 0.0003, 0.001])
    ks = hp.Choice('kernel_size', values=[3, 5])
    l2_reg = hp.Float('l2_reg', min_value=0.000001, max_value=0.001, sampling='log')

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
    class_to_idx = {name: idx for idx, name in enumerate(train_ds.class_names)}
    num_classes = len(class_to_idx)
    print(f"Clases encontradas (en total {num_classes}): {class_to_idx}")

    # 1.2 capas de normalizado + augmentación (en GPU)
    data_aug = Sequential([
        layers.Rescaling(1. / 255),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.10),
        layers.RandomTranslation(0.10, 0.10),
        layers.RandomContrast(0.2),
        layers.RandomFlip("horizontal")
    ])

    # train dataset → aug + prefetch
    train_ds_aug = (train_ds
                    .map(lambda x, y: (data_aug(x, training=True), y),
                         num_parallel_calls=AUTOTUNE)
                    .cache()
                    .prefetch(AUTOTUNE))

    # val dataset → solo normalizar
    val_ds_norm = (val_ds
                   .map(lambda x, y: (x / 255.0, y))
                   .cache()
                   .prefetch(AUTOTUNE))

    print("GPUs/MPS disponibles:", tf.config.list_physical_devices("GPU"))

    # 2. Definimos RandomSearch
    tuner = kt.RandomSearch(
        hypermodel=lambda hp: build_model(
            hp,
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            num_classes=num_classes
        ),
        objective='val_accuracy',
        max_trials=1,  # cambiarlo a 24
        executions_per_trial=1,
        overwrite=True,
        directory=str(TUNER_DIR),
        project_name='my_cnn_randomsearch'
    )

    # 3. Lanzamos la búsqueda de hiperparámetros
    tuner.search(
        train_ds_aug,
        epochs=2,  # cambiar a EPOCHS
        validation_data=val_ds_norm,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)]
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
        train_ds_aug,
        epochs=EPOCHS,
        validation_data=val_ds_norm,
        verbose=1
    )

    # 6. Guardar el mejor modelo
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    best_model.save(MODEL_PATH)
    print(f"Modelo guardado en: {MODEL_PATH}")

    # 7. Guardar mapeo de clases
    with open(LABELS_PATH, "w") as f:
        f.write("\n".join(class_names))
    print("Se ha guardado labels.txt con las clases.")

    # 8. Evaluar
    val_loss, val_acc = best_model.evaluate(val_ds_norm, verbose=0)
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
