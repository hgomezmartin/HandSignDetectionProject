"""
random_search.py
-----------------
Realiza una busqueda aleatoria (Random Seach) de hiperparámetros
sobre nuestro modelo. Busca los mejores hiperparámetros en 10 epoch
hasta agotar la cantidad de intentos expuestos

Flujo:
1. Cargamos las imagenes desde nuestro dataset desordenado "DISORDERED_DATADIR
con una division 80 train / 20 test
2. Definimos aumentos ligeros para dar robustez al modelo
3. Ejecutamos Keras-Tuner con Randomsearch
5. Guardamos modjoers hiperparametros en JSON + HTML, el modleo final,
etiquetas y gráficas

Autor: Hugo Gómez Martín
Contacto: hgm1001@alu.ubu.es
Fecha: 05/07/2025
"""
import json
import os
import random

import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
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
    IMG_SIZE, BATCH_SIZE, EPOCHS_RS, SEED,
    DISORDERED_DATADIR, RANDOM_SEARCH_DIR
)

# Rutas y constantes
DATA_DIR = DISORDERED_DATADIR  # Cogemos el conjunto de datos desordenado
MODEL_PATH = RANDOM_SEARCH_DIR / "my_cnn_randomsearch.h5"  # Denominamos el modelo final "my_cnn_randomsearch.h5"
LABELS_PATH = RANDOM_SEARCH_DIR / "labels.txt"  # Establecemos el conjutno de estiquetas en un fichero de texto
PLOTS_DIR = RANDOM_SEARCH_DIR / "plots"  # Creamos una carpeta para almacenar las gráficas
HPARAMS_JSON_PATH = RANDOM_SEARCH_DIR / "hparams" / "best_hparams.json"  # Creamos un .json para mejores hparams
HPARAMS_HTML_PATH = RANDOM_SEARCH_DIR / "hparams" / "best_hparams.html"  # Creamos un .html para mejores hparams
TUNER_DIR = RANDOM_SEARCH_DIR / "tuner_dir"

AUTOTUNE = tf.data.AUTOTUNE  # para paralelizar tf.data

# Semillas para la reproducibilidad (con el randomsearch)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def build_rs_model(hp, input_shape, num_classes):
    base = EfficientNetB0(include_top=False,  # Quitamos cabecera
                          weights="imagenet",  # Pesos base
                          input_shape=input_shape)
    base.trainable = False  # Congelamos tod o el backbone por bug el TF-metal

    # Establecemos el rango de hiperprametros que queremos q pruebe
    dropout_rate = hp.Choice("dropout_rate", values=[0.2, 0.3, 0.4])
    lr = hp.Choice("learning_rate", values=[0.001, 0.0003, 0.0001])
    l2_reg = hp.Choice("l2_reg", values=[0.001, 0.0005, 0.0001])

    model = Sequential([
        layers.Input(shape=input_shape),
        layers.Lambda(preprocess_input),  # normalización específica EfficientNet
        base,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dropout(dropout_rate),

        Dense(num_classes,
              activation="softmax",
              kernel_regularizer=l2(l2_reg))
    ])

    # Compilamos
    model.compile(optimizer=Adam(lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def main():
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
    train_ds_aug = (train_ds
                    .map(lambda x, y: (aug(x, training=True), y),
                         num_parallel_calls=AUTOTUNE)
                    .cache()
                    .prefetch(AUTOTUNE))

    # solo cache y prefech
    val_ds_norm = val_ds.cache().prefetch(AUTOTUNE)

    # Listamos GPU para comporbar que se está utilizando
    print("GPUs/MPS disponibles:", tf.config.list_physical_devices("GPU"))

    # 2. Definimos RandomSearch
    tuner = kt.RandomSearch(
        hypermodel=lambda hp: build_rs_model(
            hp,
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            num_classes=num_classes
        ),
        objective='val_accuracy',  # métrica a optimizar
        max_trials=30,  # numero de comprobaciones (alguna mas para abarcar mas espacio en caso de repetición)
        executions_per_trial=1,  # 1 fit por cada trial
        overwrite=True,
        directory=str(TUNER_DIR),
        project_name='my_cnn_randomsearch'
    )

    # 3. Callbacks
    earlystop_cb = EarlyStopping(
        monitor="val_accuracy",
        patience=3,  # al tener solo 10 epochs, establecemos 3
        restore_best_weights=True,
        verbose=1
    )

    # 4. Lanzamos la búsqueda de hiperparámetros
    tuner.search(
        train_ds_aug,
        epochs=EPOCHS_RS,
        validation_data=val_ds_norm,
        callbacks=[earlystop_cb]
    )

    # 5. Obtenemos los mejores hiperparámetros
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Mejores hiperparámetros encontrados:")
    for param, value in best_hp.values.items():
        print(f"  {param}: {value}")

    # 5.1 Guardar hiperparámetros en JSON
    os.makedirs(os.path.dirname(HPARAMS_JSON_PATH), exist_ok=True)
    with open(HPARAMS_JSON_PATH, "w") as f:
        json.dump(best_hp.values, f, indent=4)
    print(f"Hiperparámetros guardados en: {HPARAMS_JSON_PATH}\n")

    # 5.2 Guardar en HTML
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

    # 6. Construimos y entrenamos el modelo con mejores hiperparámetros
    best_model = tuner.hypermodel.build(best_hp)
    best_model.summary()

    # 7. Entrenamiento (una sola fase)
    history = best_model.fit(
        train_ds_aug,
        epochs=EPOCHS_RS,
        validation_data=val_ds_norm,
        verbose=1
    )

    # 8. Guardar el mejor modelo y etiquetas
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    best_model.save(MODEL_PATH)
    print(f"Modelo guardado en: {MODEL_PATH}")

    with open(LABELS_PATH, "w") as f:
        f.write("\n".join(class_names))
    print("Se ha guardado labels.txt con las clases.")

    # 9. Evaluación
    val_loss, val_acc = best_model.evaluate(val_ds_norm, verbose=0)
    print(f"Precisión en validación: {val_acc * 100:.2f}%, Pérdida: {val_loss:.4f}")

    # 10. Graficas
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

    y_true = np.concatenate([y.numpy() for _, y in val_ds_norm])  # etiquetas reales
    y_pred_probs = best_model.predict(val_ds_norm, verbose=0)  # probabilidades
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
