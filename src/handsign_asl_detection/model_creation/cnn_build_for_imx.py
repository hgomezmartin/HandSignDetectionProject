import random
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D,
    GlobalAveragePooling2D, Dense, Lambda
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow_model_optimization.python.core.quantization.keras.quantize_layer import QuantizeLayer

# from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper

# ────── hiperparámetros y rutas ──────
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 1  # súbelo en producción
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]
DATA_DIR = PROJECT_ROOT / "Data" / "disordered"
OUT_DIR = PROJECT_ROOT / "models" / "imx_ready"
OUT_DIR.mkdir(parents=True, exist_ok=True)

QAT_MODEL_PATH = OUT_DIR / "asl_cnn_int8_imx_qat.keras"
STRIPPED_MODEL_PATH = OUT_DIR / "asl_cnn_int8_imx.keras"
LABELS_TXT = OUT_DIR / "class_labels.txt"

# ────── reproducibilidad ──────
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# ────── generadores de datos ──────
train_gen = ImageDataGenerator(
    rescale=1 / 255, rotation_range=15, zoom_range=0.1,
    width_shift_range=0.1, height_shift_range=0.1,
    brightness_range=[0.8, 1.2], horizontal_flip=True,
    fill_mode="reflect", validation_split=0.2
)
val_gen = ImageDataGenerator(rescale=1 / 255, validation_split=0.2)

train_ds = train_gen.flow_from_directory(
    DATA_DIR, target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE, class_mode="sparse",
    subset="training", seed=seed
)
val_ds = val_gen.flow_from_directory(
    DATA_DIR, target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE, class_mode="sparse",
    subset="validation", seed=seed
)
NUM_CLASSES = len(train_ds.class_indices)
print("Clases:", train_ds.class_indices)


# ────── modelo base (sin BN / Dropout, logits lineal) ──────
def build_base():
    inp = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input")
    x = Conv2D(32, 3, activation='relu')(inp)
    x = MaxPooling2D()(x)
    x = Conv2D(64, 3, activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(128, 3, activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(256, 3, activation='relu')(x)
    x = MaxPooling2D()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(4e-4))(x)
    logits = Dense(NUM_CLASSES, activation='linear', name="output")(x)
    return tf.keras.Model(inp, logits, name="asl_cnn_imx")


float_model = build_base()

# ────── 1) QAT: inserta FakeQuant ──────
with tfmot.quantization.keras.quantize_scope():
    qat_model = tfmot.quantization.keras.quantize_model(float_model)

# ────── 2) Entrena (fine-tuning) ──────
qat_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)
callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=8,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_accuracy", factor=0.5,
                      patience=3, min_lr=1e-6, verbose=1)
]
qat_model.fit(train_ds, validation_data=val_ds,
              epochs=EPOCHS, callbacks=callbacks, verbose=1)

# ────── 3) Guarda modelo QAT temporal ──────
qat_model.save(QAT_MODEL_PATH)
print("Modelo QAT guardado en", QAT_MODEL_PATH)


# ────── 4) Strip completo: Wrapper + QuantizeLayer ──────
def clone_fn(layer):
    # Desempaqueta Wrapper
    if isinstance(layer, QuantizeWrapper):
        return layer.layer
    # Sustituye QuantizeLayer por identidad
    if isinstance(layer, QuantizeLayer):
        return Lambda(lambda x: x, name=f"{layer.name}_id")
    return layer


stripped_model = tf.keras.models.clone_model(qat_model, clone_function=clone_fn)

# Copiar pesos (QuantizeLayer no posee pesos)
for src_layer, dst_layer in zip(qat_model.layers, stripped_model.layers):
    if isinstance(src_layer, QuantizeWrapper):
        dst_layer.set_weights(src_layer.layer.get_weights())
    elif not isinstance(src_layer, QuantizeLayer):
        dst_layer.set_weights(src_layer.get_weights())

# ────── 5) Guarda el modelo INT8 limpio ──────
stripped_model.save(STRIPPED_MODEL_PATH, save_format="keras")
print("Modelo final INT8 listo en", STRIPPED_MODEL_PATH)

# ────── 6) Guarda etiquetas ──────
with LABELS_TXT.open("w") as f:
    for cls in sorted(train_ds.class_indices, key=train_ds.class_indices.get):
        f.write(cls + "\n")
print("Labels guardadas en", LABELS_TXT)
