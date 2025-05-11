# train_and_qat.py
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization,
                                     MaxPooling2D, GlobalAveragePooling2D,
                                     Dense, Dropout)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2

# ─── rutas y parámetros ────────────────────────────────────
IMG_SIZE = 224
EPOCHS = 2
QAT_EPOCHS = 5      # bastan pocos para QAT
BATCH_SIZE = 32

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "Data_disordered"

OUT_DIR = PROJECT_ROOT / "models" / "IMX_ready_int"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FLOAT_MODEL_PATH = OUT_DIR / "asl_cnn_linear.keras"
QAT_MODEL_PATH = OUT_DIR / "asl_cnn_int8.keras"
# ────────────────────────────────────────────────────────────

# reproducibilidad
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# ─── generadores ───────────────────────────────────────────
train_gen = ImageDataGenerator(
    rescale=1/255, rotation_range=15, zoom_range=0.1,
    width_shift_range=0.1, height_shift_range=0.1,
    brightness_range=[0.8,1.2], horizontal_flip=True,
    fill_mode='reflect', validation_split=0.2)

val_gen = ImageDataGenerator(rescale=1/255, validation_split=0.2)

train_ds = train_gen.flow_from_directory(
    DATA_DIR, target_size=(IMG_SIZE,IMG_SIZE), batch_size=BATCH_SIZE,
    class_mode='sparse', subset='training', seed=42)

val_ds = val_gen.flow_from_directory(
    DATA_DIR, target_size=(IMG_SIZE,IMG_SIZE), batch_size=BATCH_SIZE,
    class_mode='sparse', subset='validation', seed=42)

NUM_CLASSES = len(train_ds.class_indices)

# ─── modelo float (sin softmax, salida linear) ────────────
inp = Input(shape=(IMG_SIZE,IMG_SIZE,3), name="input")
x = Conv2D(32,3,activation='relu')(inp)
x = BatchNormalization()(x)
x = MaxPooling2D()(x)
x = Conv2D(64,3,activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D()(x)
x = Conv2D(128,3,activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D()(x)
x = Conv2D(256,3,activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D()(x)
x = GlobalAveragePooling2D()(x)
x = Dense(512,activation='relu',kernel_regularizer=l2(4e-4))(x)
x = Dropout(0.5)(x)
logits = Dense(NUM_CLASSES, activation='linear', name="output")(x)

float_model = tf.keras.Model(inp, logits, name="asl_cnn_imx")
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
float_model.compile(Adam(1e-4), loss=loss_fn, metrics=["accuracy"])

# Entrena el float model (opcional, si no lo tienes ya)
cb_early = EarlyStopping("val_accuracy", patience=7, restore_best_weights=True, verbose=1)
cb_redlr = ReduceLROnPlateau("val_accuracy", factor=0.5, patience=3, min_lr=1e-6, verbose=1)
float_model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds,
                callbacks=[cb_early, cb_redlr], verbose=1)
float_model.save(FLOAT_MODEL_PATH, save_format="keras")

# ─── PASO QAT ──────────────────────────────────────────────
# 1) define función para anotar sólo Conv2D y Dense
def apply_quantization_to_layer(layer):
    if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
        # anotamos la capa para QAT usando la configuración por defecto de 8 bits
        return tfmot.quantization.keras.quantize_annotate_layer(layer)
    return layer

# 2) clona el modelo anotando solo las capas válidas
annotated_model = tf.keras.models.clone_model(
    float_model,
    clone_function=apply_quantization_to_layer)

# 3) convierte anotaciones en wrappers de QAT
with tfmot.quantization.keras.quantize_scope():
    qat_model = tfmot.quantization.keras.quantize_apply(annotated_model)

# 4) compila y reentrena un puñado de epochs
qat_model.compile(Adam(1e-4),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
qat_model.fit(train_ds, epochs=QAT_EPOCHS, validation_data=val_ds, verbose=1)

# 5) strip wrappers y guarda el modelo final int8
final_model = tfmot.quantization.keras.strip_quantization(qat_model)
final_model.save(QAT_MODEL_PATH, save_format="keras")
print("Modelo cuantizado listo:", QAT_MODEL_PATH)
