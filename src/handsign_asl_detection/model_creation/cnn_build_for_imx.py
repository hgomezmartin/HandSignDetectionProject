import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization,
                                     MaxPooling2D, GlobalAveragePooling2D,
                                     Dense, Dropout)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2

from HandSignDetectionProject.src.handsign_asl_detection.dataset.sample_rep_dataset import PROJECT_ROOT

# ---------- hiper‑parámetros ----------
IMG_SIZE = 224
EPOCHS = 70
BATCH_SIZE = 32
THIS_FILE = Path(__file__).resolve()

PROJECT_ROOT = THIS_FILE.parents[3]
DATA_DIR = PROJECT_ROOT / "Data" / "Data_disordered"

OUT_DIR = PROJECT_ROOT / "Model"/ "IMX_ready"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = OUT_DIR / "asl_cnn.keras"  # *** .keras ***
LABELS_TXT = OUT_DIR / "class_labels.txt"
PLOTS_DIR = OUT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# ---------- data generators ----------
train_gen = ImageDataGenerator(
    rescale=1. / 255, rotation_range=15, zoom_range=0.1,
    width_shift_range=0.1, height_shift_range=0.1,
    brightness_range=[0.8, 1.2], horizontal_flip=True,
    fill_mode='reflect', validation_split=0.2)

val_gen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

train_ds = train_gen.flow_from_directory(
    DATA_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    class_mode='sparse', subset='training', seed=42)

val_ds = val_gen.flow_from_directory(
    DATA_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    class_mode='sparse', subset='validation', seed=42)

NUM_CLASSES = len(train_ds.class_indices)
print("Clases:", train_ds.class_indices)

# ---------- modelo funcional plano ----------
inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input")

x = Conv2D(32, 3, activation='relu')(inputs)
x = BatchNormalization()(x)
x = MaxPooling2D()(x)

x = Conv2D(64, 3, activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D()(x)

x = Conv2D(128, 3, activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D()(x)

x = Conv2D(256, 3, activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D()(x)

x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu', kernel_regularizer=l2(0.0004))(x)
x = Dropout(0.5)(x)
outputs = Dense(NUM_CLASSES, activation='softmax', name="output")(x)

model = tf.keras.Model(inputs, outputs, name="asl_cnn_imx")
model.compile(Adam(0.0001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# ---------- callbacks ----------
early = EarlyStopping(monitor="val_accuracy", patience=7, restore_best_weights=True, verbose=1)
redlr = ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=3,
                          min_lr=0.000001, verbose=1)

# ---------- entrenamiento ----------
hist = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds,
                 callbacks=[early, redlr], verbose=1)

# ---------- guardado ----------
model.save(MODEL_PATH, save_format="keras")  # <- fichero .keras listo para IMX
print("Modelo guardado en", MODEL_PATH)

with LABELS_TXT.open("w") as f:
    for cls in sorted(train_ds.class_indices, key=train_ds.class_indices.get):
        f.write(cls + "\n")

# ---------- plots  ----------
for metric, fname in [("loss", "loss.png"), ("accuracy", "accuracy.png")]:
    plt.figure()
    plt.plot(hist.epoch, hist.history[metric], label="train")
    plt.plot(hist.epoch, hist.history["val_" + metric], label="val")
    plt.title(metric.capitalize())
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOTS_DIR / fname, dpi=150)
