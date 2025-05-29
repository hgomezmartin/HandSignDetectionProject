# quantize_imx500.py
import model_compression_toolkit as mct
import numpy as np
import tensorflow as tf
from edgemdt_tpc import get_target_platform_capabilities  # paquete edge‑mdt‑tpc
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ───────rutas───────
FLOAT_MODEL = "asl_cnn.keras"
INT8_MODEL_K = "asl_cnn_linear_int8_v2.keras"
INT8_MODEL_ON = "asl_cnn_linear_int8_v2.onnx"
DATA_DIR = "rep_data"
IMG_SIZE, BATCH, CAL_STEPS = 224, 32, 250  # imágenes de calibración

# ───────cargar modelo en fp32───────
float_model = tf.keras.models.load_model(FLOAT_MODEL, compile=False)

# ───────Dejar la capa Dropout pero "apagada”───────
for layer in float_model.layers:
    if isinstance(layer, tf.keras.layers.Dropout):
        layer.rate = 0.0  # desactiva el drop
        layer.trainable = False

# ───────generador de datos representativos───────
datagen = ImageDataGenerator(rescale=1 / 255)
flow = datagen.flow_from_directory(DATA_DIR,
                                   target_size=(IMG_SIZE, IMG_SIZE),
                                   batch_size=BATCH,
                                   class_mode=None, shuffle=True)


def representative_data_gen():
    for _ in range(CAL_STEPS):
        images = next(flow)
        yield [images.astype(np.float32)]


# ───────descripción hardware IMX500───────
tpc = get_target_platform_capabilities(tpc_version="1.0", device_type="imx500")

# (opcional) configuración avanzada
core_cfg = mct.core.CoreConfig()  # mixed‑precision, GPTQ, etc.

# ───────cuantización post‑entrenamiento───────
quant_model, qinfo = mct.ptq.keras_post_training_quantization(
    in_model=float_model,
    representative_data_gen=representative_data_gen,
    core_config=core_cfg,
    target_platform_capabilities=tpc)  # :contentReference[oaicite:1]{index=1}

# ───────guardar modelos───────
quant_model.save(INT8_MODEL_K)
# tf.saved_model.save(quant_model, "tmp_saved_model")  # vía onnx‑tf

print("Modelo cuantizado guardado:", INT8_MODEL_K)
