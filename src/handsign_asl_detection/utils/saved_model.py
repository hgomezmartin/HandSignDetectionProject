from pathlib import Path

import model_compression_toolkit as mct
import tensorflow as tf
from model_compression_toolkit import get_target_platform_capabilities
from model_compression_toolkit.legacy.keras_quantization_facade import (
    keras_post_training_quantization
)

model = tf.keras.models.load_model("asl_cnn_linear.keras", compile=False)

# 2) Prepara el generador de dataset representativo (256 lotes)
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]
VAL_DIR = PROJECT_ROOT / "Data" / "Data_disordered"
IMG_SIZE = 224
BATCH = 32

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255, validation_split=0.2)

val_ds = datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode=None,
    subset="validation",
    seed=0)


def rep_data_gen():
    for _ in range(256):
        yield next(val_ds)


# 3) Obtén las capacidades de la plataforma IMX500
tpc = get_target_platform_capabilities('tensorflow', 'imx500')
core_cfg = mct.core.CoreConfig(quantization_config=mct.core.QuantizationConfig())

q_model, quant_info = keras_post_training_quantization(
    in_model=model,
    representative_data_gen=rep_data_gen,
    n_iter=256,
    target_platform_capabilities=tpc,
    core_config=core_cfg
)

q_model.save("asl_cnn_int8.keras", save_format="keras")
