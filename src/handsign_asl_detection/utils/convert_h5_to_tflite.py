"""
convert_h5_to_tflite.py
-----------------------
Convierte un modelo guardado en formato TF/Keras HDF5(.h5) a dos
variantes de TensorFlow Lite (.tflite) para la Raspberry Pi:

- FP32: Precisión por defecto de 32 bits en coma flotante
- FP16: 16 bits ocupando la mirad de memoria y acelerando
        la inferencia

Autor: Hugo Gómez Martín
Contacto: hgm1001@alu.ubu.es
Fecha: 05/07/2025
"""

import tensorflow as tf

from handsign_asl_detection.config import FINAL_DIR, FINAL_TFL_DIR

H5_PATH = FINAL_DIR / "my_cnn_model.h5"  # modelo entrenado
FP32_PATH = FINAL_TFL_DIR / "keras_model_fp32.tflite"  # Salida FP32
FP16_PATH = FINAL_TFL_DIR / "keras_model_fp16.tflite"  # Salida FP16


def convert_h5_to_tflite(verbose: bool = True):
    # Aseguramos la existencia de la carpeta destino
    FINAL_TFL_DIR.mkdir(parents=True, exist_ok=True)

    # Cargamos el modelo gracias a la herramienta de TF/Keras "load_model"
    if verbose:
        print("Cargando modelo", H5_PATH)
    model = tf.keras.models.load_model(H5_PATH)

    # Conversión FP32
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    FP32_PATH.write_bytes(conv.convert())
    if verbose:
        print("Guardado FP32 en", FP32_PATH)

    # Conversión FP16
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    # Especificamente FP16
    conv.target_spec.supported_types = [tf.float16]
    FP16_PATH.write_bytes(conv.convert())
    if verbose:
        print("Guardado FP16 en", FP16_PATH)

    return FP32_PATH, FP16_PATH


# Punto de entrada
if __name__ == "__main__":
    convert_h5_to_tflite()
