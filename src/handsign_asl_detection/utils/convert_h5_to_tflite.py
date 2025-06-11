from pathlib import Path

import numpy as np
import tensorflow as tf

from handsign_asl_detection.config import TEACHABLE_DIR, TEACHABLE_TFL_DIR

H5_PATH = TEACHABLE_DIR / "keras_model.h5"
FP32_PATH = TEACHABLE_TFL_DIR / "keras_model_fp32.tflite"
FP16_PATH = TEACHABLE_TFL_DIR / "keras_model_fp16.tflite"


def convert_h5_to_tflite(verbose: bool = True):
    TEACHABLE_TFL_DIR.mkdir(parents=True, exist_ok=True)

    if verbose: print("Cargando modelo", H5_PATH)
    model = tf.keras.models.load_model(H5_PATH)

    # FP32
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    FP32_PATH.write_bytes(conv.convert())
    if verbose: print("Guardado FP32 en", FP32_PATH)

    # FP16
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_types = [tf.float16]
    FP16_PATH.write_bytes(conv.convert())
    if verbose: print("Guardado FP16 en", FP16_PATH)

    # test rápido
    def _quick(path: Path):
        itp = tf.lite.Interpreter(model_path=str(path))
        itp.allocate_tensors()
        shape = itp.get_input_details()[0]["shape"]
        itp.set_tensor(itp.get_input_details()[0]["index"],
                       np.zeros(shape, dtype=np.float32))
        itp.invoke()
        if verbose:
            print(f"{path.name} ok – out shape",
                  itp.get_output_details()[0]["shape"])

    _quick(FP32_PATH)
    _quick(FP16_PATH)
    return FP32_PATH, FP16_PATH


if __name__ == "__main__":
    convert_h5_to_tflite()
