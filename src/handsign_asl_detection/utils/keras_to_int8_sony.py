import tensorflow as tf
import tensorflow_model_optimization as tfmot

FLOAT_MODEL = "asl_cnn.keras"
INT8_MODEL_K = "asl_cnn_int8_sony.keras"

float_model = tf.keras.models.load_model(FLOAT_MODEL, compile=False)

# 1. Crea un modelo cuantizado (8 bits simétrico)
quantize_model = tfmot.quantization.keras.quantize_model
with tfmot.quantization.keras.quantize_scope():
    q_model = quantize_model(float_model)

# 2. (Opcional) Fine‑tune 1‑3 epochs con tu calibration set
q_model.compile(optimizer="adam",
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=["accuracy"])
# q_model.fit(calib_ds, epochs=3)

# 3. Q‑model ya contiene FakeQuant op en todas las capas
q_model.save(INT8_MODEL_K)
print("✔ Modelo INT8 Sony guardado:", INT8_MODEL_K)
