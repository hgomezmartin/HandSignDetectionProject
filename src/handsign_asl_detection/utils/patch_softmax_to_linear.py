# file: remove_softmax.py
from pathlib import Path

import tensorflow as tf

SRC = Path("asl_cnn.keras")      # modelo entrenado (softmax)
DST = Path("asl_cnn_linear.keras")

# 1) Carga el modelo entrenado
src_model = tf.keras.models.load_model(SRC, compile=False)
print("cargado", SRC)

# 2) Última capa Dense (con softmax) y su entrada
old_dense   = src_model.layers[-1]
prev_tensor = src_model.layers[-2].output          # salida antes del softmax

# 3) Crea una capa idéntica pero activation='linear'
new_dense = tf.keras.layers.Dense(
    units=old_dense.units,
    activation="linear",          # ← requisito de IMX500
    use_bias=old_dense.use_bias,
    name="output"
)(prev_tensor)

# 4) Ensambla el modelo funcional y copia pesos
dst_model = tf.keras.Model(src_model.input, new_dense, name="asl_cnn_linear")
dst_model.layers[-1].set_weights(old_dense.get_weights())

# (opcional) compílalo para poder evaluarlo en Keras
dst_model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

# 5) Guarda en formato .keras (aceptado por imxconv‑tf)
dst_model.save(DST, save_format="keras")
print("modelo lineal guardado en", DST)
