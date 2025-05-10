from pathlib import Path

import tensorflow as tf

SRC = Path("keras_model_fp32.tflite")      # tu modelo original
DST = Path("asl_flat.keras")      # salida

print("→ Cargando", SRC)
base = tf.keras.models.load_model(SRC, compile=False)

# ---------- aplanado ----------
def add_layers(graph_input, layer):
    """Devuelve (tensor_actual, lista_capas_aplanadas)"""
    if isinstance(layer, tf.keras.Model):           # sub‑modelo; recursivo
        x = graph_input
        for sub in layer.layers:
            if not isinstance(sub, tf.keras.layers.InputLayer):
                x, _ = add_layers(x, sub)
        return x, []
    else:
        x = layer(graph_input)
        return x, [layer]

inp = tf.keras.Input(shape=base.input_shape[1:], name="input")
x, flat_layers = inp, []
for layer in base.layers:
    if isinstance(layer, tf.keras.layers.InputLayer):
        continue
    x, new = add_layers(x, layer)
    flat_layers.extend(new)

flat_model = tf.keras.Model(inp, x, name="asl_flat")
# Copiamos pesos capa a capa
for l_old, l_new in zip([l for l in base.layers if not isinstance(l, tf.keras.layers.InputLayer)],
                        flat_layers):
    l_new.set_weights(l_old.get_weights())

flat_model.save(DST, save_format="keras")
print("Modelo aplanado guardado en", DST)
