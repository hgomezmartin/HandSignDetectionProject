# conv_h5_to_keras.py
import tensorflow as tf

model = tf.keras.models.load_model("keras_model.h5", compile=False)
model.save("keras_model.keras", save_format="tf")
