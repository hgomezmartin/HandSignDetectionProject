import tensorflow as tf
m = tf.keras.models.load_model("keras_model.h5")
tf.saved_model.save(m, "saved_fp32")     # exporta
m2 = tf.keras.models.load_model("saved_fp32")
m2.save("asl_saved.keras", save_format="keras")
