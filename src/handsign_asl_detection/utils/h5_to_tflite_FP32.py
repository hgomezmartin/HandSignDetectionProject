import numpy as np
import tensorflow as tf

MODEL_PATH = ""

# Cargamos el modelo .h5 inicial para convertirlo a tf lite para la Raspberry Pi
model = tf.keras.models.load_model('keras_model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

file = open("keras_model.tflite", "wb")
file.write(tflite_model)

# Pruebas para comprobar la conversión correcta

# 1. Carga el intérprete
out_path = 'keras_model.tflite'
interpreter = tf.lite.Interpreter(model_path=out_path)
interpreter.allocate_tensors()

# 2. Detalles de los tensores
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Entrada :", input_details)
print("Salida  :", output_details)

# 3. Infiere con datos de prueba (por ejemplo, ceros)
input_shape = input_details[0]['shape']
dummy_input = np.zeros(input_shape, dtype=input_details[0]['dtype'])
interpreter.set_tensor(input_details[0]['index'], dummy_input)
interpreter.invoke()
result = interpreter.get_tensor(output_details[0]['index'])
print("Inferencia de prueba resultó en un array de forma", result.shape)
