import math
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.models import load_model


class RealTimeASLClassifier:
    def __init__(self, model_path, labels_path, img_size=224, offset=20):
        # Carga del modelo entrenado
        self.model_path = Path(model_path)
        self.is_tflite = self.model_path.suffix == ".tflite"

        # ---------- 1. Carga del modelo ----------
        if self.is_tflite:
            self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        else:
            self.model = load_model(self.model_path)

        # Carga de etiquetas (una por línea)
        with open(labels_path, "r") as f:
            self.labels = [line.strip() for line in f.readlines()]

        # Utilidades
        self.img_size = img_size
        self.offset = offset
        # Inicializa el detector de manos (máximo 1 mano)
        self.detector = HandDetector(maxHands=1)
        # Inicializa la cámara
        self.cap = cv2.VideoCapture(0)

    def preprocess(self, img, bbox):
        """
        Recorta la región de interés de la mano, la redimensiona y la normaliza.
        Se utiliza una imagen blanca de tamaño fijo (img_size x img_size) para mantener la relación de aspecto.
        """
        x, y, w, h = bbox
        # Recorte con offset
        img_crop = img[max(0, y - self.offset):y + h + self.offset,
                   max(0, x - self.offset):x + w + self.offset]
        # Crea una imagen blanca base
        img_white = np.ones((self.img_size, self.img_size, 3), np.uint8) * 255

        # Ajusta el recorte según la relación de aspecto de la caja
        aspect_ratio = h / w
        if aspect_ratio > 1:
            k = self.img_size / h
            # new_w = math.ceil(k * w)
            new_w = min(self.img_size, math.ceil(k * w))
            if img_crop.size == 0:
                return None
            img_resize = cv2.resize(img_crop, (new_w, self.img_size))
            w_gap = (self.img_size - new_w) // 2
            img_white[:, w_gap:w_gap + new_w] = img_resize
        else:
            k = self.img_size / w
            # new_h = math.ceil(k * h)
            new_h = min(self.img_size, math.ceil(k * h))  # nunca mayor que img_size
            if img_crop.size == 0:
                return None
            img_resize = cv2.resize(img_crop, (self.img_size, new_h))
            h_gap = (self.img_size - new_h) // 2
            img_white[h_gap:h_gap + new_h, :] = img_resize

        # Normaliza la imagen a rango [0,1]
        img_normalized = img_white.astype(np.float32) / 255.0
        return img_normalized

    def run(self):
        while True:
            ret, img = self.cap.read()
            if not ret:
                print("ERROR: cámara no disponible, saliendo...")
                break

            # Espeja la imagen para mayor naturalidad en un futuro?, duplicar dataset espejado con data augmentation?
            # img = cv2.flip(img, 1)

            # Detecta la mano (si la hay)
            hands, img = self.detector.findHands(img)
            img_output = img.copy()

            if hands:
                hand = hands[0]
                bbox = hand['bbox']  # x, y, w, h
                hand_type = hand['type']
                processed_img = self.preprocess(img, bbox)  # BGR para teachable_machine
                proc_img_rgb = processed_img[:, :, ::-1]  # RGB para nuestro modelo, no el de teachable_machine

                if processed_img is not None:
                    # Agrega dimensión batch (1, img_size, img_size, 3)
                    in_dtype = self.input_details[0]['dtype']  # float32, float16, uint8, int8…
                    input_img = np.expand_dims(proc_img_rgb, 0).astype(in_dtype)  # proc_img_rgb
                    # Realiza la predicción
                    if self.is_tflite:  # NUEVO
                        self.interpreter.set_tensor(
                            self.input_details[0]['index'], input_img
                        )
                        self.interpreter.invoke()
                        prediction = self.interpreter.get_tensor(
                            self.output_details[0]['index']
                        )
                    else:  # Keras .h5
                        prediction = self.model.predict(input_img, verbose=0)

                    index = np.argmax(prediction)
                    confidence = prediction[0][index] * 100
                    label = self.labels[index]

                    # Dibuja la caja y el resultado en la imagen de salida
                    x, y, w, h = bbox
                    if w < 20 or h < 20:
                        continue
                    cv2.rectangle(img, (x - self.offset, y - self.offset),
                                  (x + w + self.offset, y + h + self.offset), (0, 0, 0), 6)
                    cv2.rectangle(img, (x - self.offset, y - self.offset),
                                  (x + w + self.offset, y + h + self.offset), (255, 255, 255), 2)
                    cv2.putText(img, f"{label} {confidence:.1f}%", (x + 60, y - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 6)
                    cv2.putText(img, f"{label} {confidence:.1f}%", (x + 60, y - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
                    # Muestra la imagen preprocesada (recortada y ajustada)
                    # cv2.imshow("Processed", processed_img)
                    cv2.imshow("Processed", (proc_img_rgb * 255).astype(np.uint8))  # proc_img_rgb

            cv2.imshow("ASL Recognition", img)
            key = cv2.waitKey(1)
            if key in (ord('q'), 27):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Ruta al modelo y a las etiquetas según se han guardado tras el entrenamiento

    model_dir = Path("rpi_tflite")
    model_path = model_dir / "keras_model_fp16.tflite"
    labels_path = model_dir / "labels.txt"

    classifier = RealTimeASLClassifier(model_path, labels_path, img_size=224, offset=20)
    classifier.run()
