"""
classifier.py
--------------
Detecta la mano en tiempo real con la cámara, preprocesa la ROI (región
de interés) y clasifica el signo ASL usando un modelo cnn.

Flujo:
1. Abrir webcam y detectar una mano con cvzone
2.Recortar la ROI + margen offser
3. Ajustar el recorte a un liezo blanco cuadrado de (300x300)
4. BGR->RGB y enviar al modelo
5. Pintar la predicción y confianza en porcentaje sobre el vídeo

Autor: Hugo Gómez Martín
Contacto: hgm1001@alu.ubu.es
Fecha: 05/07/2025
"""
import math

import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.models import load_model

from handsign_asl_detection.config import IMG_SIZE, OFFSET, FINAL_DIR

MODEL_PATH = FINAL_DIR / "my_cnn_model.h5"  # models/augmented/my_cnn_model.h5
LABELS_PATH = FINAL_DIR / "labels.txt"


class RealTimeASLClassifier:
    """
    Lógica de la cámara, preprocesado y clasificación
    """

    # Constructor
    def __init__(
            self,
            model_path: str = MODEL_PATH,
            labels_path: str = LABELS_PATH,
            img_size: int = IMG_SIZE,
            offset: int = OFFSET
    ):
        # Carga del modelo entrenado
        self.model = load_model(model_path)

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
        Devuelve una imágen 3 canales RGB cuadrada de 300x300 lista
        para el modelo o none si el recorte falla
        """
        x, y, w, h = bbox

        # Recorte con margen (offset)
        img_crop = img[max(0, y - self.offset):y + h + self.offset,
                   max(0, x - self.offset):x + w + self.offset]

        # Crea una imagen blanca base donde pegaremos la mano
        img_white = np.ones((self.img_size, self.img_size, 3), np.uint8) * 255

        # Ajusta el recorte según la relación al alto o ancho
        aspect_ratio = h / w
        if aspect_ratio > 1:  # Mano alta
            k = self.img_size / h
            # new_w = math.ceil(k * w)
            new_w = min(self.img_size, math.ceil(k * w))
            if img_crop.size == 0:
                return None
            img_resize = cv2.resize(img_crop, (new_w, self.img_size))
            w_gap = (self.img_size - new_w) // 2
            img_white[:, w_gap:w_gap + new_w] = img_resize
        else:  # mano ancha
            k = self.img_size / w
            # new_h = math.ceil(k * h)
            new_h = min(self.img_size, math.ceil(k * h))  # nunca mayor que img_size
            if img_crop.size == 0:
                return None
            img_resize = cv2.resize(img_crop, (self.img_size, new_h))
            h_gap = (self.img_size - new_h) // 2
            img_white[h_gap:h_gap + new_h, :] = img_resize

        return img_white  # BGR

    def run(self):
        """
        Captura webcam y muestra predicciones en tiempo real
        """
        while True:
            ret, img = self.cap.read()
            if not ret:
                print("ERROR: cámara no disponible, saliendo...")
                break

            # Detecta la mano (si la hay) y dibuja los landmarks
            hands, img = self.detector.findHands(img)

            if hands:
                hand = hands[0]
                bbox = hand['bbox']  # x, y, w, h
                processed_img = self.preprocess(img, bbox)  # BGR

                if processed_img is not None:
                    proc_img_rgb = processed_img[:, :, ::-1]  # BGR->RGB

                    # Agrega dimensión batch (1, img_size, img_size, 3)
                    input_img = np.expand_dims(proc_img_rgb, axis=0)

                    # Realiza la predicción
                    prediction = self.model.predict(input_img)
                    index = np.argmax(prediction)
                    confidence = prediction[0][index] * 100
                    label = self.labels[index]

                    # Dibuja la caja y el resultado en la imagen de salida
                    x, y, w, h = bbox
                    if w < 20 or h < 20:
                        continue

                    # caja con doble borde blanco y negro
                    cv2.rectangle(img, (x - self.offset, y - self.offset),
                                  (x + w + self.offset, y + h + self.offset), (0, 0, 0), 6)
                    cv2.rectangle(img, (x - self.offset, y - self.offset),
                                  (x + w + self.offset, y + h + self.offset), (255, 255, 255), 2)

                    # texto con doble borde vlanco y negro
                    cv2.putText(img, f"{label} {confidence:.1f}%", (x + 60, y - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 6)
                    cv2.putText(img, f"{label} {confidence:.1f}%", (x + 60, y - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

            # ventana
            cv2.imshow("ASL Recognition", img)
            key = cv2.waitKey(1)
            if key in (ord('q'), 27):  # q o ESC para salir
                break

        # liberar recursos
        self.cap.release()
        cv2.destroyAllWindows()


# Punto de entrada
if __name__ == "__main__":
    classifier = RealTimeASLClassifier()
    classifier.run()
