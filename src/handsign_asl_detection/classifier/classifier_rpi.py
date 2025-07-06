"""
classifier_rpi.py
------------------

Envuelve toda la lógica necesaria para clasificar en tiempo real con el
modelo ASL

Flujo:
1. Detecta la mano
2. Recorta la ROI y el margen (offset)
3. Ajusta la imagen al lienzo cuadrado (300x300)
4. Envía la imagen al modelo .tflite (o .h5 para el futuro)
5. Devuelve el frame con la predicción

Autor: Hugo Gómez Martín
Contacto: hgm1001@alu.ubu.es
Fecha: 05/07/2025
"""

import math
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector

from handsign_asl_detection.config import TEACHABLE_TFL_DIR, IMG_SIZE, OFFSET

MODEL_PATH = TEACHABLE_TFL_DIR / "keras_model_fp16.tflite"
LABELS_PATH = TEACHABLE_TFL_DIR / "labels.txt"

class RealTimeASLClassifier:
    """
    Clasificador ASL en tiempo real
    """

    def __init__(
            self,
            model_path: str | Path = MODEL_PATH,
            labels_path: str | Path = LABELS_PATH,
            img_size: int = IMG_SIZE,
            offset: int = OFFSET
    ):
        # Normaliza y detecta si es .tflite
        self.model_path = Path(model_path)
        self.is_tflite = self.model_path.suffix == ".tflite"

        # cargar intérprete o modelo Keras
        if self.is_tflite:  # interprete mas optimizado .tflite
            self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        else:  # interprete .h5
            from tensorflow.keras.models import load_model
            self.model = load_model(self.model_path)

        # cargar etiquetas
        with open(labels_path, "r") as f:
            self.labels = [ln.strip() for ln in f]

        self.img_size = img_size  # 224
        self.offset = offset  # 20
        self.detector = HandDetector(maxHands=1)  # solo una mano

    def preprocess(
            self,
            img: np.ndarray,
            bbox: tuple[int, int, int, int]
    ) -> np.ndarray | None:

        """
        Recorta la mano, la resscala y la centra en el lienzo blanco.
        Devuelve BGR uint8 listo para convertir a RGB
        """
        x, y, w, h = bbox

        # Recorte con margen
        crop = img[max(0, y - self.offset):y + h + self.offset,
               max(0, x - self.offset):x + w + self.offset]

        if crop.size == 0:  # ROI vacía, omitimos frame
            return None

        # lienzo blanco base
        canvas = np.ones((self.img_size, self.img_size, 3), np.uint8) * 255
        ar = h / w
        if ar > 1:  # mano alta
            k = self.img_size / h
            nw = min(self.img_size, math.ceil(k * w))
            rz = cv2.resize(crop, (nw, self.img_size))
            gap = (self.img_size - nw) // 2
            canvas[:, gap:gap + nw] = rz
        else:  # mano ancha
            k = self.img_size / w
            nh = min(self.img_size, math.ceil(k * h))
            rz = cv2.resize(crop, (self.img_size, nh))
            gap = (self.img_size - nh) // 2
            canvas[gap:gap + nh, :] = rz

        return canvas  # BGR uint8

    def classify_frame(
            self,
            img: np.ndarray
    ) -> tuple[np.ndarray, str | None, float | None]:

        """
        Clasifica el frame devolviendo frame con anotaciones dibujadas,
        clase predicha y la probabilidad en porcentaje
        """

        hands, img = self.detector.findHands(img)  # dibujamos landmarks
        if not hands:  # pasamos si no hay mano detectada
            return img, None, None

        bbox = hands[0]["bbox"]
        proc_bgr = self.preprocess(img, bbox)
        if proc_bgr is None:
            return img, None, None

        # convertimos BGR→RGB
        proc_rgb = proc_bgr[:, :, ::-1]  # invertimos canales
        inp = (proc_rgb.astype(np.float32) / 255.0)[None, ...]

        # inferencia
        if self.is_tflite:
            # dtype puede ser FP32 o PF16
            self.interpreter.set_tensor(self.input_details[0]["index"], inp.astype(self.input_details[0]["dtype"]))
            self.interpreter.invoke()
            preds = self.interpreter.get_tensor(self.output_details[0]["index"])[0]  # una dimensión para predecir
        else:
            preds = self.model.predict(inp, verbose=0)[0]

        idx = int(np.argmax(preds))  # clase mas probable
        label = self.labels[idx]
        confidence = float(preds[idx] * 100)  # en porcentaje la confianza

        # anotación en img (BGR)
        x, y, w, h = bbox
        # Bounding box negro grueso con blanco fino
        cv2.rectangle(img, (x - self.offset, y - self.offset), (x + w + self.offset, y + h + self.offset), (0, 0, 0), 6)
        cv2.rectangle(img, (x - self.offset, y - self.offset), (x + w + self.offset, y + h + self.offset),
                      (255, 255, 255), 2)
        txt = f"{label} {confidence:.1f}%"

        # texto con doble trazo al igual que el bbox (negro + blanco) mas legible
        cv2.putText(img, txt, (x + 60, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 6)
        cv2.putText(img, txt, (x + 60, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

        return img, label, confidence
