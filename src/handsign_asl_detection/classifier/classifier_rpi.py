import math
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector

from handsign_asl_detection.config import TEACHABLE_TFL_DIR, IMG_SIZE, OFFSET

# Rutas
MODEL_PATH = TEACHABLE_TFL_DIR / "keras_model_fp16.tflite"
LABELS_PATH = TEACHABLE_TFL_DIR / "labels.txt"


class RealTimeASLClassifier:
    def __init__(
            self,
            model_path: str | Path = MODEL_PATH,
            labels_path: str | Path = LABELS_PATH,
            img_size: int = IMG_SIZE,
            offset: int = OFFSET,
    ):
        self.model_path = Path(model_path)
        self.is_tflite = self.model_path.suffix == ".tflite"

        # Carga del intérprete o del modelo Keras
        if self.is_tflite:
            self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        else:
            from tensorflow.keras.models import load_model
            self.model = load_model(self.model_path)

        # Carga de etiquetas
        with open(labels_path, "r") as f:
            self.labels = [line.strip() for line in f]

        # Parámetros de imagen
        self.img_size = img_size
        self.offset = offset

        # Detector de mano (cvzone)
        self.detector = HandDetector(maxHands=1)

    def preprocess(self, img: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
        """
        Recorta, ajusta relación de aspecto y normaliza.
        Devuelve imagen tamaño (IMG_SIZE, IMG_SIZE, 3) en float32 [0,1].
        """
        x, y, w, h = bbox
        img_crop = img[
                   max(0, y - self.offset): y + h + self.offset,
                   max(0, x - self.offset): x + w + self.offset,
                   ]
        if img_crop.size == 0:
            return None

        img_white = np.ones((self.img_size, self.img_size, 3), np.uint8) * 255
        aspect = h / w

        if aspect > 1:
            k = self.img_size / h
            new_w = min(self.img_size, math.ceil(k * w))
            img_resized = cv2.resize(img_crop, (new_w, self.img_size))
            w_gap = (self.img_size - new_w) // 2
            img_white[:, w_gap: w_gap + new_w] = img_resized
        else:
            k = self.img_size / w
            new_h = min(self.img_size, math.ceil(k * h))
            img_resized = cv2.resize(img_crop, (self.img_size, new_h))
            h_gap = (self.img_size - new_h) // 2
            img_white[h_gap: h_gap + new_h, :] = img_resized

        return img_white.astype(np.float32) / 255.0

    def classify_frame(self, img: np.ndarray) -> tuple[np.ndarray, str | None, float | None]:
        """
        Procesa un frame BGR, detecta mano, hace inferencia y devuelve:
        (frame_anotado, label, confidence).
        Si no hay mano detectada, label y confidence son None.
        """
        hands, img = self.detector.findHands(img)  # img ya viene con keypoints
        if not hands:
            return img, None, None

        hand = hands[0]
        bbox = hand["bbox"]  # x, y, w, h
        proc = self.preprocess(img, bbox)
        if proc is None:
            return img, None, None

        # Del BGR (cvzone) a RGB (modelo) y añadimos batch
        inp = proc[:, :, ::-1][None].astype(self.input_details[0]["dtype"] if self.is_tflite else np.float32)

        # Inferencia
        if self.is_tflite:
            self.interpreter.set_tensor(self.input_details[0]["index"], inp)
            self.interpreter.invoke()
            preds = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        else:
            preds = self.model.predict(inp, verbose=0)[0]

        idx = int(np.argmax(preds))
        label = self.labels[idx]
        confidence = float(preds[idx] * 100)

        # Anotación
        x, y, w, h = bbox

        cv2.rectangle(img, (x - self.offset, y - self.offset),
                      (x + w + self.offset, y + h + self.offset), (0, 0, 0), 6)
        cv2.rectangle(img, (x - self.offset, y - self.offset),
                      (x + w + self.offset, y + h + self.offset), (255, 255, 255), 2)
        cv2.putText(img, f"{label} {confidence:.1f}%", (x + 60, y - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 6)
        cv2.putText(img, f"{label} {confidence:.1f}%", (x + 60, y - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

        return img, label, confidence
