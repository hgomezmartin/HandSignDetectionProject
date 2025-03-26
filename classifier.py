import math

import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.models import load_model


class RealTimeASLClassifier:
    def __init__(self, model_path, labels_path, img_size=224, offset=20):
        # Carga del modelo entrenado
        self.model = load_model(model_path)
        # Carga de etiquetas (una por línea)
        with open(labels_path, "r") as f:
            self.labels = [line.strip() for line in f.readlines()]
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
            new_w = math.ceil(k * w)
            if img_crop.size == 0:
                return None
            img_resize = cv2.resize(img_crop, (new_w, self.img_size))
            w_gap = (self.img_size - new_w) // 2
            img_white[:, w_gap:w_gap + new_w] = img_resize
        else:
            k = self.img_size / w
            new_h = math.ceil(k * h)
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
                break

            # Espeja la imagen para mayor naturalidad en un futuro?, duplicar dataset espejado con data augmentation?
            #img = cv2.flip(img, 1)
            img_output = img.copy()

            # Detecta la mano (si la hay)
            hands, img = self.detector.findHands(img)
            if hands:
                hand = hands[0]
                bbox = hand['bbox']  # x, y, w, h
                processed_img = self.preprocess(img_output, bbox)
                if processed_img is not None:
                    # Agrega dimensión batch (1, img_size, img_size, 3)
                    input_img = np.expand_dims(processed_img, axis=0)
                    # Realiza la predicción
                    prediction = self.model.predict(input_img)
                    index = np.argmax(prediction)
                    confidence = prediction[0][index] * 100
                    label = self.labels[index]

                    # Dibuja la caja y el resultado en la imagen de salida
                    x, y, w, h = bbox
                    cv2.rectangle(img_output, (x - self.offset, y - self.offset),
                                  (x + w + self.offset, y + h + self.offset), (255, 0, 255), 2)
                    cv2.putText(img_output, f"{label} {confidence:.1f}%", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 2)
                    # Muestra la imagen preprocesada (recortada y ajustada)
                    cv2.imshow("Processed", processed_img)

            cv2.imshow("ASL Recognition", img_output)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Ruta al modelo y a las etiquetas según se han guardado tras el entrenamiento
    model_path = "Model/my_cnn_model.h5"
    labels_path = "Model/class_labels.txt"
    classifier = RealTimeASLClassifier(model_path, labels_path, img_size=224, offset=20)
    classifier.run()
