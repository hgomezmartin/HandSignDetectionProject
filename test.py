import math

import cv2
import numpy as np
from cvzone.ClassificationModule import Classifier
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/MyCNN/UltEntreno/224x224/my_cnn_model.h5",
                        "Model/MyCNN/UltEntreno/224x224/class_labels.txt")

offset = 20
imgSize = 224

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
          'V', 'W', 'X', 'Y', 'Z']
# labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)] jugando con posiciones ASCII

while True:
    success, img = cap.read()
    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            # imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            if imgCrop.size > 0:
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            else:
                continue

            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            # imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            if imgCrop.size > 0:
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            else:
                continue

            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        # Cambio crucial: normalizar la imagen
        # Convierte imgWhite a float32 y escala a [0,1] para que coincida con el entrenamiento.
        imgWhite_norm = imgWhite.astype(np.float32) / 255.0

        prediction, index = classifier.getPrediction(imgWhite_norm, draw=False)
        prediction_percentage = [f"{p * 100:.1f}%" for p in prediction]
        print(prediction_percentage, index)

        confidence = prediction[index] * 100  # Convertir a porcentaje
        confidence_text = f"{round(confidence, 1)}%"  # Redondeamos a 1 decimal

        text = f"{labels[index]} {confidence_text}"

        cv2.putText(imgOutput, text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (255, 0, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 2)

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)
