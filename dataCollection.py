import os
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

while True:
    folder_name = input("Introduce la letra que quieres guardar (A-Z): ").strip().upper()
    if len(folder_name) == 1 and folder_name.isalpha():
        break
    print("Entrada no válida. Por favor, introduce una única letra (A-Z).")

# Crear la carpeta si no existe
folder = f"Data/{folder_name}"
os.makedirs(folder, exist_ok=True)
print(f"Las imágenes se guardarán en la carpeta: {folder}")

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

count = 0
max_images = 1500

while True:
    success, img = cap.read()
    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        #imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h/w

        if aspectRatio > 1:
            k = imgSize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop,(wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal)/2)
            imgWhite[:, wGap:wCal+wGap] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        count += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(f"Image saved: {count}/{max_images}")
        if count >= max_images:
            print("Limit of images reached.")
            break

cap.release()
cv2.destroyAllWindows()