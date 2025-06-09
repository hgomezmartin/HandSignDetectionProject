import math
import os
import time

import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

from handsign_asl_detection.config import IMG_SIZE_DS, OFFSET, MAX_IMAGES_DS


def collect_dataset():
    while True:
        folder_name = input("Introduce la letra o número que quieres guardar (A-Z): ").strip().upper()
        if len(folder_name) == 1 and folder_name.isalnum():
            break
        print("Entrada no válida. Por favor, introduce una única letra (A-Z).")

    # Crear la carpeta si no existe
    folder = f"Data/{folder_name}"
    os.makedirs(folder, exist_ok=True)
    print(f"Las imágenes se guardarán en la carpeta: {folder}")

    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)

    count = 0

    while True:
        success, img = cap.read()
        imgWhite = np.ones((IMG_SIZE_DS, IMG_SIZE_DS, 3), np.uint8) * 255
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgCrop = img[y - OFFSET:y + h + OFFSET, x - OFFSET:x + w + OFFSET]
            aspectRatio = h / w

            if aspectRatio > 1:
                k = IMG_SIZE_DS / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, IMG_SIZE_DS))
                wGap = math.ceil((IMG_SIZE_DS - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize

            else:
                k = IMG_SIZE_DS / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (IMG_SIZE_DS, hCal))
                hGap = math.ceil((IMG_SIZE_DS - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == ord("s"):
            count += 1
            cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
            print(f"Image saved: {count}/{MAX_IMAGES_DS}")
            if count >= MAX_IMAGES_DS:
                print("Límite de imágenes alcanzado.")
                break

    cap.release()
    cv2.destroyAllWindows()


def main_menu():
    while True:
        print("\nMenú principal")
        print("1. Introducir un nuevo Dataset para una letra o número")
        print("2. Salir")

        choice = input("Selecciona una opción (1-2): ").strip()
        match choice:
            case "1":
                collect_dataset()
            case "2":
                print("Saliendo del programa...")
                break
            case _:
                print("Opción no válida. Por favor, selecciona 1 o 2.")


# Inicia el programa con el menú principal
if __name__ == "__main__":
    main_menu()
