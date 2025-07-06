"""
collection.py
--------------

Captura en tiempo rela las imagenes de la mano del usuario a través
de la cámara y las guarda preprocesaadas para ampliar el conjunto de
datos del entrenamiento estableciendo un límite para entrenar el modelo
ASL.

Flujo:
1. El usuario introduce la etiqueta destino (letra) que da nombre a la
carpeta.
2. Se inicailiza la cámara y el detector de manos (HandTrackingModule)
3. Para cada frame válido, se recorta la ROI de la mano, se reescala
sobre un lienzo blanco cuadrado de tamaño ImageSize (300x300) y se
muestra
4. Al pulsar "S" se guarda la imagen procesada en el disco.
5. El porceso termina al alcanzar "MAX_IMAGES_DS" (500)

Autor: Hugo Gómez Martín
Contacto: hgm1001@alu.ubu.es
Fecha: 05/07/2025
"""

import math
import os
import time

import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

from handsign_asl_detection.config import IMG_SIZE_DS, OFFSET, MAX_IMAGES_DS, ORDERED_DATADIR


def collect_dataset():
    """
    Inicia la captura de imágenes para una clase concreta (A-Z).

    El usuario debe pulsar "S" para guardar ejemplos preporcesados,
    si mantenemos pulsado, este creará una ráfaga de imágenes para
    un guardado rápido.

    """

    # Se solicita la etiqueta destino
    while True:
        # Convertimos la entrada a mayusculas
        folder_name = input("Introduce la letra que quieres guardar (A-Z) o (0-9): ").strip().upper()
        # Solo se admiten Letras y números (números para un dataset futuro)
        if len(folder_name) == 1 and folder_name.isalnum():  # Alfanuméricos y un único caracter
            break
        print("Entrada no válida. Por favor, introduce una única letra (A-Z) o número (0-9).")

    # Creamos la carpeta destino
    folder = ORDERED_DATADIR / folder_name
    os.makedirs(folder, exist_ok=True)
    print(f"Las imágenes se guardarán en la carpeta: {folder}")

    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)
    count = 0

    # Bucle principal
    while True:
        success, img = cap.read()  # Frame crudo
        if not success:
            print("No se pudo leer de la cámara.")
            break

        # Establecemos un lienzo blanco cuadrado donde colocaremos la mano
        imgWhite = np.ones((IMG_SIZE_DS, IMG_SIZE_DS, 3), np.uint8) * 255

        # Detectamos la mano y anotamos sobre la imagen
        hands, img = detector.findHands(img)  # "img" es la versión con landmarks
        if hands:
            # Asumimos la mano detectada
            hand = hands[0]
            x, y, w, h = hand['bbox']  # Bounding-Box (x,y, ancho, alto)

            # Recortamos con margen
            imgCrop = img[
                      max(0, y - OFFSET):y + h + OFFSET,
                      max(0, x - OFFSET):x + w + OFFSET
                      ]
            aspectRatio = h / w

            # Normalizamos a un lienzo cuadrado
            if aspectRatio > 1:
                # Mano alta -> ajustamos alto a IMG_SIZE_DS = 300
                k = IMG_SIZE_DS / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, IMG_SIZE_DS))
                wGap = math.ceil((IMG_SIZE_DS - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize

            else:
                # Mano ancha -> ajustamos ancho a IMG_SIZE_DS = 300
                k = IMG_SIZE_DS / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (IMG_SIZE_DS, hCal))
                hGap = math.ceil((IMG_SIZE_DS - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # Ventanas auxiliares para la depuración
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

        # Ventana principal con landmarks
        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == ord("s"):
            # Guardamos la imagen normalizada
            count += 1
            cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
            print(f"Image saved: {count}/{MAX_IMAGES_DS}")
            if count >= MAX_IMAGES_DS:
                print("Límite de imágenes alcanzado.")
                break

    # Liberamos los recursos
    cap.release()
    cv2.destroyAllWindows()


def main_menu():
    """Menu interactivo de consola para lanzar la recogida de datos"""
    while True:
        print("\nMenú principal")
        print("1. Introducir un nuevo Dataset para una letra")
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


# Punto de entrada
if __name__ == "__main__":
    main_menu()
