"""
disordered.py
--------------

Copia todas las imagenes del dataset ordenado a uno desordenado, barajando
el orden de los ficheros y renombrandolos con la forma:
    -   "shuffled_XXXXX.ext"

Basicamente se utiliza para evitar sesgos de lectura secuencial cuando se
entrenan los modleos no complicando mucho el código del propio modelo

Autor: Hugo Gómez Martín
Contacto: hgm1001@alu.ubu.es
Fecha: 05/07/2025
"""

import os
import random
import shutil
import sys

from handsign_asl_detection.config import ORDERED_DATADIR, DISORDERED_DATADIR

SOURCE_DIR = ORDERED_DATADIR  # Carpeta ordenada origen
DESTINY_DIR = DISORDERED_DATADIR  # Carpeta desordenada destino


def copy_and_shuffle_images():
    """
    Copia el datasset ordenado y baraaja los nombres del archivo dentro de
    cada clase/carpeta

    Se conservan las subcarpetas pero se eliminan los nombres originales
    de las imágenes, sustituyendolos por: "shuffled_XXXXX.ext"

    """
    # Verificar que el directorio origen existe
    if not os.path.isdir(SOURCE_DIR):
        print(f"Error: El directorio de origen '{SOURCE_DIR}' no existe o no es una carpeta.")
        sys.exit(1)

    # Si la carpeta destino existe, lo borramos entero para garantiza un orden nuevo limpio
    if os.path.exists(DESTINY_DIR):
        print(f"Borrando carpeta existente: {DESTINY_DIR}")
        shutil.rmtree(DESTINY_DIR)

    # Creamos la carpeta de destino vacía
    os.makedirs(DESTINY_DIR, exist_ok=True)

    # Listmosr subcarpetas (clases) en el directorio destino
    classes = sorted(os.listdir(SOURCE_DIR))

    # Bucle sobre cada clase (A-Z)
    for cls_name in classes:
        cls_folder_src = os.path.join(SOURCE_DIR, cls_name)

        # Omitimos si no es una carpeta
        if not os.path.isdir(cls_folder_src):
            continue

        # Creamos la carpeta de destino para esa clase
        cls_folder_dst = os.path.join(DESTINY_DIR, cls_name)
        os.makedirs(cls_folder_dst, exist_ok=True)

        # Listamos imágenes en la carpeta origen
        all_files = os.listdir(cls_folder_src)

        # Filtramos solo extensiones de imagen
        valid_ext = (".jpg", ".jpeg", ".png")
        image_files = [f for f in all_files
                       if f.lower().endswith(valid_ext)]

        # Barajamos la lista de ficheros
        random.shuffle(image_files)

        print(f"Clase '{cls_name}': Encontradas {len(image_files)} imágenes. Copiando y barajando...")

        # Copiamos cada imagen con un nombre nuevo shuffled_XXXXX.ext
        for i, fname in enumerate(image_files):
            src_path = os.path.join(cls_folder_src, fname)

            # Extraermos extensión
            _, ext = os.path.splitext(fname)  # solo la extensión
            ext = ext.lower()  # normalizar a minusculas

            # Construir nuevo nombre
            new_fname = f"shuffled_{i:05d}{ext}"  # 5 dígitos (suficiente para datasets futuros)
            dst_path = os.path.join(cls_folder_dst, new_fname)

            shutil.copy2(src_path, dst_path)  # copy2 para preservar metadata si se quiere

    print("\n¡Proceso completado!")
    print(f"Se ha creado '{DESTINY_DIR}' con las mismas subcarpetas que '{SOURCE_DIR}', pero con imágenes barajadas.")


# Punto de entrada
if __name__ == "__main__":
    copy_and_shuffle_images()
