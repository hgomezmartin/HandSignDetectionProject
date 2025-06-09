import os
import random
import shutil
import sys

from handsign_asl_detection.config import ORDERED_DATADIR, DISORDERED_DATADIR

SOURCE_DIR = ORDERED_DATADIR
DESTINY_DIR = DISORDERED_DATADIR


def copy_and_shuffle_images():
    # Verificar que src_dir existe
    if not os.path.isdir(SOURCE_DIR):
        print(f"Error: El directorio de origen '{SOURCE_DIR}' no existe o no es una carpeta.")
        sys.exit(1)

    # Si dst_dir existe, lo borramos entero
    if os.path.exists(DESTINY_DIR):
        print(f"Borrando carpeta existente: {DESTINY_DIR}")
        shutil.rmtree(DESTINY_DIR)

    # Crear la carpeta de destino vacía
    os.makedirs(DESTINY_DIR, exist_ok=True)

    # Listar subcarpetas (clases) en src_dir
    classes = sorted(os.listdir(SOURCE_DIR))

    for cls_name in classes:
        cls_folder_src = os.path.join(SOURCE_DIR, cls_name)

        # Omitir si no es una carpeta
        if not os.path.isdir(cls_folder_src):
            continue

        # Crear la carpeta de destino para esa clase
        cls_folder_dst = os.path.join(DESTINY_DIR, cls_name)
        os.makedirs(cls_folder_dst, exist_ok=True)

        # Listar imágenes en la carpeta origen
        all_files = os.listdir(cls_folder_src)

        # Filtrar solo extensiones de imagen
        valid_ext = (".jpg", ".jpeg", ".png")
        image_files = [f for f in all_files
                       if f.lower().endswith(valid_ext)]

        # Barajar la lista de ficheros
        random.shuffle(image_files)

        print(f"Clase '{cls_name}': Encontradas {len(image_files)} imágenes. Copiando y barajando...")

        # Copiar cada imagen con un nombre nuevo shuffled_XXXXX.ext
        for i, fname in enumerate(image_files):
            src_path = os.path.join(cls_folder_src, fname)

            # Extraer extensión
            _, ext = os.path.splitext(fname)
            ext = ext.lower()  # normalizar

            # Construir nuevo nombre
            new_fname = f"shuffled_{i:05d}{ext}"
            dst_path = os.path.join(cls_folder_dst, new_fname)

            shutil.copy2(src_path, dst_path)  # copy2 para preservar metadata si se quiere

    print("\n¡Proceso completado!")
    print(f"Se ha creado '{DESTINY_DIR}' con las mismas subcarpetas que '{SOURCE_DIR}', pero con imágenes barajadas.")


if __name__ == "__main__":
    copy_and_shuffle_images()
