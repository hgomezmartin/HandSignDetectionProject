import os
import shutil
import sys

import numpy as np
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    img_to_array,
    load_img,
    save_img
)


def augment_dataset(src_dir, dst_dir):
    """
    Copia todas las imágenes desde src_dir hasta dst_dir,
    conservando la estructura de subcarpetas, y para cada imagen
    genera 2 copias aumentadas (rotación, zoom, brillo).

    NOTA: No se aplica ninguna baraja/mezcla adicional.
    """
    # Verificar si la carpeta de origen existe
    if not os.path.isdir(src_dir):
        print(f"Error: El directorio de origen '{src_dir}' no existe o no es una carpeta.")
        sys.exit(1)

    # Si dst_dir existe, lo borramos entero para comenzar limpio
    if os.path.exists(dst_dir):
        print(f"Borrando carpeta existente: {dst_dir}")
        shutil.rmtree(dst_dir)

    # Crear la carpeta de destino vacía
    os.makedirs(dst_dir, exist_ok=True)

    # Definir el generador de data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,  # Rotación ±10°
        zoom_range=0.1,  # Zoom ±10%
        brightness_range=[0.8, 1.2],  # Ajuste de brillo entre 80% y 120%
        fill_mode='nearest'
        # No se incluyen shifts o flips en este ejemplo
    )

    # Listar subcarpetas (clases) en src_dir
    classes = sorted(os.listdir(src_dir))

    for cls_name in classes:
        cls_folder_src = os.path.join(src_dir, cls_name)

        # Omitir si no es una carpeta
        if not os.path.isdir(cls_folder_src):
            continue

        # Crear la carpeta de destino para esa clase
        cls_folder_dst = os.path.join(dst_dir, cls_name)
        os.makedirs(cls_folder_dst, exist_ok=True)

        # Listar imágenes en la carpeta origen (SIN barajar)
        all_files = os.listdir(cls_folder_src)
        # Filtrar extensiones de imagen
        valid_ext = (".jpg", ".jpeg", ".png")
        image_files = [f for f in all_files if f.lower().endswith(valid_ext)]

        print(f"\nClase '{cls_name}': {len(image_files)} imágenes encontradas. Generando aumentos...")

        for i, fname in enumerate(image_files):
            src_path = os.path.join(cls_folder_src, fname)

            # Extraer extensión para mantenerla
            _, ext = os.path.splitext(fname)
            ext = ext.lower()

            # 1) Copiar la imagen original
            #    Puedes mantener el nombre original o renombrarlo si lo prefieres
            original_fname = f"img_{i:05d}{ext}"
            dst_path_original = os.path.join(cls_folder_dst, original_fname)
            shutil.copy2(src_path, dst_path_original)

            # 2) Cargar la imagen en un array
            img = load_img(src_path)  # Carga como objeto PIL
            x = img_to_array(img)  # Convierte a np.array
            x = np.expand_dims(x, axis=0)  # shape (1, h, w, 3)

            # Generar 2 imágenes aumentadas
            aug_iter = datagen.flow(x, batch_size=1)

            for aug_index in range(2):
                # Genera un batch de tamaño 1 (una imagen aumentada)
                batch = next(aug_iter)
                # Quitar la dimensión extra
                aug_img = batch[0].astype('uint8')

                # Guardar con nombre nuevo
                aug_fname = f"img_{i:05d}_aug{aug_index + 1}{ext}"
                dst_path_aug = os.path.join(cls_folder_dst, aug_fname)
                save_img(dst_path_aug, aug_img)

        print(f" -> Clase '{cls_name}': completado. "
              f"Ahora hay {len(os.listdir(cls_folder_dst))} imágenes en la carpeta de destino.")

    print("\n¡Proceso de data augmentation completado!")
    print(f"Se ha creado '{dst_dir}' con 3x más imágenes por clase (1 original + 2 augmentations).")


if __name__ == "__main__":
    src_directory = "Data/Data_disordered"  # Carpeta origen con subcarpetas A-Z
    dst_directory = "Data/Data_disordered_augmented"  # Carpeta destino

    augment_dataset(src_directory, dst_directory)
