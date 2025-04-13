import os
import random
import shutil
import sys

def copy_and_shuffle_images(src_dir, dst_dir):
    """
    Copia todas las imágenes desde src_dir hasta dst_dir,
    manteniendo la estructura de subcarpetas, pero en un orden aleatorio.
    Cada imagen se renombra a shuffled_XXXXX con su extensión original.
    """
    # Verificar que src_dir existe
    if not os.path.isdir(src_dir):
        print(f"Error: El directorio de origen '{src_dir}' no existe o no es una carpeta.")
        sys.exit(1)

    # Si dst_dir existe, lo borramos entero
    if os.path.exists(dst_dir):
        print(f"Borrando carpeta existente: {dst_dir}")
        shutil.rmtree(dst_dir)

    # Crear la carpeta de destino vacía
    os.makedirs(dst_dir, exist_ok=True)

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
    print(f"Se ha creado '{dst_dir}' con las mismas subcarpetas que '{src_dir}', pero con imágenes barajadas.")


# Ejemplo de uso
if __name__ == "__main__":
    src_directory = "Data/Data_ordered"  # Tu carpeta original
    dst_directory = "Data/Data_disordered"  # Nueva carpeta con imágenes desordenadas

    copy_and_shuffle_images(src_directory, dst_directory)
