"""
photo.py
---------

Componente Streamlit que permite subir una imagen y clasificarla
con el mismo modelo ASL que usamos en tiempo real (elegido)

Hay varios ejemplos en la carpeta data/ejemplos para probar

Flujo:
1. El usuario sube: .jpg .jepg o .png
2. Se decodifica a BGR (OpenCV) (cuando hagamos la llamada a classify_image() se establecera como RGB)
3. classify_image() devuelve label, confidence
4. se muestra la foto, prediciion y una barra de progreso.
5. opcion para descargar la imagen original

Autor: Hugo Gómez Martín
Contacto: hgm1001@alu.ubu.es
Fecha: 05/07/2025
"""

import io
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from handsign_asl_detection.classifier.classifier_photo import classify_image


def photo_section(model_path: str | Path):
    """
    Renderiza la seccion reconocimiento for foto
    """
    st.header("📷 Reconocimiento por foto")
    up = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])  # extensiones permitidas
    if up is None:  # nada subido, salir
        return

    # Decodificamos la imagen
    img_bgr = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)  # buffer en memoria + 3 canales
    if img_bgr is None:  # archivo corrupto
        st.error("No se pudo leer la imagen.")
        return

    # la clasificamos
    label, conf = classify_image(img_bgr, model_path)

    # UI de resultado
    st.image(img_bgr[:, :, ::-1], caption="Imagen subida", use_container_width=True)  # de BGR a RGB para Streamlit
    st.markdown(f"### Predicción: **{label}** &nbsp; ({conf:.1f} %)")  # Imprimimos resultados
    st.progress(int(conf))  # barra (0-100)

    # Botón de descarga opcional
    ok, enc = cv2.imencode(".png", img_bgr)
    if ok:
        st.download_button(
            "Descargar imagen",
            data=io.BytesIO(enc.tobytes()).getvalue(),
            file_name="uploaded.png",
            mime="image/png",
        )
