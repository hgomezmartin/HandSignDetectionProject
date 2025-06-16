import io
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from handsign_asl_detection.classifier.classifier_photo import classify_image


def photo_section(model_path: str | Path):
    st.header("📷 Reconocimiento por foto")
    up = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
    if up is None:
        return

    img_bgr = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        st.error("No se pudo leer la imagen.")
        return

    label, conf = classify_image(img_bgr, model_path)

    # UI
    st.image(img_bgr[:, :, ::-1], caption="Imagen subida", use_container_width=True)
    st.markdown(f"### Predicción: **{label}** &nbsp; ({conf:.1f} %)")
    st.progress(int(conf))

    # descarga opcional
    ok, enc = cv2.imencode(".png", img_bgr)
    if ok:
        st.download_button(
            "Descargar imagen",
            data=io.BytesIO(enc.tobytes()).getvalue(),
            file_name="uploaded.png",
            mime="image/png",
        )
