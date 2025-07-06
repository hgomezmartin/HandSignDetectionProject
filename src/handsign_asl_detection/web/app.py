"""
app.py (Interfaz Streamlit principal)
--------------------------------------

Arranca la alicación web que une la interfaz principal con los tres
pilares del proyecto:

    - Reconocer el alfabeto de señas ASL en tiempo real.
    - Reconocimiento por foto.
    - Entrenar el modelo con un dataset nuevo subiendo imágenes

Cada pilar vive en el módulo components, este archivo se limita a:

- Configurar Streamlit (Iconos, layout y CSS)
- Construir el menú lateral
- Mantener en st.session_state la ruta del modelo elegido (FP32 o FP16)
- Despachar a la sección correspondiente

Autor: Hugo Gómez Martín
Contacto: hgm1001@alu.ubu.es
Fecha: 05/07/2025
"""

import streamlit as st
from streamlit_option_menu import option_menu

from handsign_asl_detection.config import TEACHABLE_TFL_DIR, STATIC_DIR, IMG_WEB_DIR
from handsign_asl_detection.web.components.photo import photo_section
from handsign_asl_detection.web.components.realtime import realtime_section
from handsign_asl_detection.web.components.trainer import trainer_v2_section

logo_path = IMG_WEB_DIR / "logo_ASL.png"  # nuestro logo ASL

# Configuración base
st.set_page_config(
    page_title="ASL Detection",  # Título de la página
    page_icon=str(logo_path),  # nuestro logo como icono de página
    layout="wide"  # ancho completo
)

# Cargamos el CSS
css_path = STATIC_DIR / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text('utf-8')}</style>",
                unsafe_allow_html=True)  # lo insertamos como <style>...</style>


# Definimos la sección de Inicio
def home_section():
    col1, col2, col3 = st.columns([1, 1, 1])  # 3 columnas para centrar el logo
    with col2:  # usamos la central para que la imagen se centre
        st.image(str(logo_path), use_container_width=True)

    st.title("👋 ¡Bienvenido/a a ASL Detection!")
    st.markdown(
        """
        Este proyecto te permite:
        - 🔴 **Reconocer el alfabeto de señas ASL** en tiempo real.
        - 📷 **Clasificar** una imagen con un modelo pre-entrenado.
        - 🛠️ **Entrenar** tu propio modelo subiendo imágenes.

        Elige una de estas **tres** opciones representadas en el **menú lateral**.
        """
    )


# Construimos el menú lateral
with st.sidebar:
    st.title("✌️ ASL Detection")

    # menú con iconos
    seleccion = option_menu(
        None,  # sin encabezado
        ["Inicio", "Tiempo real", "Foto", "Entrenamiento"],
        icons=["house", "camera-video", "image", "graph-up"],  # iconos
        default_index=0,  # arranca en inicio
        orientation="vertical",
    )

    # Selector de modelo
    model_opts = {
        "FP16 (rápido, menos RAM)": TEACHABLE_TFL_DIR / "keras_model_fp16.tflite",
        "FP32 (más preciso)": TEACHABLE_TFL_DIR / "keras_model_fp32.tflite",
    }

    # guardamos la selección en session_state
    st.session_state.model_name = st.selectbox(
        "🧠 Elige un modelo:",
        list(model_opts.keys()),
        index=0
    )

    # ruta al .tflite elegido
    st.session_state.model_path = model_opts[st.session_state.model_name]

# Renderizamos la sección correspondiente
if seleccion == "Inicio":
    home_section()
elif seleccion == "Tiempo real":
    realtime_section(st.session_state.model_path)
elif seleccion == "Foto":
    photo_section(st.session_state.model_path)
elif seleccion == "Entrenamiento":
    trainer_v2_section()

# Footer personalizado
st.markdown(
    '<div class="custom-footer">ASL Detection Project · 2025</div>',
    unsafe_allow_html=True,
)
