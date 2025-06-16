import streamlit as st
from streamlit_option_menu import option_menu

from handsign_asl_detection.config import TEACHABLE_TFL_DIR, STATIC_DIR
from handsign_asl_detection.web.components.photo import photo_section
from handsign_asl_detection.web.components.realtime import realtime_section
from handsign_asl_detection.web.components.trainer import trainer_v2_section

# Configuración base
st.set_page_config(
    page_title="Hand Sign Detection",
    page_icon="✌️",
    layout="wide"
)

# Carga tu CSS si lo necesitas
css_path = STATIC_DIR / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text('utf-8')}</style>", unsafe_allow_html=True)
else:
    st.warning(f"No encontré el CSS en: {css_path}")

# SIDEBAR CON OPTION_MENU
with st.sidebar:
    st.title("✌️ Hand Sign Detection")

    # menú principal idéntico al de streamlit-geospatial
    seleccion = option_menu(
        None,  # sin título extra
        ["Tiempo real", "Foto", "Entrenamiento"],
        icons=["camera-video", "image", "graph-up"],  # Tabler icons
        menu_icon="cast",  # icono del menú (puedes quitarlo con None)
        default_index=0,
        orientation="vertical",

    )

# Zona principal según la selección
# también puedes hacer selectbox de modelo aquí o en un expander aparte
model_opts = {
    "FP16 (rápido, menos RAM)": TEACHABLE_TFL_DIR / "keras_model_fp16.tflite",
    "FP32 (más preciso)": TEACHABLE_TFL_DIR / "keras_model_fp32.tflite",
}
model_name = st.sidebar.selectbox("🧠 Elige un modelo:", list(model_opts.keys()), index=0)
model_path = model_opts[model_name]

if seleccion == "Tiempo real":
    realtime_section(model_path)
elif seleccion == "Foto":
    photo_section(model_path)
else:
    trainer_v2_section()
