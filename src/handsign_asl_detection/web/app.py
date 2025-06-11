import streamlit as st

from handsign_asl_detection.config import TEACHABLE_TFL_DIR
from handsign_asl_detection.web.components.photo import photo_section
from handsign_asl_detection.web.components.realtime import realtime_section
from handsign_asl_detection.web.components.trainer import trainer_v2_section

# Configuración base
st.set_page_config(page_title="Hand Sign Detection",
                   page_icon="✌️",
                   layout="wide")
st.sidebar.title("✌️ Hand Sign Detection")

# Selección de modelo
model_opts = {
    "FP16 (más rápido, menos RAM)": TEACHABLE_TFL_DIR / "keras_model_fp16.tflite",
    "FP32 (más preciso)": TEACHABLE_TFL_DIR / "keras_model_fp32.tflite",
}
model_name = st.sidebar.radio("Modelo:", list(model_opts.keys()), index=0)
model_path = model_opts[model_name]

# Selección de modo
modo = st.sidebar.radio("", ["🔴 Tiempo real", "📷 Foto", "🧪 Entrenamiento"])

if modo == "🔴 Tiempo real":
    realtime_section(model_path)
elif modo == "📷 Foto":
    photo_section(model_path)
else:  # Entrenar
    trainer_v2_section()
