"""
realtime.py
------------
Componente Streamlit que muestra el reconocimiento ASL en tiempo real

Flujo:
1. Abre la cámara
2. Lanza el clasificador sobre cada frame
3. Muestra FPS, uso de CPU/RAM y temperatura sonbre la Raspberry Pi
4. Incluye logica start/stop y liberacion de recursos

Autor: Hugo Gómez Martín
Contacto: hgm1001@alu.ubu.es
Fecha: 05/07/2025
"""

import platform
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import psutil
import streamlit as st

from handsign_asl_detection.classifier.classifier_rpi import RealTimeASLClassifier

# Constantes
METRICS_BUFFER = deque(maxlen=120)  # guardamos las ultimas 120 muestras


# Leemos la temperatura (funciona solo en Raspberry)
def _read_temp() -> Optional[float]:
    """
    Lee la temperara de la CPU en RPI
    retorna none si no existe (ej. windows)
    """
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return float(f.read()) / 1000.0
    except FileNotFoundError:
        return None


def initialize_camera():
    """
    Abre la cámara de forma cross-platfoem
    - Windows -> se usa backend DirectShow
    - Linux -> backend por defecto (V4L2)
    - Si el índice 0 falla, prueba indice 1 (segunda cámara USB)
    """
    # Inicializa cámara compatible Win / Linux (igual que antes)
    if platform.system() == "Windows":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # evita delay con DirectShow
    else:
        cap = cv2.VideoCapture(0)  # linux/macOS

    if not cap.isOpened():  # si resulta que no se abrió
        cap = cv2.VideoCapture(1)  # probamos con el siguiente indice
    return cap


# sección pública
def realtime_section(model_path):
    """
    Renderiza la interfaz completa de tiempo real.
    Se llama desde app.py y comparte st.sessionstate para mantener
    la cámara y el clasificador vivos entre recargas del navegador.
    """
    # Muestra toda la interfaz de tiempo real
    st.header("🔴 Reconocimiento en tiempo real")

    # Estado persistente para la cámara
    if "camera_active" not in st.session_state:
        st.session_state.camera_active = False

    # Recursos
    if "camera_resources" not in st.session_state:
        st.session_state.camera_resources = {"cap": None, "clf": None}

    # Contenedores UI
    btn_container = st.empty()
    status_container = st.empty()
    frame_container = st.empty()
    kpi_container = st.empty()
    chart_container = st.empty()

    if not st.session_state.camera_active:
        status_container.info("👆 Presione 'Iniciar cámara' para comenzar")

    # Botón principal
    if btn_container.button("▶️ Iniciar cámara", key="main_btn", disabled=st.session_state.camera_active):
        st.session_state.camera_active = True  # levantamos bandera

    # bucle principal
    if st.session_state.camera_active:
        # Mensaje inicial mientras tarda en arrancar
        with status_container:
            st.info("🔄 Iniciando cámara... Por favor espere")
            time.sleep(0.5)

        try:

            # camara
            if st.session_state.camera_resources["cap"] is None:
                cap = initialize_camera()  # abre dispositivo

                if not cap.isOpened():  # prueba final
                    raise RuntimeError("❌ No se pudo abrir la cámara. Asegúrese de que está conectada.")

                # Forzamos resolucion (evita latencia alta)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                st.session_state.camera_resources["cap"] = cap
            else:
                cap = st.session_state.camera_resources["cap"]

            # clasificador
            if st.session_state.camera_resources["clf"] is None:
                clf = RealTimeASLClassifier(model_path=model_path)

                # Calentamiento
                dummy = np.zeros((480, 640, 3), np.uint8)
                clf.classify_frame(dummy)

                st.session_state.camera_resources["clf"] = clf
            else:
                clf = st.session_state.camera_resources["clf"]

            status_container.success("✅ Cámara lista - Detectando señas...")

            # Botón detener (aparece solo si esta activo)
            if btn_container.button("⏹️ Detener cámara", key="stop_btn"):
                st.session_state.camera_active = False  # rompe el bucle

                # Liberar recursos inmediatamente
                if st.session_state.camera_resources["cap"]:
                    st.session_state.camera_resources["cap"].release()
                    st.session_state.camera_resources["cap"] = None

                if st.session_state.camera_resources["clf"]:
                    if hasattr(st.session_state.camera_resources["clf"], "close"):
                        st.session_state.camera_resources["clf"].close()
                    st.session_state.camera_resources["clf"] = None

                # Limpiar UI
                frame_container.empty()
                kpi_container.empty()
                chart_container.empty()
                status_container.empty()

                st.rerun()  # refrescamos la pagina

            # Bucle principal
            prev_time = time.time()  # para los FPS
            chart_interval = 2  # segundos de actualización de la gráfica
            last_chart_update = 0

            while st.session_state.camera_active:
                ret, frame = cap.read()
                if not ret:  # cámara desconectada
                    status_container.error("❌ Error de captura de cámara")
                    time.sleep(2)
                    break

                # clasificación
                processed_frame, label, conf = clf.classify_frame(frame)

                # FPS
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
                prev_time = curr_time

                # Pintamos FPS sobre imagen (doble borde)
                cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 6)
                cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Mostramos el frame en streamlit
                frame_container.image(processed_frame, channels="BGR")

                # Métricas del sistema
                cpu = psutil.cpu_percent(interval=None)
                ram = psutil.virtual_memory().percent
                temp = _read_temp()  # puede ser none (windows)

                METRICS_BUFFER.append({
                    "timestamp": curr_time,
                    "fps": fps,
                    "cpu": cpu,
                    "ram": ram,
                    "temp": temp
                })

                # en columnas
                if METRICS_BUFFER:
                    latest = METRICS_BUFFER[-1]
                    kpi_container.empty()
                    with kpi_container.container():
                        cols = st.columns(4)
                        cols[0].metric("FPS", f"{latest['fps']:.1f}")
                        cols[1].metric("CPU", f"{latest['cpu']:.1f}%")
                        cols[2].metric("RAM", f"{latest['ram']:.1f}%")
                        cols[3].metric("Temperatura",
                                       f"{latest['temp']:.1f}°C" if latest['temp'] is not None else "N/D")

                # línea de tiempo cada 2 segundos
                if curr_time - last_chart_update >= chart_interval:
                    last_chart_update = curr_time  # proximo punto de control
                    chart_container.empty()
                    with chart_container.container():
                        df = pd.DataFrame(METRICS_BUFFER).set_index("timestamp")
                        st.line_chart(df[["fps", "cpu", "ram"]])

        except Exception as e:
            status_container.error(f"❌ Error crítico: {str(e)}")
            st.session_state.camera_active = False

            # Liberar recursos en caso de error
            if st.session_state.camera_resources["cap"]:
                st.session_state.camera_resources["cap"].release()
                st.session_state.camera_resources["cap"] = None
            if st.session_state.camera_resources["clf"]:
                if hasattr(st.session_state.camera_resources["clf"], "close"):
                    st.session_state.camera_resources["clf"].close()
                st.session_state.camera_resources["clf"] = None
    else:
        # Estado inical, detenido
        status_container.info("👆 Presione 'Iniciar cámara' para comenzar")
