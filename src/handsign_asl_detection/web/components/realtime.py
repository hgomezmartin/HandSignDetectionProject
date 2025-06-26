from __future__ import annotations

import os
import time
from collections import deque
from typing import Deque

import av
import cv2
import pandas as pd
import psutil
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

from handsign_asl_detection.classifier.classifier_rpi import RealTimeASLClassifier

METRICS_BUFFER: Deque[dict] = deque(maxlen=120)

IN_CLOUD = os.environ.get("PORT", "8501") == "8501"


def _initialize_local_camera() -> cv2.VideoCapture:
    """Devuelve un cv2.VideoCapture abierto o lanza RuntimeError."""
    # 0 para USB / 0 con CAP_DSHOW en Windows
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise RuntimeError("❌ No se pudo abrir la cámara. Asegúrese de que está conectada.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def _local_loop(model_path):
    st.info("📷 Cámara local iniciada")
    cap = _initialize_local_camera()
    clf = RealTimeASLClassifier(model_path=model_path)

    prev = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Error de captura")
            break

        processed, label, conf = clf.classify_frame(frame)

        fps = 1 / (time.time() - prev)
        prev = time.time()
        cv2.putText(processed, f"{label} {conf:.0%}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(processed, f"FPS {fps:.1f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        st.image(processed, channels="BGR")

        # métricas
        METRICS_BUFFER.append({
            "timestamp": time.time(),
            "fps": fps,
            "cpu": psutil.cpu_percent(),
            "ram": psutil.virtual_memory().percent
        })

        if not st.session_state.get("camera_active", True):
            break

    cap.release()
    st.success("🛑 Cámara detenida")


class ASLProcessor(VideoProcessorBase):
    def __init__(self, model_path: str):
        self.clf = RealTimeASLClassifier(model_path=model_path)
        self.prev = time.time()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        processed, label, conf = self.clf.classify_frame(img)

        fps = 1 / (time.time() - self.prev)
        self.prev = time.time()

        cv2.putText(processed, f"{label} {conf:.0%}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(processed, f"FPS {fps:.1f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        METRICS_BUFFER.append({
            "timestamp": time.time(),
            "fps": fps,
            "cpu": psutil.cpu_percent(),
            "ram": psutil.virtual_memory().percent
        })

        return av.VideoFrame.from_ndarray(processed, format="bgr24")


def _webrtc_loop(model_path):
    webrtc_streamer(
        key="asl-webrtc",
        video_processor_factory=lambda: ASLProcessor(model_path),
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )
    st.caption("La detección ocurre localmente en tu navegador; ningún vídeo se sube al servidor.")


def realtime_section(model_path):
    st.header("🔴 Reconocimiento en tiempo real")

    if "camera_active" not in st.session_state:
        st.session_state.camera_active = False

    if not st.session_state.camera_active:
        if st.button("▶️ Iniciar cámara"):
            st.session_state.camera_active = True
            st.rerun()
    else:
        if st.button("⏹️ Detener cámara"):
            st.session_state.camera_active = False
            st.rerun()

    if st.session_state.camera_active:
        if IN_CLOUD:
            _webrtc_loop(model_path)
        else:
            _local_loop(model_path)

        if METRICS_BUFFER:
            df = pd.DataFrame(METRICS_BUFFER).set_index("timestamp")
            st.line_chart(df[["fps", "cpu", "ram"]])
