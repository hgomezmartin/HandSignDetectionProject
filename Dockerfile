# Imagen base ligera
FROM python:3.11-slim

# librerías del sistema para OpenCV necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
        v4l-utils libv4l-dev \
    && rm -rf /var/lib/apt/lists/*

# carpeta de trabajo
WORKDIR /app

# 4 ─ Instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# 5 Copiar el resto del proyecto
COPY . .

# 6 Añadir ./src al path de Python
ENV PYTHONPATH=/app/src \
    OPENCV_VIDEOIO_PRIORITY_BACKEND=V4L2

# 7 Puerto de Streamlit (leerá 8502 de .streamlit/config.toml)
EXPOSE 8502

# 8 ─ Comando de arranque
CMD ["streamlit", "run", "src/handsign_asl_detection/web/app.py"]
