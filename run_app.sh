#!/usr/bin/env bash
set -e                        # aborta ante cualquier error

# carpeta del script
cd "$(dirname "$0")"

# 2. Crea (si no existe) y activa el entorno virtual .venv
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate

# Instala dependencias y paquete
pip install --upgrade pip
pip install -e .
pip install -r requirements.txt


# Lanza Streamlit
exec streamlit run src/handsign_asl_detection/web/app.py