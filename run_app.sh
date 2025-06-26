#!/usr/bin/env bash
set -e                        # aborta ante cualquier error

# 1. Posiciónate en la carpeta del script
cd "$(dirname "$0")"

# 2. Crea (si no existe) y activa el entorno virtual .venv
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate

# 3. Instala dependencias y paquete
pip install -e .
pip install -r requirements.txt


# 4. Lanza Streamlit
exec streamlit run src/handsign_asl_detection/web/app.py