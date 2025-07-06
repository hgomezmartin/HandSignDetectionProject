@echo off
cd /d "%~dp0"

if not exist ".venv\" (
    echo [INFO] Creando entorno virtual .venv...
    python -m venv .venv
)

call ".venv\Scripts\activate"

python -m pip install --upgrade pip
python -m pip install -e .
pip install -r requirements.txt

streamlit run src\handsign_asl_detection\web\app.py
pause
