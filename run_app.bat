@echo off
REM nos colocamos en la carpeta raiz
cd /d "%~dp0"

REM instala/actualiza dependencias (idempotente)
python -m pip install --upgrade pip
pip install -r requirements.txt

REM lanzamos la app
python -m streamlit run src\handsign_asl_detection\web\app.py
pause
