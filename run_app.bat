@echo off
cd /d "%~dp0"
set "PYTHONPATH=%CD%\src;%PYTHONPATH%"

python -m pip install --upgrade pip
pip install -r requirements.txt

streamlit run src\handsign_asl_detection\web\app.py
pause
