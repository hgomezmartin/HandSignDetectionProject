@echo off
cd /d "%~dp0"                          

set "PYTHONPATH=%CD%\src"            

python -m pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

python -m streamlit run src\handsign_asl_detection\web\app.py
pause
