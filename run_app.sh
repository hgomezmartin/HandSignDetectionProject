#!/bin/bash

set -e

cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src:$PYTHONPATH"

python3 -m pip install --upgrade --no-cache-dir pip
python3 -m pip install --no-cache-dir -r requirements.txt

streamlit run src/handsign_asl_detection/web/app.py