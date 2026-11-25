#!/usr/bin/env bash
cd "$(dirname "$0")"
python3 -m pip install -q -r requirements.txt
echo "CrystalScout API at http://127.0.0.1:8000  (docs: /docs)"
python3 -m uvicorn api.main:app --host 127.0.0.1 --port 8000
