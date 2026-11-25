#!/usr/bin/env bash
cd "$(dirname "$0")"
python3 -m pip install -q -r requirements.txt
echo "CrystalScout: http://127.0.0.1:5000"
python3 -m webapp.app
