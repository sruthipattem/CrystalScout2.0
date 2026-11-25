@echo off
cd /d "%~dp0"
echo Installing requirements (first run only takes a minute)...
python -m pip install -q -r requirements.txt
echo Starting the CrystalScout API at http://127.0.0.1:8000  (interactive docs: /docs)
start "" http://127.0.0.1:8000/docs
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
pause
