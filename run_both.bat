@echo off
cd /d "%~dp0"
echo Installing requirements (first run only takes a minute)...
python -m pip install -q -r requirements.txt
echo Starting the web app (port 5000) and the API (port 8000) in two windows...
start "CrystalScout web app" cmd /k python -m webapp.app
start "CrystalScout API" cmd /k python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
timeout /t 5 >nul
start "" http://127.0.0.1:5000
start "" http://127.0.0.1:8000/docs
