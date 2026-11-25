@echo off
cd /d "%~dp0"
echo Installing requirements (first run only takes a minute)...
python -m pip install -q -r requirements.txt
echo Starting CrystalScout at http://127.0.0.1:5000
start "" http://127.0.0.1:5000
python -m webapp.app
pause
