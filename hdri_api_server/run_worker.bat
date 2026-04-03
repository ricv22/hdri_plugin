@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Missing .venv — run: py -3.12 -m venv .venv
  echo Then: .\.venv\Scripts\pip.exe install -r requirements.txt
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m uvicorn examples.comfyui_worker:app --host 127.0.0.1 --port 8001
pause
