@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Missing .venv — run: py -3.12 -m venv .venv
  echo Then: .\.venv\Scripts\pip.exe install -r requirements.txt
  pause
  exit /b 1
)
set "PANORAMA_MODE=http_json"
set "PANORAMA_HTTP_URL=http://127.0.0.1:8001/v1/panorama"
set "HDRI_PUBLIC_BASE_URL=http://127.0.0.1:8000"
set "HDRI_SIGNING_SECRET=change-me"
".venv\Scripts\python.exe" -m uvicorn app:app --host 127.0.0.1 --port 8000
pause
