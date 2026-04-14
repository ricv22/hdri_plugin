@echo off
REM ComfyUI + GMNet env (edit COMFYUI_* below if your install differs)
REM GMNet clone: https://github.com/qtlark/GMNet — place checkpoints under GMNet\checkpoints\

set "GMNET_CODES_ROOT=D:\gmnet\GMNet\codes"
set "GMNET_REPO_ROOT=D:\gmnet\GMNet"
set "GMNET_CHECKPOINT=D:\gmnet\GMNet\checkpoints\G_real.pth"

set "COMFYUI_VENV_PYTHON=D:\ComfyUI\.venv\Scripts\python.exe"
set "COMFYUI_MAIN_DIR=D:\ComfyUI\resources\ComfyUI"

if not exist "%COMFYUI_VENV_PYTHON%" (
  echo ERROR: Python not found: %COMFYUI_VENV_PYTHON%
  pause
  exit /b 1
)
if not exist "%COMFYUI_MAIN_DIR%\main.py" (
  echo ERROR: main.py not found: %COMFYUI_MAIN_DIR%\main.py
  pause
  exit /b 1
)
if not exist "%GMNET_CODES_ROOT%\models" (
  echo WARNING: GMNET_CODES_ROOT may be wrong — expected models\ under: %GMNET_CODES_ROOT%
)

cd /d "%COMFYUI_MAIN_DIR%"
"%COMFYUI_VENV_PYTHON%" main.py %*
exit /b %ERRORLEVEL%
