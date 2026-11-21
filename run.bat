@echo off
rem One-click launcher.
rem  - Runs the PyInstaller build if it exists (dist\vcam-clone\vcam-clone.exe).
rem  - Otherwise sets up the venv + models on first use and runs from source.
setlocal
cd /d "%~dp0"

if exist "dist\vcam-clone\vcam-clone.exe" (
    start "" "dist\vcam-clone\vcam-clone.exe" %*
    exit /b 0
)

if not exist ".venv\Scripts\python.exe" (
    echo [run] First run: creating virtual environment...
    py -3.12 -m venv .venv || py -3.11 -m venv .venv || goto :fail
    .venv\Scripts\python.exe -m pip install --upgrade pip
    .venv\Scripts\python.exe -m pip install -r requirements.txt || goto :fail
)
if not exist "models\rvm_mobilenetv3_fp32.onnx" (
    echo [run] Downloading models...
    .venv\Scripts\python.exe scripts\download_models.py || goto :fail
)
start "" ".venv\Scripts\pythonw.exe" -m vcam %*
exit /b 0

:fail
echo [run] Setup failed. See the messages above.
pause
exit /b 1
