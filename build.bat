@echo off
rem Build dist\ClearFrame\ClearFrame.exe with PyInstaller and bundle the models.
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [build] Creating virtual environment...
    py -3.12 -m venv .venv || py -3.11 -m venv .venv || goto :fail
    .venv\Scripts\python.exe -m pip install --upgrade pip
    .venv\Scripts\python.exe -m pip install -r requirements.txt || goto :fail
)
.venv\Scripts\python.exe scripts\download_models.py || goto :fail
.venv\Scripts\python.exe -m PyInstaller --noconfirm --clean clearframe.spec || goto :fail

echo [build] Copying models...
if not exist "dist\ClearFrame\models" mkdir "dist\ClearFrame\models"
copy /y models\*.onnx "dist\ClearFrame\models\" >nul
copy /y models\*.tflite "dist\ClearFrame\models\" >nul

echo.
echo [build] Done: dist\ClearFrame\ClearFrame.exe
exit /b 0

:fail
echo [build] FAILED
exit /b 1
