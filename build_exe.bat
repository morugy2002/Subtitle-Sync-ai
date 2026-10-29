@echo off
setlocal
cd /d "%~dp0"
echo ================================================
echo  SubtitleSync AI - build exe
echo ================================================

echo [1/5] Creating a clean virtual environment (important: no torch inside!)
if exist build_venv rmdir /s /q build_venv
python -m venv build_venv || goto :err
call build_venv\Scripts\activate.bat
python -m pip install --upgrade pip || goto :err
pip install -r requirements.txt pyinstaller || goto :err

echo [2/5] Downloading the MiniLM model (about 120 MB, skipped if already downloaded)
python download_models.py || goto :err

echo [3/5] Building the program
pyinstaller --noconfirm --clean --windowed --name SubtitleSyncAI ^
  --collect-binaries onnxruntime --hidden-import charset_normalizer ^
  --exclude-module matplotlib --exclude-module scipy --exclude-module pandas --exclude-module torch ^
  app.py || goto :err

echo [4/5] Copying the model next to the exe
xcopy /E /I /Y models\minilm dist\SubtitleSyncAI\models\minilm >nul || goto :err

echo [5/5] Creating the ZIP to share
if exist SubtitleSyncAI.zip del SubtitleSyncAI.zip
powershell -NoProfile -Command "Compress-Archive -Path 'dist\SubtitleSyncAI' -DestinationPath 'SubtitleSyncAI.zip' -Force" || goto :err

echo.
echo DONE.  Program: dist\SubtitleSyncAI\SubtitleSyncAI.exe
echo        ZIP to share: SubtitleSyncAI.zip
pause
exit /b 0

:err
echo.
echo *** Build failed. Copy the messages above and send them to get help. ***
pause
exit /b 1
