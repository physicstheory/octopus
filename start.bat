@echo off
REM Octopus launcher for Windows. The first run installs what your computer needs.
cd /d "%~dp0"
set NVIDIA=0
where nvidia-smi >nul 2>nul && set NVIDIA=1
if not exist .venv (
  echo First run: setting up Octopus...
  py -3 -m venv .venv || python -m venv .venv
  call .venv\Scripts\activate.bat
  python -m pip install --upgrade pip
  if "%NVIDIA%"=="1" (
    echo NVIDIA GPU found: installing PyTorch with CUDA and the training libraries, a few GB
    pip install torch --index-url https://download.pytorch.org/whl/cu124
    pip install -r requirements-gpu.txt
  ) else (
    echo No NVIDIA GPU: installing the light version. Chat runs through Ollama, training and 3D use the free cloud notebook.
    pip install -r requirements.txt
  )
) else (
  call .venv\Scripts\activate.bat
)
if "%NVIDIA%"=="0" (
  where ollama >nul 2>nul || (
    if not exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
      echo.
      echo Octopus chats through Ollama, which isn't installed yet.
      echo Download it from https://ollama.com/download, open it once, then run start.bat again.
      start "" https://ollama.com/download
    )
  )
)
start "" cmd /c "timeout /t 5 >nul & start http://localhost:8000"
python -m octopus
pause
