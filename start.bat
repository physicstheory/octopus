@echo off
REM Octopus launcher for Windows. First run installs everything (a few GB).
cd /d "%~dp0"
if not exist .venv (
  echo First run: setting up Octopus, this takes a few minutes...
  py -3 -m venv .venv || python -m venv .venv
  call .venv\Scripts\activate.bat
  python -m pip install --upgrade pip
  where nvidia-smi >nul 2>nul && (
    echo NVIDIA GPU found, installing PyTorch with CUDA
    pip install torch --index-url https://download.pytorch.org/whl/cu124
  ) || (
    pip install torch
  )
  pip install -r requirements.txt
) else (
  call .venv\Scripts\activate.bat
)
start "" cmd /c "timeout /t 5 >nul & start http://localhost:8000"
python -m octopus
pause
