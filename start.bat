@echo off
setlocal
title Octopus
cd /d "%~dp0"
echo.
echo   Octopus is starting. The first run installs everything it needs, so it can take 5 to 15 minutes.
echo.

REM ---------- 1. Python ----------
set "PY="
call :findpython
if not defined PY call :installpython
if not defined PY (
  echo.
  echo Couldn't install Python automatically. Install Python 3.12 from python.org, tick "Add Python to PATH", then run start.bat again.
  start "" https://www.python.org/downloads/
  pause
  exit /b 1
)

REM ---------- 2. Octopus's own Python environment ----------
set NVIDIA=0
where nvidia-smi >nul 2>nul && set NVIDIA=1
if not exist ".venv\Scripts\python.exe" (
  echo Setting up Octopus's private Python environment...
  "%PY%" -m venv .venv || goto :fail
)
set "VPY=%~dp0.venv\Scripts\python.exe"
copy /b requirements.txt+requirements-gpu.txt "%TEMP%\octopus-reqs.txt" >nul
fc /b "%TEMP%\octopus-reqs.txt" ".venv\octopus-reqs.txt" >nul 2>nul && goto :deps_ok
echo Installing Octopus's libraries...
"%VPY%" -m pip install --upgrade pip --quiet --disable-pip-version-check
if "%NVIDIA%"=="1" (
  echo NVIDIA GPU found: installing PyTorch with CUDA and the training libraries, a few GB.
  "%VPY%" -m pip install torch --index-url https://download.pytorch.org/whl/cu124 || goto :fail
  "%VPY%" -m pip install -r requirements-gpu.txt || goto :fail
) else (
  "%VPY%" -m pip install -r requirements.txt || goto :fail
)
copy /y "%TEMP%\octopus-reqs.txt" ".venv\octopus-reqs.txt" >nul
:deps_ok

REM ---------- 3. Ollama runs chat on computers without an NVIDIA GPU ----------
if "%NVIDIA%"=="1" goto :run
call :findollama
if not defined OLLAMA call :installollama
if not defined OLLAMA (
  echo.
  echo Couldn't install Ollama automatically. Chat needs it: download it from https://ollama.com/download and run start.bat again.
  start "" https://ollama.com/download
  goto :run
)
"%OLLAMA%" list >nul 2>nul && goto :run
if exist "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe" (
  start "" "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe"
) else (
  start "Ollama" /min "%OLLAMA%" serve
)

:run
echo.
echo   Opening Octopus in your browser. Keep this window open while you use it.
echo.
start "" cmd /c "timeout /t 6 >nul & start http://localhost:8000"
"%VPY%" -m octopus
pause
exit /b 0

:fail
echo.
echo Setup didn't finish. Check your internet connection and run start.bat again.
echo If it keeps failing, delete the .venv folder next to start.bat and try once more.
pause
exit /b 1

REM ---------- helpers ----------
:findpython
for %%V in (3.12 3.11 3.10 3.13) do (
  if not defined PY (
    py -%%V -c "import sys" >nul 2>nul && for /f "delims=" %%P in ('py -%%V -c "import sys;print(sys.executable)"') do set "PY=%%P"
  )
)
if defined PY exit /b 0
for %%C in (python python3) do (
  if not defined PY (
    %%C -c "import sys;sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,13) else 1)" >nul 2>nul && for /f "delims=" %%P in ('%%C -c "import sys;print(sys.executable)"') do set "PY=%%P"
  )
)
if defined PY exit /b 0
for %%D in (Python312 Python311 Python310 Python313) do (
  if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\%%D\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\%%D\python.exe"
)
exit /b 0

:installpython
echo Python isn't installed. Installing Python 3.12 for you...
where winget >nul 2>nul && winget install -e --id Python.Python.3.12 --scope user --silent --accept-package-agreements --accept-source-agreements
call :findpython
if defined PY exit /b 0
echo Downloading Python from python.org...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -UseBasicParsing -Uri 'https://www.python.org/ftp/python/3.12.8/python-3.12.8-amd64.exe' -OutFile (Join-Path $env:TEMP 'octopus-python.exe')"
if exist "%TEMP%\octopus-python.exe" "%TEMP%\octopus-python.exe" /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1 Include_test=0
call :findpython
exit /b 0

:findollama
set "OLLAMA="
for /f "delims=" %%P in ('where ollama 2^>nul') do if not defined OLLAMA set "OLLAMA=%%P"
if not defined OLLAMA if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" set "OLLAMA=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
exit /b 0

:installollama
echo Installing Ollama, which runs Octopus's chat model on your computer...
where winget >nul 2>nul && winget install -e --id Ollama.Ollama --silent --accept-package-agreements --accept-source-agreements
call :findollama
if defined OLLAMA exit /b 0
echo Downloading Ollama from ollama.com...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -UseBasicParsing -Uri 'https://ollama.com/download/OllamaSetup.exe' -OutFile (Join-Path $env:TEMP 'OllamaSetup.exe')"
if exist "%TEMP%\OllamaSetup.exe" "%TEMP%\OllamaSetup.exe" /VERYSILENT /NORESTART /SUPPRESSMSGBOXES
call :findollama
exit /b 0
