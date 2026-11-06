@echo off
REM Installs local 3D modelling (Hunyuan3D-2) and image generation. NVIDIA GPU only.
REM Textures also need the CUDA Toolkit and Visual Studio Build Tools (C++).
cd /d "%~dp0"
where nvidia-smi >nul 2>nul || (echo No NVIDIA GPU found. Use the cloud notebook instead, see README. & pause & exit /b 1)
if not exist .venv (echo Run start.bat once first. & pause & exit /b 1)
call .venv\Scripts\activate.bat
pip install diffusers
if not exist vendor mkdir vendor
if not exist vendor\Hunyuan3D-2 git clone --depth 1 https://github.com/Tencent/Hunyuan3D-2 vendor\Hunyuan3D-2
pip install -r vendor\Hunyuan3D-2\requirements.txt
pip install -e vendor\Hunyuan3D-2
pushd vendor\Hunyuan3D-2\hy3dgen\texgen\custom_rasterizer
pip install --no-build-isolation . || echo Texture renderer failed to build, models will be untextured.
popd
pushd vendor\Hunyuan3D-2\hy3dgen\texgen\differentiable_renderer
pip install --no-build-isolation .
popd
echo Done. Restart Octopus.
pause
