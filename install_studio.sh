#!/usr/bin/env bash
# Installs local 3D modelling (Hunyuan3D-2) and image generation. NVIDIA GPU only.
# Without an NVIDIA GPU, use the free cloud notebook instead (see README).
set -e
cd "$(dirname "$0")"
command -v nvidia-smi >/dev/null 2>&1 || { echo "No NVIDIA GPU found. Use the cloud notebook: colab/Octopus_Cloud.ipynb"; exit 1; }
[ -d .venv ] || { echo "Run ./start.sh once first."; exit 1; }
. .venv/bin/activate
pip install diffusers
mkdir -p vendor
[ -d vendor/Hunyuan3D-2 ] || git clone --depth 1 https://github.com/Tencent/Hunyuan3D-2 vendor/Hunyuan3D-2
pip install -r vendor/Hunyuan3D-2/requirements.txt
pip install -e vendor/Hunyuan3D-2
echo "Building the texture renderer (needs the CUDA toolkit: nvcc)…"
( cd vendor/Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer && pip install --no-build-isolation . ) || echo "Texture renderer failed to build; models will be untextured."
( cd vendor/Hunyuan3D-2/hy3dgen/texgen/differentiable_renderer && pip install --no-build-isolation . ) || true
echo "✅ Done. Restart Octopus."
