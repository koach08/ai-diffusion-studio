#!/bin/bash
# SD WebUI Forge Launcher for macOS (Apple Silicon MPS)
cd "$(dirname "$0")/sd-webui-forge"
echo "========================================="
echo "  SD WebUI Forge - Starting..."
echo "  No NSFW filter / No restrictions"
echo "========================================="
echo ""
./venv/bin/python3 launch.py --skip-torch-cuda-test --no-half --use-cpu interrogate --disable-safe-unpickle
