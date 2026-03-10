#!/bin/bash
# ComfyUI Launcher for macOS (Apple Silicon MPS)
cd "$(dirname "$0")/comfyui"
echo "========================================="
echo "  ComfyUI - Starting on MPS (Metal)..."
echo "========================================="
echo ""
./venv/bin/python3 main.py --force-fp16 --preview-method auto
