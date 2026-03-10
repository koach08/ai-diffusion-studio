#!/bin/bash
# AI-diffusion Studio Launcher
# ComfyUI (backend) + Custom Gradio UI (frontend) を同時起動
BASE="$(dirname "$0")"
clear
echo "========================================="
echo "  AI-diffusion Studio"
echo "========================================="
echo ""

# Start ComfyUI in background
echo "[1/2] ComfyUI Server 起動中..."
cd "$BASE/comfyui"
./venv/bin/python3 main.py --force-fp16 --preview-method auto --listen 127.0.0.1 &
COMFY_PID=$!

# Wait for ComfyUI to be ready
echo "      ComfyUI の起動を待っています..."
for i in $(seq 1 60); do
    if curl -s http://127.0.0.1:8188/system_stats > /dev/null 2>&1; then
        echo "      ComfyUI Ready!"
        break
    fi
    sleep 2
done

# Start Custom UI
echo "[2/2] AI-diffusion Studio UI 起動中..."
cd "$BASE/app"
./venv/bin/python3 main.py &
UI_PID=$!

echo ""
echo "========================================="
echo "  起動完了!"
echo "  Studio UI:  http://localhost:7860"
echo "  ComfyUI:    http://127.0.0.1:8188"
echo "========================================="
echo ""
echo "  停止するには Ctrl+C を押してください"
echo ""

# Open browser
sleep 3
open http://localhost:7860

# Wait and cleanup
trap "kill $COMFY_PID $UI_PID 2>/dev/null; echo ''; echo 'シャットダウン完了'; exit" INT TERM
wait
