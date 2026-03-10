#!/bin/bash
# AI-diffusion Launcher
clear
echo "========================================="
echo "     AI-diffusion Launcher"
echo "========================================="
echo ""
echo "  s) AI-diffusion Studio  (カスタムUI + ComfyUI)"
echo ""
echo "  1) ComfyUI のみ        (ノードベース・高機能)"
echo "  2) SD WebUI Forge のみ  (シンプルUI)"
echo "  3) ComfyUI + Forge 両方"
echo ""
echo "  m) models フォルダを開く"
echo "  o) outputs フォルダを開く"
echo "  q) 終了"
echo ""
echo "========================================="
echo -n "  選択 > "
read choice

BASE="$(dirname "$0")"

case $choice in
  s)
    echo ""
    echo "AI-diffusion Studio を起動中..."
    exec "$BASE/launch_studio.command"
    ;;
  1)
    echo ""
    echo "ComfyUI を起動中..."
    echo "ブラウザで http://127.0.0.1:8188 を開いてください"
    echo ""
    cd "$BASE/comfyui"
    ./venv/bin/python3 main.py --force-fp16 --preview-method auto
    ;;
  2)
    echo ""
    echo "SD WebUI Forge を起動中..."
    echo "自動的にブラウザが開きます"
    echo ""
    cd "$BASE/sd-webui-forge"
    ./venv/bin/python3 launch.py --skip-torch-cuda-test --no-half --use-cpu interrogate --disable-safe-unpickle
    ;;
  3)
    echo ""
    echo "両方を起動中..."
    echo "  ComfyUI:  http://127.0.0.1:8188"
    echo "  Forge:    http://127.0.0.1:7860"
    echo ""
    cd "$BASE/comfyui"
    ./venv/bin/python3 main.py --force-fp16 --preview-method auto &
    COMFY_PID=$!
    cd "$BASE/sd-webui-forge"
    ./venv/bin/python3 launch.py --skip-torch-cuda-test --no-half --use-cpu interrogate --disable-safe-unpickle &
    FORGE_PID=$!
    echo ""
    echo "停止するには Ctrl+C を押してください"
    trap "kill $COMFY_PID $FORGE_PID 2>/dev/null; exit" INT
    wait
    ;;
  m)
    open "$BASE/models"
    ;;
  o)
    open "$BASE/outputs"
    ;;
  q)
    exit 0
    ;;
  *)
    echo "無効な選択です"
    ;;
esac
