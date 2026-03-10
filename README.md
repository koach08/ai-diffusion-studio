# AI-diffusion Studio

**ローカルAI画像・動画生成ツール / Local AI image & video generation tool**

ComfyUI バックエンド + Gradio フロントエンドによる、統合 AI 画像・動画生成デスクトップアプリケーション。ローカル GPU (Apple Silicon MPS / NVIDIA CUDA) とクラウド GPU (RunPod) をワンクリックで切り替え可能。

A unified AI image & video generation desktop application with ComfyUI backend and Gradio frontend. Seamlessly switch between local GPU (Apple Silicon MPS / NVIDIA CUDA) and cloud GPU (RunPod) with one click.

---

## Features / 主な機能

- **34+ checkpoint models** / 34種以上のチェックポイントモデル対応
- **101 LoRA support** / 101個の LoRA サポート
- **ComfyUI integration** / ComfyUI ネイティブ統合
- **Multi-backend** / マルチバックエンド対応
  - Local (MPS / CUDA) / ローカル GPU
  - RunPod Cloud GPU / クラウド GPU
  - fal.ai, Replicate, Together.ai, Dezgo, Novita.ai (API)
- **Image generation** / 画像生成 (txt2img, Hires Fix)
- **Video generation** / 動画生成 (txt2vid, img2vid, vid2vid via AnimateDiff)
- **Flux.1 support** / Flux.1 対応 (Dev, Schnell, Pro)
- **CivitAI integration** / CivitAI 連携 (モデル検索・ダウンロード)
- **AI Assistant** / AI アシスタント (Claude, OpenAI, Grok)
- **Prompt Generator** / プロンプト生成 (タグビルダー + AI 自動生成)
- **44 samplers, 9 schedulers** / 44 サンプラー・9 スケジューラー

## UI Tabs / UIタブ

| Tab | Description |
|-----|-------------|
| **Quick** | Standard image generation with Hires Fix |
| **Advanced** | ComfyUI native node-based UI (iframe) |
| **Adult** | NSFW image generation (separate output folder) |
| **Video** | Video generation (txt2vid / img2vid / vid2vid) |
| **Flux (Cloud)** | Flux.1 high-quality generation via cloud APIs |
| **CivitAI** | Model search, download, and upload |
| **AI Assistant** | AI-powered advice (Claude / OpenAI / Grok) |
| **Prompt Generator** | Tag builder + AI prompt generation |
| **Guide** | Usage guide and prompt templates |
| **Settings** | Configuration, backend switching, RunPod management |

---

## Tech Stack / 技術スタック

- **Frontend**: [Gradio](https://gradio.app/) (Python)
- **Backend**: [ComfyUI](https://github.com/comfyanonymous/ComfyUI) (local), cloud API providers
- **Language**: Python 3.12
- **Image Generation**: Stable Diffusion 1.5 / SDXL / Flux.1
- **Video Generation**: AnimateDiff (via ComfyUI custom nodes)
- **Cloud GPU**: RunPod (GraphQL API)
- **AI APIs**: Anthropic Claude, OpenAI, xAI Grok, fal.ai, Replicate, Together.ai, Dezgo, Novita.ai

---

## Directory Structure / ディレクトリ構成

```
AI-diffusion/
├── app/                        # Gradio UI + Python application
│   ├── main.py                 # Main UI (all tabs, event handlers)
│   ├── config.py               # Configuration management, model scanning
│   ├── comfyui_api.py          # ComfyUI API communication, workflow generation
│   ├── runpod_manager.py       # RunPod Cloud GPU management (GraphQL API)
│   ├── civitai_api.py          # CivitAI integration (search/download/upload)
│   ├── ai_assistant.py         # AI Assistant (Claude/OpenAI/Grok)
│   ├── fal_api.py              # fal.ai API client (Flux image & video)
│   ├── replicate_api.py        # Replicate API client
│   ├── together_api.py         # Together.ai API client
│   ├── dezgo_api.py            # Dezgo API client
│   ├── novita_api.py           # Novita.ai API client
│   ├── guide.py                # Guide & prompt templates
│   ├── settings.json.example   # Settings template (copy to settings.json)
│   └── venv/                   # Python virtual environment
├── comfyui/                    # ComfyUI installation (not included)
├── models/                     # Model files (not included, download separately)
│   ├── checkpoints/
│   ├── loras/
│   ├── vae/
│   ├── controlnet/
│   ├── upscale_models/
│   └── ...
├── outputs/                    # Generated images/videos
├── launch.command              # Main launcher menu
├── launch_studio.command       # ComfyUI + Gradio simultaneous launch
├── launch_comfyui.command      # ComfyUI standalone launch
└── launch_forge.command        # SD WebUI Forge standalone launch
```

---

## Setup / セットアップ

### Prerequisites / 前提条件

- Python 3.12
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) installed in `comfyui/`
- macOS (Apple Silicon recommended) or Linux with NVIDIA GPU

### Installation / インストール

```bash
# 1. Clone this repository
git clone https://github.com/koach08/ai-diffusion-studio.git
cd ai-diffusion-studio

# 2. Create Python virtual environment
cd app
python3.12 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install gradio Pillow

# 4. Set up ComfyUI (in project root)
cd ..
git clone https://github.com/comfyanonymous/ComfyUI.git comfyui
cd comfyui
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 5. Configure settings
cd ../app
cp settings.json.example settings.json
# Edit settings.json with your API keys

# 6. Download models
# Place checkpoint models in models/checkpoints/
# Place LoRA files in models/loras/
# Place VAE files in models/vae/
```

### Running / 起動

```bash
# Option 1: Use the launcher (recommended)
bash launch.command
# Select 's' for Studio mode

# Option 2: Launch manually
# Terminal 1 - Start ComfyUI
cd comfyui
./venv/bin/python3 main.py --force-fp16 --preview-method auto --listen 127.0.0.1

# Terminal 2 - Start Gradio UI
cd app
./venv/bin/python3 main.py
```

### Access / アクセス

- **Studio UI**: http://localhost:7860
- **ComfyUI**: http://127.0.0.1:8188

---

## API Keys / APIキー設定

Settings tab or `app/settings.json` で設定。全てオプションです。

| Key | Provider | Required for |
|-----|----------|-------------|
| `runpod_api_key` | [RunPod](https://runpod.io/) | Cloud GPU |
| `civitai_api_key` | [CivitAI](https://civitai.com/) | Model search/download |
| `anthropic_api_key` | [Anthropic](https://anthropic.com/) | AI Assistant (Claude) |
| `openai_api_key` | [OpenAI](https://openai.com/) | AI Assistant (GPT) |
| `xai_api_key` | [xAI](https://x.ai/) | AI Assistant (Grok) |
| `replicate_api_key` | [Replicate](https://replicate.com/) | Cloud image generation |
| `fal_api_key` | [fal.ai](https://fal.ai/) | Cloud image/video generation |
| `together_api_key` | [Together.ai](https://together.ai/) | Cloud image generation |
| `dezgo_api_key` | [Dezgo](https://dezgo.com/) | Cloud image/video generation |
| `novita_api_key` | [Novita.ai](https://novita.ai/) | Cloud image generation |

---

## Performance / パフォーマンス

| Task | SD 1.5 | SDXL | Flux.1 |
|------|--------|------|--------|
| Image (1 pic) | 15-30s | 30-60s | Cloud only |
| + Hires Fix | +20-40s | +30-60s | Cloud only |
| Video (16 frames) | 5-15min | N/A | N/A |

*Tested on Mac M3 Pro, 18GB RAM*

---

## License / ライセンス

MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
