# AI-diffusion Studio - Local Personal Version Handoff

## Overview
自分専用のAI画像・動画生成アプリ。ComfyUIバックエンド + Gradio UI。
制限なし（NSFW/R18対応）。ローカルMac MPS + RunPod Cloud GPU切り替え。
Flux.1対応（RunPod時）。

---

## Directory Structure

```
AI-diffusion/
├── app/                          # Gradio UI + Python
│   ├── main.py                   # メインUI（全タブ定義、イベントハンドラ）
│   ├── config.py                 # 設定管理、モデルスキャン
│   ├── comfyui_api.py            # ComfyUI API通信、ワークフロー生成
│   ├── runpod_manager.py         # RunPod Cloud GPU管理（GraphQL API）
│   ├── civitai_api.py            # CivitAI連携（検索/DL/アップロード）
│   ├── ai_assistant.py           # AI Assistant（Claude/OpenAI/Grok）
│   ├── guide.py                  # ガイド・プロンプトテンプレート
│   ├── settings.json             # 設定ファイル（APIキー含む）
│   └── venv/                     # Python 3.12 仮想環境
├── comfyui/                      # ComfyUI本体
│   ├── main.py                   # ComfyUIエントリポイント
│   ├── venv/                     # ComfyUI用Python 3.12 venv
│   ├── custom_nodes/
│   │   ├── ComfyUI-AnimateDiff-Evolved/  # 動画生成用
│   │   │   └── models/mm_sd_v15_v2.ckpt  # Motion Model (1.8GB)
│   │   ├── ComfyUI-VideoHelperSuite/     # 動画出力(GIF/MP4)
│   │   └── ComfyUI-Impact-Pack/          # FaceDetailer等
│   └── input/                    # img2vid用入力画像
├── models/                       # 共有モデルフォルダ（symlinks）
│   ├── checkpoints/              # 34モデル
│   ├── loras/                    # 101 LoRA
│   ├── vae/                      # 3 VAE
│   ├── controlnet/               # 3モデル
│   ├── embeddings/
│   ├── upscale_models/           # 4x-UltraSharp.pth
│   ├── diffusion_models/         # Flux UNET等
│   ├── clip/                     # CLIP/T5テキストエンコーダ
│   └── animatediff_models/
├── outputs/
│   ├── normal/                   # SFW出力
│   └── adult/                    # NSFW出力
├── sd-webui-forge/               # Forge（スタンドアロン起動）
├── launch.command                # メインメニューランチャー
├── launch_studio.command         # ComfyUI + Gradio 同時起動
├── launch_comfyui.command        # ComfyUI単体起動
└── launch_forge.command          # Forge単体起動
```

---

## How to Start

### 方法1: ランチャーから（推奨）
```bash
# Finderで launch.command をダブルクリック
# または:
cd ~/Desktop/アプリ開発プロジェクト/AI-diffusion
bash launch.command
```
メニューが表示される → 1 (Studio) を選択

### 方法2: 個別起動
```bash
# ComfyUI起動
cd ~/Desktop/アプリ開発プロジェクト/AI-diffusion/comfyui
./venv/bin/python3 main.py --force-fp16 --preview-method auto --listen 127.0.0.1

# Gradio UI起動（別ターミナル）
cd ~/Desktop/アプリ開発プロジェクト/AI-diffusion/app
./venv/bin/python3 main.py
```

### アクセスURL
- **Gradio UI**: http://localhost:7860/
- **ComfyUI**: http://127.0.0.1:8188/

---

## UI Tabs

| Tab | 機能 | 備考 |
|-----|------|------|
| **Quick (Normal)** | 画像生成（SFW） | Hires Fix対応 |
| **Advanced (ComfyUI)** | ComfyUIネイティブUI | iframe埋め込み |
| **Adult (R18)** | NSFW画像生成 | Hires Fix対応、出力先別フォルダ |
| **Video** | 動画生成 | txt2vid / img2vid / vid2vid |
| **Flux (RunPod)** | Flux.1 高品質生成 | RunPod時のみ |
| **CivitAI** | モデル検索/DL/アップロード | API Key必要 |
| **AI Assistant** | AI相談（3プロバイダー） | Claude/OpenAI/Grok |
| **Prompt Generator** | タグビルダー / AI自動生成 | 日本語入力→英語プロンプト |
| **Guide** | ガイド・テンプレート | プロンプトの書き方等 |
| **Settings** | 設定・RunPod管理 | Backend切り替え、APIキー |

---

## Key Features

### Backend切り替え（画面上部）
- `local`: Mac MPS (M3 Pro) でローカル生成
- `runpod`: RunPod Cloud GPU で高速生成
- 全タブ（Quick/Adult/Video/Flux）に共通で反映

### Hires Fix（顔・ディテール改善）
- Quick/Adult タブの「Hires Fix」アコーディオンから有効化
- 低解像度で生成→アップスケール→再描画の2パス
- **4x-UltraSharp**モデル使用（高品質）or Latent Upscale（軽量）
- 倍率(1.25-2.0x)、denoise(0.2-0.7)、steps調整可能
- 顔の崩れを劇的に改善

### Flux.1（RunPod専用）
- SD 1.5/SDXLを凌ぐ最高品質の画像生成
- ネガティブプロンプト不要、自然な文章でOK
- Guidance 3.5が標準（CFGとは別物）
- 顔・手がSD系より圧倒的に綺麗
- 必要モデル: flux1-dev-fp8 + clip_l + t5xxl_fp8 + ae.safetensors

### 動画生成（Video Tab）
- **txt2vid**: テキスト → 動画（AnimateDiff）
- **img2vid**: 画像 → アニメーション
- **vid2vid**: 既存動画 → AI変換（スタイル変換、アニメ化等）
- SD1.5モデルのみ対応（SDXLは非対応）
- Motion Model: `mm_sd_v15_v2.ckpt`（インストール済み）
- 出力: GIF / MP4 / WebP 選択可

### Prompt Generator
- **Tag Builder**: カテゴリ別チェックボックスでプロンプト組み立て
  - 品質/被写体/顔/髪/体型・服装/ポーズ/背景/照明/カメラ/スタイル
- **AI Prompt**: 日本語でイメージ入力→AIが最適な英語プロンプト+設定を自動生成

### AI Assistant
- **Claude** (Anthropic): 高品質アドバイス
- **OpenAI** (GPT): 一般相談
- **Grok** (xAI): NSFW対応アドバイス
- モデル選択可: auto / claude-opus / sonnet / haiku / gpt-4o / grok-3 等
- クイック質問ボタン10個

### RunPod Cloud GPU
- Settings Tab から起動/停止
- **自動フォールバック**: 選択GPUが在庫切れなら15種類を安い順に自動試行
- **空きGPU確認ボタン**: 利用可能なGPUと料金を一覧表示
- **ノンブロッキング起動**: Pod作成後即応答、「状態確認」で接続
- 起動後 Backend を `runpod` に切り替え

### Samplers / Schedulers
- **44サンプラー**: euler, dpmpp_2m, cfg_pp系, res_multistep, sa_solver等
- **9スケジューラー**: normal, karras, exponential, linear_quadratic, kl_optimal等

---

## Settings / API Keys

`app/settings.json` に保存。Settings Tab から変更可能。

| Key | Provider | Status |
|-----|----------|--------|
| `runpod_api_key` | RunPod | 設定済み |
| `civitai_api_key` | CivitAI | 設定済み |
| `anthropic_api_key` | Claude | 設定済み |
| `openai_api_key` | OpenAI (GPT) | 設定済み |
| `xai_api_key` | xAI (Grok) | 設定済み |

---

## Models Location

ローカル: `AI-diffusion/models/`
Google Drive: symlink で連携済み
```
~/Library/CloudStorage/GoogleDrive-japanesebusinessman4@gmail.com/マイドライブ/ComfyUI/models
```

---

## Flux モデルセットアップ（RunPod）

RunPod Pod の Web Terminal で実行:
```bash
# Flux.1 dev fp8 (~12GB)
cd /workspace/ComfyUI/models/diffusion_models
wget https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors

# CLIP-L (~250MB)
cd /workspace/ComfyUI/models/clip
wget https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors

# T5-XXL fp8 (~5GB)
wget https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors

# Flux VAE (~300MB)
cd /workspace/ComfyUI/models/vae
wget https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors
```
Volume に保存されるので、Pod 停止→再起動しても消えない。

---

## Key Technical Details

### Python Environment
- Python 3.12（3.14はForge/Pillow非互換で使用不可）
- app/venv, comfyui/venv, sd-webui-forge/venv が各々独立

### ComfyUI Launch Options
```
--force-fp16        # FP16強制（MPS向け）
--preview-method auto
--listen 127.0.0.1  # ローカルのみ
```

### Custom Nodes
- **ComfyUI-AnimateDiff-Evolved**: 動画生成（txt2vid/img2vid/vid2vid）
- **ComfyUI-VideoHelperSuite**: 動画入出力（VHS_LoadVideoPath, VHS_VideoCombine）
- **ComfyUI-Impact-Pack**: FaceDetailer、SAM等（V8.28.2）

### Port
- Gradio: 7860（7000はmacOS AirPlay Receiverが使用中）
- ComfyUI: 8188

### Performance (Mac M3 Pro, 18GB)
| Task | SD1.5 | SDXL | Flux |
|------|-------|------|------|
| 画像1枚 | 15-30秒 | 30-60秒 | RunPodのみ |
| Hires Fix | +20-40秒 | +30-60秒 | RunPodのみ |
| 動画16f | 5-15分 | N/A | N/A |

メモリ空きが5GB以下だと大幅に遅くなる。不要なアプリを閉じると改善。

---

## Pending / TODO

- [x] Prompt Generator（タグビルダー + AI変換）
- [x] Samplers/Schedulers を ComfyUI の全サンプラーに更新（22→44/7→9）
- [x] Hires Fix 実装（顔・ディテール改善）
- [x] vid2vid（既存動画のAI変換）
- [x] Flux.1 対応（RunPod専用）
- [x] RunPod GPU自動フォールバック + 空きGPU確認
- [x] RunPodノンブロッキング起動
- [ ] クイック質問ボタンが model_override を渡すよう修正済み（要テスト）
- [ ] Prompt Templates（guide.py）を Quick/Adult タブに直接反映するボタン
- [ ] バッチ生成・バッチアップロード機能
- [ ] 生成履歴・ギャラリー管理
- [ ] ComfyUI ワークフロー保存/ロード機能
- [ ] FaceDetailer ワークフロー統合（Impact Pack インストール済み）

---

## Related: Commercial Version

商用版は別フォルダ:
```
~/Desktop/アプリ開発プロジェクト/ai-studio/
```
- ROADMAP.md に実装計画あり
- legal.py に法的フレームワーク（日英、地域別ルール）
- ローカル版のコードをベースに、FastAPI + Next.js に移行予定

---

## Troubleshooting

### ComfyUI が起動しない
```bash
cd ~/Desktop/アプリ開発プロジェクト/AI-diffusion/comfyui
./venv/bin/python3 main.py --force-fp16 --listen 127.0.0.1
# エラーメッセージを確認
```

### Gradio UI が起動しない
```bash
cd ~/Desktop/アプリ開発プロジェクト/AI-diffusion/app
./venv/bin/python3 main.py
# エラーメッセージを確認
```

### 生成が異常に遅い
- `curl -s http://127.0.0.1:8188/system_stats | python3 -m json.tool` で空きメモリ確認
- vram_free が 5GB 以下なら不要アプリを閉じる
- SDXLモデルはローカルで非常に重い → RunPod推奨

### ポート競合
```bash
lsof -ti:7860 | xargs kill -9  # Gradio
lsof -ti:8188 | xargs kill -9  # ComfyUI
```

### RunPod 全GPU在庫切れ
- 「空きGPU確認」ボタンで利用可能GPUをチェック
- 15種類を安い順に自動フォールバック
- 日本の朝〜昼が空きやすい（米国が深夜）
- 数分〜数時間で空く場合が多い
