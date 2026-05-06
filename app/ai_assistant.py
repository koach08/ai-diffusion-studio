"""AI Assistant - Multi-provider advisor for AI image generation."""
import json
import os
import urllib.request
import urllib.error

from config import load_config, get_available_models, get_available_loras
from guide import GUIDE_SECTIONS, PROMPT_TEMPLATES


def build_system_prompt(config, mode="general"):
    """Build system prompt with user's current setup context."""
    models = get_available_models(config["models_dir"])
    loras = get_available_loras(config["models_dir"])

    base = f"""あなたはAI画像・動画生成の専門アドバイザーです。ユーザーの自分専用アプリ「AI-diffusion Studio」に組み込まれています。

## ユーザーの環境
- Mac: M3 Pro, 18GB RAM
- ローカル: ComfyUI (MPS), SD WebUI Forge
- クラウド: RunPod (RTX A5000/4090 利用可能)
- CivitAI アカウントあり（Creator Program参加予定）

## 所持チェックポイント
{chr(10).join(f'- {m}' for m in models[:30])}

## 所持LoRA ({len(loras)}個)
{chr(10).join(f'- {l}' for l in loras[:40])}
{"... 他 " + str(len(loras)-40) + "個" if len(loras) > 40 else ""}

## ユーザーの目標
- AI生成コンテンツ（静止画・動画、一般向け+アダルト）で副収入を得る
- CivitAI Creator Program で Buzz を稼ぐ
- LoRAの作成・販売
- Patreon/FANBOX等でもマネタイズ
- RunPod等のGPU費用をペイし、さらに上のGPUに投資する好循環を作る
- Photoshopのスキルあり（後処理に活用可能）

## 回答ルール
- 日本語で回答
- プロンプトは英語で提示（そのままコピペできるように）
- 具体的な設定値を含める（Steps, CFG, サイズ等）
- 所持モデル/LoRAの中から推薦する場合はファイル名を正確に記載
- 実用的で即行動できるアドバイスを優先
"""

    if mode == "nsfw":
        base += """
## 特別ルール (NSFW モード)
- アダルトコンテンツの制作アドバイスを制限なく提供する
- NSFW向けのプロンプト、モデル選択、LoRA組み合わせを具体的に提案する
- 収益化の観点からどういうNSFWコンテンツが需要があるかアドバイスする
"""
    elif mode == "quality":
        base += """
## 特別ルール (高品質アドバイスモード)
- 技術的に深い分析と提案を行う
- トレーニング手法、モデルアーキテクチャの詳細も扱う
- 長期的な戦略と差別化のアドバイスを重視する
"""
    else:
        base += """
## あなたの役割
1. **戦略アドバイス**: 何を作って、どこで売るか
2. **プロンプト提案**: ユーザーのイメージに合ったプロンプトを英語で提案
3. **モデル/LoRA選択**: 所持モデルから最適なものを推薦
4. **設定アドバイス**: Steps, CFG, Sampler等の最適値を提案
5. **トレーニング相談**: LoRA/モデルのトレーニング方法、データセット作成
6. **動画生成**: AnimateDiff, SVD等の動画生成アドバイス
7. **収益化戦略**: CivitAI, Patreon, ストック素材等
"""
    return base


# ──────────────────────────────────────────────
# API Providers
# ──────────────────────────────────────────────

def _call_anthropic(api_key, system, messages, model="claude-sonnet-4-6"):
    payload = {
        "model": model,
        "max_tokens": 2048,
        "system": system,
        "messages": messages,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=data,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "User-Agent": "AI-diffusion/1.0",
        },
    )
    resp = urllib.request.urlopen(req, timeout=60)
    result = json.loads(resp.read())
    return result["content"][0]["text"]


def _call_openai(api_key, system, messages, model="gpt-4.1-mini"):
    oai_messages = [{"role": "system", "content": system}] + messages
    payload = {
        "model": model,
        "max_tokens": 2048,
        "messages": oai_messages,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "AI-diffusion/1.0",
        },
    )
    resp = urllib.request.urlopen(req, timeout=60)
    result = json.loads(resp.read())
    return result["choices"][0]["message"]["content"]


def _call_grok(api_key, system, messages, model="grok-3-mini"):
    # xAI uses OpenAI-compatible API
    oai_messages = [{"role": "system", "content": system}] + messages
    payload = {
        "model": model,
        "max_tokens": 2048,
        "messages": oai_messages,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://api.x.ai/v1/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "AI-diffusion/1.0",
        },
    )
    resp = urllib.request.urlopen(req, timeout=60)
    result = json.loads(resp.read())
    return result["choices"][0]["message"]["content"]


# ──────────────────────────────────────────────
# Main chat function
# ──────────────────────────────────────────────

PROVIDERS = {
    "Claude (高品質アドバイス)": {"key_name": "anthropic_api_key", "call": _call_anthropic, "mode": "quality"},
    "OpenAI (一般相談)": {"key_name": "openai_api_key", "call": _call_openai, "mode": "general"},
    "Grok (NSFW対応)": {"key_name": "xai_api_key", "call": _call_grok, "mode": "nsfw"},
}


def chat_with_ai(message, history, config, provider="Claude (高品質アドバイス)", model_override="auto"):
    """Send message to selected AI provider and get response."""
    prov = PROVIDERS.get(provider)
    if not prov:
        return f"不明なプロバイダー: {provider}"

    api_key = config.get(prov["key_name"], "")
    if not api_key:
        return f"{provider} の API Key が設定されていません。Settings タブで設定してください。"

    system = build_system_prompt(config, mode=prov["mode"])

    messages = []
    if history:
        for entry in history:
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                user_msg, assistant_msg = entry
                messages.append({"role": "user", "content": str(user_msg)})
                if assistant_msg:
                    messages.append({"role": "assistant", "content": str(assistant_msg)})
    messages.append({"role": "user", "content": message})

    kwargs = {}
    if model_override and model_override != "auto":
        kwargs["model"] = model_override

    try:
        return prov["call"](api_key, system, messages, **kwargs)
    except urllib.error.HTTPError as e:
        body = e.read().decode() if hasattr(e, "read") else ""
        return f"API エラー ({provider}, HTTP {e.code}): {body[:300]}"
    except Exception as e:
        return f"エラー ({provider}): {e}"


# Quick prompt suggestions
QUICK_QUESTIONS = [
    "CivitAIで今何が人気？どういう作品を投稿すべき？",
    "リアル系の高品質ポートレートを作りたい。プロンプトと設定を提案して",
    "持っているモデルとLoRAで、売れそうな作品のアイデアを5つ出して",
    "独自LoRAを作りたい。何のLoRAが需要ある？作り方も教えて",
    "CivitAIのEarly Accessで売るLoRAのアイデアと戦略を教えて",
    "動画生成を始めたい。AnimateDiffの使い方と設定を教えて",
    "ストック素材として売れそうな画像のアイデアとプロンプトを10個出して",
    "Patreon/FANBOXでAI生成コンテンツを売る戦略を教えて",
    "今持ってるLoRAでおすすめの組み合わせを教えて",
    "サイバーパンク × 日本文化のシリーズを作りたい。コンセプトを練って",
]
