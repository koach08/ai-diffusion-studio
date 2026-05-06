"""Vision-based image analysis for img2vid workflows.

Provides two analysis modes:
- analyze_for_motion(): Extract natural motion description (preserve mode)
- describe_for_inspiration(): Extract full scene description (inspired mode)

Uses OpenAI GPT-4o or Anthropic Claude vision via urllib (no SDK dependency).
"""
import base64
import io
import json
import os
import urllib.request
import urllib.error

from PIL import Image


def _image_to_base64_jpeg(image, max_side=1024):
    """Convert PIL Image / numpy array / file path to base64 JPEG string."""
    if isinstance(image, str):
        img = Image.open(image)
    elif isinstance(image, Image.Image):
        img = image
    else:
        # numpy array
        img = Image.fromarray(image)

    img = img.convert("RGB")
    # Downscale to keep token cost low
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ──────────────────────────────────────────────
# OpenAI GPT-4o vision
# ──────────────────────────────────────────────

def _call_openai_vision(api_key, prompt, image_b64, model="gpt-4.1", max_tokens=512):
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}",
                            "detail": "low",
                        },
                    },
                ],
            }
        ],
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
    try:
        resp = urllib.request.urlopen(req, timeout=60)
        result = json.loads(resp.read())
        return result["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI Vision API Error ({e.code}): {body[:300]}")


# ──────────────────────────────────────────────
# Anthropic Claude vision
# ──────────────────────────────────────────────

def _call_claude_vision(api_key, prompt, image_b64, model="claude-sonnet-4-6", max_tokens=512):
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
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
    try:
        resp = urllib.request.urlopen(req, timeout=60)
        result = json.loads(resp.read())
        return result["content"][0]["text"].strip()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Claude Vision API Error ({e.code}): {body[:300]}")


# ──────────────────────────────────────────────
# High-level analyzers
# ──────────────────────────────────────────────

MOTION_PROMPT_SYSTEM = """You are an expert cinematographer writing motion prompts for image-to-video AI.

Your task: Look at this image and write a SHORT natural motion description (1-2 sentences, under 200 chars) describing how the contents should subtly move to animate naturally.

CRITICAL rules:
- Keep ALL objects, characters, composition, and style EXACTLY as shown
- Describe only small natural motion (hair in breeze, subtle smile, gentle sway, cloth flutter, eye blink, steam rising, water rippling, leaves moving, slow camera push-in)
- Focus on what is PRESENT in the image, do not invent new elements
- Output ONLY the motion prompt in English, no preamble, no quotes, no explanation

Examples of good output:
- "Soft breeze moving her hair gently, subtle smile forming, slow cinematic push-in"
- "Steam rising from the cup, warm light flickering, shallow depth of field"
- "Waves gently rolling in, seagulls crossing sky, slow aerial pan"
"""


INSPIRATION_SYSTEM = """You are an expert visual director. Look at this image and write a DETAILED scene description that could be used as a text-to-video prompt to recreate or reimagine the scene.

Include:
- Subject and appearance (people, objects, creatures)
- Setting and environment
- Lighting, mood, color palette
- Style (photorealistic, anime, cinematic, painterly, etc.)
- Camera angle and composition
- Suggested natural motion for a video

Output a single paragraph in English, 100-250 words, no preamble, ready to use as a video prompt.
"""


def analyze_for_motion(image, openai_key="", anthropic_key=""):
    """Analyze image and return a natural motion prompt (preserve mode).

    Returns: motion prompt string suitable for fal.ai img2vid models.
    """
    image_b64 = _image_to_base64_jpeg(image, max_side=768)

    # Prefer OpenAI for speed/cost, fall back to Claude
    errors = []
    if openai_key:
        try:
            return _call_openai_vision(openai_key, MOTION_PROMPT_SYSTEM, image_b64, max_tokens=200)
        except Exception as e:
            errors.append(f"OpenAI: {e}")

    if anthropic_key:
        try:
            return _call_claude_vision(anthropic_key, MOTION_PROMPT_SYSTEM, image_b64, max_tokens=200)
        except Exception as e:
            errors.append(f"Claude: {e}")

    if not openai_key and not anthropic_key:
        raise RuntimeError("OpenAI または Anthropic API Key が必要です (Settingsタブで設定)")
    raise RuntimeError("Vision API failed: " + " | ".join(errors))


def describe_for_inspiration(image, openai_key="", anthropic_key=""):
    """Analyze image and return a full scene description (inspired mode).

    Returns: paragraph description suitable for text-to-video prompt.
    """
    image_b64 = _image_to_base64_jpeg(image, max_side=1024)

    errors = []
    if openai_key:
        try:
            return _call_openai_vision(openai_key, INSPIRATION_SYSTEM, image_b64, max_tokens=500)
        except Exception as e:
            errors.append(f"OpenAI: {e}")

    if anthropic_key:
        try:
            return _call_claude_vision(anthropic_key, INSPIRATION_SYSTEM, image_b64, max_tokens=500)
        except Exception as e:
            errors.append(f"Claude: {e}")

    if not openai_key and not anthropic_key:
        raise RuntimeError("OpenAI または Anthropic API Key が必要です (Settingsタブで設定)")
    raise RuntimeError("Vision API failed: " + " | ".join(errors))


# ──────────────────────────────────────────────
# Motion presets (can be appended to auto-analyzed prompt)
# ──────────────────────────────────────────────

MOTION_PRESETS = {
    "(なし)": "",
    "微細な動き (Subtle)": ", subtle motion, gentle breeze, barely noticeable movement, cinematic",
    "スローモーション": ", slow motion, dreamy, ethereal, smooth movement",
    "カメラ寄せ (Push-in)": ", slow cinematic push-in camera, dolly forward, building intensity",
    "カメラ引き (Pull-out)": ", slow cinematic pull-out camera, dolly backward, reveal shot",
    "パン 左→右": ", slow camera pan from left to right, smooth tracking shot",
    "パン 右→左": ", slow camera pan from right to left, smooth tracking shot",
    "チルトアップ": ", slow camera tilt upward, revealing shot",
    "チルトダウン": ", slow camera tilt downward, descending shot",
    "オービット": ", camera orbiting around subject, 360 rotation, dynamic",
    "ズームイン": ", zoom in on main subject, increasing focus, dramatic",
    "ズームアウト": ", zoom out revealing the environment, epic scale",
    "パララックス": ", parallax effect, depth of field, 3D feel, foreground and background moving",
    "風・髪の揺れ": ", wind blowing, hair flowing, fabric fluttering, natural movement",
    "光の変化": ", changing light, sun rays shifting, golden hour glow, cinematic lighting",
    "ダイナミック": ", dynamic motion, energetic, lively, vivid action",
}
