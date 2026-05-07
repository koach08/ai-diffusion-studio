"""Vision-based image analysis for img2vid workflows.

Mode-aware routing:
- SFW (mode="normal"): GPT-4o → Claude Opus 4.7 → Grok Vision → Florence-2
- NSFW (mode="adult"): Grok Vision → Florence-2 (skips OpenAI/Claude — they refuse)
"""
import base64
import io
import json
import os
import time
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
        img = Image.fromarray(image)

    img = img.convert("RGB")
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

def _call_openai_vision(api_key, prompt, image_b64, model="gpt-4o", max_tokens=512):
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{image_b64}",
                    "detail": "low",
                }},
            ],
        }],
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "AI-diffusion/1.0",
        },
    )
    try:
        resp = urllib.request.urlopen(req, timeout=60)
        return json.loads(resp.read())["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI Vision API Error ({e.code}): {body[:300]}")


# ──────────────────────────────────────────────
# Anthropic Claude vision
# ──────────────────────────────────────────────

def _call_claude_vision(api_key, prompt, image_b64, model="claude-opus-4-7", max_tokens=512):
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg", "data": image_b64,
                }},
                {"type": "text", "text": prompt},
            ],
        }],
    }
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "User-Agent": "AI-diffusion/1.0",
        },
    )
    try:
        resp = urllib.request.urlopen(req, timeout=60)
        return json.loads(resp.read())["content"][0]["text"].strip()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Claude Vision API Error ({e.code}): {body[:300]}")


# ──────────────────────────────────────────────
# xAI Grok vision (NSFW-tolerant, OpenAI-compatible API)
# ──────────────────────────────────────────────

def _call_grok_vision(api_key, prompt, image_b64, model="grok-4-fast-non-reasoning", max_tokens=512):
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{image_b64}",
                    "detail": "high",
                }},
            ],
        }],
    }
    req = urllib.request.Request(
        "https://api.x.ai/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "AI-diffusion/1.0",
        },
    )
    try:
        resp = urllib.request.urlopen(req, timeout=90)
        return json.loads(resp.read())["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Grok Vision API Error ({e.code}): {body[:300]}")


# ──────────────────────────────────────────────
# fal.ai Florence-2 (last-resort permissive captioner)
# ──────────────────────────────────────────────

NSFW_REFUSAL_MARKERS = (
    "i can't ", "i cannot ", "i won't ", "i'm sorry", "i am sorry",
    "i'm not able", "i am not able", "i'm unable", "unable to assist",
    "explicit sexual", "explicit content", "nudity", "pornographic",
    "sexually explicit", "against my", "policy",
    "i'd be happy to help with", "share a different image",
)


def _looks_like_refusal(text):
    if not text:
        return True
    head = text.strip().lower()[:300]
    return any(m in head for m in NSFW_REFUSAL_MARKERS)


def _fal_upload_image(api_key, image_bytes, content_type="image/jpeg", filename="image.jpg"):
    init_req = urllib.request.Request(
        "https://rest.alpha.fal.ai/storage/upload/initiate",
        data=json.dumps({"file_name": filename, "content_type": content_type}).encode("utf-8"),
        headers={"Authorization": f"Key {api_key}", "Content-Type": "application/json"},
    )
    init = json.loads(urllib.request.urlopen(init_req, timeout=30).read())
    put_req = urllib.request.Request(
        init["upload_url"], data=image_bytes, method="PUT",
        headers={"Content-Type": content_type},
    )
    urllib.request.urlopen(put_req, timeout=60)
    return init["file_url"]


def _fal_run(api_key, model_id, args, timeout=180):
    submit_req = urllib.request.Request(
        f"https://queue.fal.run/{model_id}",
        data=json.dumps(args).encode("utf-8"),
        headers={"Authorization": f"Key {api_key}", "Content-Type": "application/json"},
    )
    submitted = json.loads(urllib.request.urlopen(submit_req, timeout=30).read())
    status_url = submitted["status_url"]
    response_url = submitted["response_url"]
    deadline = time.time() + timeout
    while time.time() < deadline:
        s_req = urllib.request.Request(status_url, headers={"Authorization": f"Key {api_key}"})
        st = json.loads(urllib.request.urlopen(s_req, timeout=15).read())
        if st.get("status") == "COMPLETED":
            r_req = urllib.request.Request(response_url, headers={"Authorization": f"Key {api_key}"})
            return json.loads(urllib.request.urlopen(r_req, timeout=15).read())
        if st.get("status") in ("FAILED", "ERROR"):
            raise RuntimeError(f"fal.ai job failed: {st}")
        time.sleep(2)
    raise RuntimeError("fal.ai job timeout")


def _call_florence_caption(fal_key, image_b64):
    image_bytes = base64.b64decode(image_b64)
    url = _fal_upload_image(fal_key, image_bytes, content_type="image/jpeg")
    result = _fal_run(fal_key, "fal-ai/florence-2-large/more-detailed-caption", {"image_url": url})
    return (result.get("results") or "").strip()


# ──────────────────────────────────────────────
# Prompt templates
# ──────────────────────────────────────────────

MOTION_PROMPT_SYSTEM = """You are an expert cinematographer writing motion prompts for image-to-video AI.

Your task: Look at this image and write a SHORT natural motion description (1-2 sentences, under 200 chars) describing how the contents should subtly move to animate naturally.

CRITICAL rules:
- Keep ALL objects, characters, composition, and style EXACTLY as shown
- Describe only small natural motion (hair in breeze, subtle smile, gentle sway, cloth flutter, eye blink, steam rising, water rippling, leaves moving, slow camera push-in)
- Focus on what is PRESENT in the image, do not invent new elements
- Output ONLY the motion prompt in English, no preamble, no quotes, no explanation
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


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _florence_fallback(fal_key, image_b64, suffix):
    caption = _call_florence_caption(fal_key, image_b64)
    if not caption:
        return None
    return f"{caption.rstrip('. ')}. {suffix}"


def _try_call(name, fn, errors):
    """Run a vision call; treat refusal text as a soft failure so we move on."""
    try:
        out = fn()
        if _looks_like_refusal(out):
            errors.append(f"{name}: refused (NSFW policy)")
            return None
        return out
    except Exception as e:
        errors.append(f"{name}: {e}")
        return None


# ──────────────────────────────────────────────
# High-level analyzers — mode-aware routing
# ──────────────────────────────────────────────

def _analyze(image, system_prompt, max_tokens, suffix,
             openai_key, anthropic_key, xai_key, fal_key, mode):
    """Mode-aware vision analyzer.

    SFW path:  OpenAI → Claude → Grok → Florence-2
    NSFW path: Grok → Florence-2  (skips OpenAI/Claude — they refuse)
    """
    image_b64 = _image_to_base64_jpeg(image, max_side=1024 if max_tokens >= 400 else 768)
    errors = []
    is_nsfw = (mode == "adult")

    if not is_nsfw:
        if openai_key:
            r = _try_call("OpenAI",
                          lambda: _call_openai_vision(openai_key, system_prompt, image_b64, max_tokens=max_tokens),
                          errors)
            if r:
                return r
        if anthropic_key:
            r = _try_call("Claude",
                          lambda: _call_claude_vision(anthropic_key, system_prompt, image_b64, max_tokens=max_tokens),
                          errors)
            if r:
                return r

    if xai_key:
        r = _try_call("Grok",
                      lambda: _call_grok_vision(xai_key, system_prompt, image_b64, max_tokens=max_tokens),
                      errors)
        if r:
            return r

    if fal_key:
        try:
            out = _florence_fallback(fal_key, image_b64, suffix)
            if out:
                return out
        except Exception as e:
            errors.append(f"fal Florence-2: {e}")

    if not (openai_key or anthropic_key or xai_key or fal_key):
        raise RuntimeError("OpenAI / Anthropic / xAI / fal.ai のいずれかの API Key が必要です (Settingsタブで設定)")
    raise RuntimeError("Vision API failed: " + " | ".join(errors))


def analyze_for_motion(image, openai_key="", anthropic_key="", xai_key="", fal_key="", mode="normal"):
    """Returns a natural motion prompt (preserve mode)."""
    return _analyze(
        image, MOTION_PROMPT_SYSTEM, max_tokens=200,
        suffix="Subtle natural motion, slow cinematic push-in, soft ambient movement",
        openai_key=openai_key, anthropic_key=anthropic_key,
        xai_key=xai_key, fal_key=fal_key, mode=mode,
    )


def describe_for_inspiration(image, openai_key="", anthropic_key="", xai_key="", fal_key="", mode="normal"):
    """Returns a full scene description (inspired mode)."""
    return _analyze(
        image, INSPIRATION_SYSTEM, max_tokens=500,
        suffix="Photorealistic, cinematic lighting, shallow depth of field, professional composition",
        openai_key=openai_key, anthropic_key=anthropic_key,
        xai_key=xai_key, fal_key=fal_key, mode=mode,
    )


# ──────────────────────────────────────────────
# Motion presets
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
