"""fal.ai API Client - Flux image & video generation (NSFW OK) + advanced features."""
import os
import json
import time
import urllib.request
import io

try:
    import httpx
    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False

try:
    import fal_client
    _HAS_FAL = True
except ImportError:
    _HAS_FAL = False


def _fal_run(api_key, model_id, arguments, timeout=300):
    """Call fal.ai via queue API with polling (bypasses broken SDK).

    1. Submit job to queue.fal.run
    2. Poll status until COMPLETED
    3. Fetch result from the correct URL (with full model path)
    """
    headers = {"Authorization": f"Key {api_key}", "Content-Type": "application/json"}
    submit_url = f"https://queue.fal.run/{model_id}"

    if _HAS_HTTPX:
        client = httpx.Client(timeout=httpx.Timeout(60, connect=30))
    else:
        client = None

    # Submit
    if client:
        r = client.post(submit_url, json=arguments, headers=headers)
        if r.status_code != 200:
            raise RuntimeError(f"fal.ai Submit Error ({r.status_code}): {r.text[:300]}")
        submit_data = r.json()
    else:
        data = json.dumps(arguments).encode()
        req = urllib.request.Request(submit_url, data=data, headers=headers)
        resp = urllib.request.urlopen(req, timeout=30)
        submit_data = json.loads(resp.read())

    request_id = submit_data["request_id"]
    # Use URLs returned by fal.ai (they strip subpaths like /v2.6/)
    status_url = submit_data.get("status_url", f"https://queue.fal.run/{model_id}/requests/{request_id}/status")
    result_url = submit_data.get("response_url", f"https://queue.fal.run/{model_id}/requests/{request_id}")

    # Poll for completion
    start = time.time()
    while time.time() - start < timeout:
        time.sleep(3)
        if client:
            sr = client.get(status_url, headers=headers)
            sd = sr.json()
        else:
            req = urllib.request.Request(status_url, headers=headers)
            resp = urllib.request.urlopen(req, timeout=10)
            sd = json.loads(resp.read())

        status = sd.get("status", "")
        if status == "COMPLETED":
            # Fetch result
            if client:
                rr = client.get(result_url, headers=headers)
                if rr.status_code == 200:
                    return rr.json()
                raise RuntimeError(f"fal.ai Result Error ({rr.status_code}): {rr.text[:300]}")
            else:
                req = urllib.request.Request(result_url, headers=headers)
                resp = urllib.request.urlopen(req, timeout=30)
                return json.loads(resp.read())
        elif status == "FAILED":
            raise RuntimeError(f"fal.ai generation failed: {sd}")
        # IN_QUEUE or IN_PROGRESS — keep polling

    raise RuntimeError(f"fal.ai timeout after {timeout}s")

FAL_MODELS = {
    "Flux Dev (高品質・NSFW OK)": {
        "model_id": "fal-ai/flux/dev",
        "defaults": {"num_inference_steps": 28, "guidance_scale": 3.5, "safety_tolerance": 6},
        "cost": "~$0.025/枚",
        "nsfw": True,
    },
    "Flux Schnell (高速・安い)": {
        "model_id": "fal-ai/flux/schnell",
        "defaults": {"num_inference_steps": 4, "safety_tolerance": 6},
        "cost": "~$0.003/枚",
        "nsfw": True,
    },
    "Flux Realism (フォトリアル・NSFW OK)": {
        "model_id": "fal-ai/flux-realism",
        "defaults": {"num_inference_steps": 28, "guidance_scale": 3.5, "safety_tolerance": 6},
        "cost": "~$0.025/枚",
        "nsfw": True,
    },
    "Flux Pro 1.1 (最高品質・SFWのみ)": {
        "model_id": "fal-ai/flux-pro/v1.1",
        "defaults": {"safety_tolerance": 6},
        "cost": "~$0.05/枚",
        "nsfw": False,
    },
    "Recraft V3 (芸術的・高品質)": {
        "model_id": "fal-ai/recraft-v3",
        "defaults": {"num_inference_steps": 28, "guidance_scale": 3.5},
        "cost": "~$0.03/枚",
        "nsfw": False,
    },
    "AuraFlow v0.3 (芸術的・軽量)": {
        "model_id": "fal-ai/aura-flow",
        "defaults": {"num_inference_steps": 28, "guidance_scale": 3.5},
        "cost": "~$0.02/枚",
        "nsfw": False,
    },
    "Flux + LoRA (カスタムLoRA・NSFW OK)": {
        "model_id": "fal-ai/flux-lora",
        "defaults": {"num_inference_steps": 28, "guidance_scale": 3.5},
        "cost": "~$0.025/枚",
        "nsfw": True,
        "supports_lora": True,
    },
}

FAL_IMG2VID_MODELS = {
    # ── NSFW OK (Wan - オープンソース) ──
    "Wan 2.1 img2vid (高品質・NSFW OK)": {
        "model_id": "fal-ai/wan-i2v",
        "cost": "~$0.05/本",
        "nsfw": True,
        "max_duration": 5, "default_duration": 5,
        "supports_duration": False,
    },
    "Wan 2.6 img2vid (最新・NSFW OK)": {
        "model_id": "fal-ai/wan-i2v",
        "cost": "~$0.08/本",
        "nsfw": True,
        "max_duration": 15, "default_duration": 5,
        "supports_duration": True,
    },
    # ── SFW only (商用モデル - 高品質) ──
    "Kling 2.5 Turbo Pro img2vid (高品質・SFW)": {
        "model_id": "fal-ai/kling-video/v2.5-turbo/pro/image-to-video",
        "cost": "~$0.10/本",
        "nsfw": False,
        "max_duration": 10, "default_duration": 5,
        "supports_duration": True,
    },
    "Veo 3 img2vid (Google最高品質・SFW)": {
        "model_id": "fal-ai/veo3/image-to-video",
        "cost": "~$0.25/本",
        "nsfw": False,
        "max_duration": 8, "default_duration": 5,
        "supports_duration": True,
    },
    "Sora 2 img2vid (OpenAI・SFW)": {
        "model_id": "fal-ai/sora-2/image-to-video",
        "cost": "~$0.30/本",
        "nsfw": False,
        "max_duration": 20, "default_duration": 10,
        "supports_duration": True,
    },
    "LTX 2.3 img2vid (高速・安い)": {
        "model_id": "fal-ai/ltx-2.3/image-to-video/fast",
        "cost": "~$0.02/本",
        "nsfw": False,
        "max_duration": 5, "default_duration": 3,
        "supports_duration": True,
    },
}

FAL_VIDEO_MODELS = {
    # ── NSFW OK (Wan - オープンソース) ──
    "Wan 2.1 txt2vid (高品質・NSFW OK)": {
        "model_id": "fal-ai/wan-t2v",
        "cost": "~$0.05/本",
        "nsfw": True,
        "max_duration": 5, "default_duration": 5,
        "supports_duration": True, "supports_resolution": True,
    },
    "Wan 2.6 txt2vid (最新・NSFW OK)": {
        "model_id": "fal-ai/wan-t2v",
        "cost": "~$0.08/本",
        "nsfw": True,
        "max_duration": 10, "default_duration": 5,
        "supports_duration": True, "supports_resolution": True,
    },
    # ── SFW only (商用モデル - 最高品質) ──
    "Veo 3 (Google最高品質・芸術的)": {
        "model_id": "fal-ai/veo3/text-to-video",
        "cost": "~$0.25/本",
        "nsfw": False,
        "max_duration": 8, "default_duration": 5,
        "supports_duration": True, "supports_resolution": False,
    },
    "Veo 3.1 Fast (Google高速)": {
        "model_id": "fal-ai/veo3.1/fast",
        "cost": "~$0.15/本",
        "nsfw": False,
        "max_duration": 8, "default_duration": 5,
        "supports_duration": True, "supports_resolution": False,
    },
    "Sora 2 (OpenAI映画品質)": {
        "model_id": "fal-ai/sora-2/text-to-video",
        "cost": "~$0.20/本",
        "nsfw": False,
        "max_duration": 20, "default_duration": 10,
        "supports_duration": True, "supports_resolution": True,
    },
    "Kling 2.5 Turbo Pro (高品質・SFW)": {
        "model_id": "fal-ai/kling-video/v2.5-turbo/pro/text-to-video",
        "cost": "~$0.10/本",
        "nsfw": False,
        "max_duration": 10, "default_duration": 5,
        "supports_duration": True, "supports_resolution": True,
    },
    "LTX 2.3 (高速・安い)": {
        "model_id": "fal-ai/ltx-2.3/text-to-video/fast",
        "cost": "~$0.02/本",
        "nsfw": False,
        "max_duration": 5, "default_duration": 3,
        "supports_duration": True, "supports_resolution": True,
    },
}

# ── Style Transfer presets ──
STYLE_PRESETS = {
    "Ghibli (ジブリ風)": {
        "prompt_prefix": "studio ghibli style, hand-painted anime, soft watercolor, hayao miyazaki, warm colors, whimsical, detailed background, ",
        "negative": "photorealistic, 3d render, dark, harsh lighting",
        "strength": 0.75,
    },
    "Anime (アニメ)": {
        "prompt_prefix": "anime style, cel shading, vibrant colors, clean lines, detailed eyes, manga aesthetic, ",
        "negative": "photorealistic, 3d, blurry, western cartoon",
        "strength": 0.7,
    },
    "Oil Painting (油絵)": {
        "prompt_prefix": "oil painting style, classical art, thick brushstrokes, rich texture, canvas, dramatic lighting, old masters technique, ",
        "negative": "digital art, flat, anime, photograph",
        "strength": 0.8,
    },
    "Watercolor (水彩画)": {
        "prompt_prefix": "watercolor painting, soft washes, wet-on-wet technique, paper texture, transparent layers, artistic, gentle bleeding colors, ",
        "negative": "digital art, sharp edges, photorealistic, 3d",
        "strength": 0.75,
    },
    "Cyberpunk (サイバーパンク)": {
        "prompt_prefix": "cyberpunk style, neon lights, futuristic, rain-soaked streets, holographic, high-tech low-life, blade runner aesthetic, ",
        "negative": "nature, pastoral, vintage, old-fashioned",
        "strength": 0.7,
    },
    "Pixel Art (ピクセルアート)": {
        "prompt_prefix": "pixel art style, 16-bit retro game aesthetic, limited palette, crisp pixels, nostalgic, sprite art, ",
        "negative": "photorealistic, smooth, high resolution, 3d",
        "strength": 0.85,
    },
    "Comic Book (アメコミ)": {
        "prompt_prefix": "comic book style, bold outlines, halftone dots, dynamic composition, vivid colors, pop art, graphic novel, ",
        "negative": "photorealistic, soft, pastel, watercolor",
        "strength": 0.75,
    },
    "Ukiyo-e (浮世絵)": {
        "prompt_prefix": "ukiyo-e style, japanese woodblock print, flat perspective, bold outlines, traditional japanese art, edo period aesthetic, ",
        "negative": "photorealistic, 3d, western art, modern",
        "strength": 0.8,
    },
    "Renaissance (ルネサンス)": {
        "prompt_prefix": "renaissance painting, chiaroscuro, sfumato technique, classical composition, divine light, leonardo da vinci style, ",
        "negative": "modern, digital, anime, flat",
        "strength": 0.8,
    },
    "Impressionist (印象派)": {
        "prompt_prefix": "impressionist painting, visible brushstrokes, light and color focus, claude monet style, en plein air, atmospheric, ",
        "negative": "sharp, digital, photorealistic, dark",
        "strength": 0.75,
    },
    "Art Nouveau (アールヌーヴォー)": {
        "prompt_prefix": "art nouveau style, ornate flowing lines, organic forms, decorative borders, alphonse mucha inspired, elegant, ",
        "negative": "minimal, brutalist, simple, photorealistic",
        "strength": 0.8,
    },
    "Surrealism (シュルレアリスム)": {
        "prompt_prefix": "surrealist art, dreamlike, impossible architecture, melting forms, salvador dali inspired, subconscious imagery, ",
        "negative": "realistic, ordinary, mundane, photographic",
        "strength": 0.75,
    },
}

# ── ControlNet types ──
CONTROLNET_TYPES = {
    "Canny (輪郭線)": "canny",
    "Depth (深度マップ)": "depth",
    "OpenPose (ポーズ)": "openpose",
    "Scribble (スケッチ)": "scribble",
}

# ── Artistic presets for high-quality generation ──
ART_PRESETS = {
    "Cinematic Portrait (映画的ポートレート)": {
        "prompt_suffix": ", cinematic lighting, shallow depth of field, 85mm lens, film grain, professional color grading, dramatic atmosphere, award-winning photography",
        "negative": "ugly, deformed, noisy, blurry, low quality, cartoon, anime, illustration",
        "model": "Flux Realism (フォトリアル・NSFW OK)",
        "steps": 30, "cfg": 4.0, "width": 832, "height": 1216,
    },
    "Fine Art Photography (芸術写真)": {
        "prompt_suffix": ", hasselblad medium format, fine art photography, museum quality, perfect composition, golden ratio, masterful use of light and shadow",
        "negative": "amateur, snapshot, low quality, blurry, oversaturated",
        "model": "Flux Dev (高品質・NSFW OK)",
        "steps": 35, "cfg": 3.5, "width": 1024, "height": 1024,
    },
    "Fantasy Illustration (ファンタジーイラスト)": {
        "prompt_suffix": ", fantasy art, detailed illustration, epic composition, magical atmosphere, concept art, trending on artstation, volumetric lighting, highly detailed",
        "negative": "photorealistic, blurry, low quality, amateur",
        "model": "Flux Dev (高品質・NSFW OK)",
        "steps": 30, "cfg": 4.0, "width": 1024, "height": 1024,
    },
    "Anime Masterpiece (アニメ傑作)": {
        "prompt_suffix": ", masterpiece, best quality, ultra detailed, beautiful anime art, vibrant colors, studio quality, perfect anatomy, expressive eyes, professional illustration",
        "negative": "worst quality, low quality, deformed, ugly, blurry, bad anatomy, bad hands",
        "model": "Flux Dev (高品質・NSFW OK)",
        "steps": 28, "cfg": 3.5, "width": 832, "height": 1216,
    },
    "Dark Gothic (ダークゴシック)": {
        "prompt_suffix": ", dark gothic art, moody atmosphere, dramatic shadows, ornate details, cathedral architecture, ravens, candlelight, dark fantasy, eerie beautiful",
        "negative": "bright, cheerful, cartoon, low quality, blurry",
        "model": "Flux Dev (高品質・NSFW OK)",
        "steps": 30, "cfg": 4.0, "width": 1024, "height": 1024,
    },
    "Japanese Aesthetic (和の美学)": {
        "prompt_suffix": ", japanese aesthetics, wabi-sabi, mono no aware, cherry blossom, zen garden, minimalist beauty, traditional japanese, seasonal awareness, poetic atmosphere",
        "negative": "western, modern, cluttered, ugly, low quality",
        "model": "Flux Dev (高品質・NSFW OK)",
        "steps": 30, "cfg": 3.5, "width": 1024, "height": 1024,
    },
    "Hyperrealistic NSFW (超リアルNSFW)": {
        "prompt_suffix": ", hyperrealistic, photorealistic, 8k uhd, dslr quality, natural skin texture, perfect anatomy, studio lighting, professional photography, detailed skin pores",
        "negative": "anime, cartoon, painting, deformed, ugly, blurry, low quality, bad anatomy, bad hands, extra fingers",
        "model": "Flux Realism (フォトリアル・NSFW OK)",
        "steps": 35, "cfg": 4.0, "width": 832, "height": 1216,
    },
    "Erotic Art (エロティックアート)": {
        "prompt_suffix": ", erotic fine art photography, tasteful sensuality, dramatic lighting, artistic nude, museum quality, professional photography, perfect body proportions, skin detail",
        "negative": "deformed, ugly, blurry, low quality, amateur, bad anatomy, extra limbs",
        "model": "Flux Realism (フォトリアル・NSFW OK)",
        "steps": 35, "cfg": 4.0, "width": 832, "height": 1216,
    },
}

# ── NSFW-optimized model/LoRA recommendations ──
NSFW_PRESETS = {
    # ── リアル系 ──
    "AV風 リアル (Photorealistic Adult)": {
        "prompt_base": "photorealistic, raw photo, 8k uhd, dslr, natural skin texture, detailed skin pores, natural body proportions, studio lighting, professional adult photography, perfect face, beautiful woman, detailed eyes",
        "negative": "anime, cartoon, painting, deformed, ugly, blurry, bad anatomy, extra fingers, bad hands, mutated, disfigured, watermark, text, logo, low quality, worst quality, jpeg artifacts",
        "recommended_models": ["Flux Realism (フォトリアル・NSFW OK)", "Flux Dev (高品質・NSFW OK)"],
        "recommended_loras": ["povSkinTexture_v2", "eroticVision_v4", "Film lora"],
        "settings": {"steps": 40, "cfg": 4.0, "width": 832, "height": 1216},
        "fallback_provider": "dezgo",
    },
    "POV リアル (First Person View)": {
        "prompt_base": "photorealistic, first person pov, raw photo, 8k uhd, natural skin texture, detailed skin pores, intimate close-up, shallow depth of field, natural lighting, professional photography",
        "negative": "anime, cartoon, deformed, ugly, blurry, bad anatomy, extra fingers, watermark, text, low quality, worst quality",
        "recommended_models": ["Flux Realism (フォトリアル・NSFW OK)"],
        "recommended_loras": ["povSkinTexture_v2", "DoggystylePOV", "Film lora"],
        "settings": {"steps": 40, "cfg": 4.0, "width": 1216, "height": 832},
        "fallback_provider": "dezgo",
    },
    "ランジェリー・下着モデル (Lingerie)": {
        "prompt_base": "photorealistic, professional lingerie photography, fashion magazine quality, studio lighting, beautiful woman, perfect body, natural skin, lace details, silk texture, soft focus background, 8k uhd",
        "negative": "deformed, ugly, blurry, bad anatomy, extra fingers, watermark, text, low quality, cartoon, anime",
        "recommended_models": ["Flux Realism (フォトリアル・NSFW OK)"],
        "recommended_loras": ["Film lora", "povSkinTexture_v2"],
        "settings": {"steps": 35, "cfg": 3.5, "width": 832, "height": 1216},
        "fallback_provider": "dezgo",
    },
    "グラビア・水着 (Gravure Swimsuit)": {
        "prompt_base": "gravure photography, japanese idol style, magazine cover quality, professional lighting, sharp focus, beautiful woman, natural beauty, perfect skin, beach setting, golden hour, 8k uhd",
        "negative": "deformed, ugly, blurry, low quality, bad anatomy, extra fingers, watermark, text, worst quality",
        "recommended_models": ["Flux Realism (フォトリアル・NSFW OK)"],
        "recommended_loras": ["Film lora", "povSkinTexture_v2"],
        "settings": {"steps": 35, "cfg": 3.5, "width": 832, "height": 1216},
        "fallback_provider": "dezgo",
    },
    # ── アニメ系 ──
    "アニメ NSFW (Anime Hentai)": {
        "prompt_base": "masterpiece, best quality, ultra detailed, beautiful anime art, perfect anatomy, smooth skin, expressive eyes, vibrant colors, professional illustration, detailed shading",
        "negative": "worst quality, low quality, deformed, ugly, blurry, bad anatomy, bad hands, extra fingers, 3d, photorealistic, sketch, rough",
        "recommended_models": ["Flux Dev (高品質・NSFW OK)"],
        "recommended_loras": ["cardosAnime_v20", "meinamix", "FappXL"],
        "settings": {"steps": 35, "cfg": 3.5, "width": 832, "height": 1216},
        "fallback_provider": "dezgo",
    },
    "アニメ 美少女 (Anime Bishoujo)": {
        "prompt_base": "masterpiece, best quality, ultra detailed, beautiful anime girl, perfect face, detailed eyes, glossy lips, perfect anatomy, pastel colors, soft lighting, kawaii, moe aesthetic",
        "negative": "worst quality, low quality, deformed, ugly, bad anatomy, extra fingers, blurry, 3d, photorealistic, sketch",
        "recommended_models": ["Flux Dev (高品質・NSFW OK)"],
        "recommended_loras": ["cardosAnime_v20", "meinamix"],
        "settings": {"steps": 35, "cfg": 3.5, "width": 832, "height": 1216},
        "fallback_provider": "dezgo",
    },
    # ── エロティックアート ──
    "エロティックアート (Fine Art Nude)": {
        "prompt_base": "erotic fine art photography, museum quality, dramatic chiaroscuro lighting, artistic nude, perfect composition, golden ratio, renaissance inspiration, skin detail, tasteful sensuality, professional photography",
        "negative": "vulgar, deformed, ugly, blurry, low quality, amateur, bad anatomy, extra limbs, cartoon, anime",
        "recommended_models": ["Flux Realism (フォトリアル・NSFW OK)", "Flux Dev (高品質・NSFW OK)"],
        "recommended_loras": ["Film lora", "povSkinTexture_v2"],
        "settings": {"steps": 40, "cfg": 4.0, "width": 832, "height": 1216},
        "fallback_provider": "dezgo",
    },
    "ボンデージ・フェティッシュ (Bondage/Fetish)": {
        "prompt_base": "photorealistic, professional fetish photography, dramatic lighting, leather, latex, rope, artistic composition, high contrast, studio photography, detailed textures, 8k uhd",
        "negative": "deformed, ugly, blurry, bad anatomy, extra fingers, watermark, text, low quality, cartoon, anime, worst quality",
        "recommended_models": ["Flux Realism (フォトリアル・NSFW OK)"],
        "recommended_loras": ["povSkinTexture_v2", "eroticVision_v4"],
        "settings": {"steps": 40, "cfg": 4.0, "width": 832, "height": 1216},
        "fallback_provider": "dezgo",
    },
}

# ── NSFW Video presets ──
NSFW_VIDEO_PRESETS = {
    "AV風 リアル動画": {
        "prompt_base": "photorealistic, natural movement, smooth motion, professional adult video, natural skin, intimate, soft lighting, shallow depth of field, cinematic",
        "negative": "blurry, distorted, low quality, jerky motion, anime, cartoon",
        "model": "Wan 2.6 txt2vid (最新・NSFW OK)",
        "img2vid_model": "Wan 2.6 img2vid (最新・NSFW OK)",
        "duration": 5,
    },
    "グラビア動画 (Gravure Video)": {
        "prompt_base": "gravure video, japanese idol, professional lighting, smooth camera movement, beach, golden hour, slow motion, cinematic, magazine quality",
        "negative": "blurry, distorted, low quality, jerky",
        "model": "Wan 2.6 txt2vid (最新・NSFW OK)",
        "img2vid_model": "Wan 2.6 img2vid (最新・NSFW OK)",
        "duration": 5,
    },
    "セクシーダンス (Sexy Dance)": {
        "prompt_base": "beautiful woman dancing sensually, smooth flowing movement, professional lighting, music video quality, slow motion, cinematic, detailed body movement",
        "negative": "blurry, distorted, static, jerky motion, low quality",
        "model": "Wan 2.6 txt2vid (最新・NSFW OK)",
        "img2vid_model": "Wan 2.6 img2vid (最新・NSFW OK)",
        "duration": 5,
    },
    "エロティックアニメ動画": {
        "prompt_base": "anime style, smooth animation, beautiful anime girl, detailed animation, vibrant colors, professional quality, fluid movement",
        "negative": "low quality, jerky, distorted, ugly, blurry, still image",
        "model": "Wan 2.1 txt2vid (高品質・NSFW OK)",
        "img2vid_model": "Wan 2.1 img2vid (高品質・NSFW OK)",
        "duration": 5,
    },
}

# ── High Quality generation defaults (for maximum quality output) ──
HQ_DEFAULTS = {
    "image": {
        "flux_realism": {"steps": 40, "guidance": 4.0, "width": 1216, "height": 1216},
        "flux_dev": {"steps": 35, "guidance": 3.5, "width": 1216, "height": 1216},
        "flux_schnell": {"steps": 4, "guidance": 0, "width": 1024, "height": 1024},
    },
    "video": {
        "wan_26": {"duration": 10, "width": 1280, "height": 720},
        "wan_21": {"duration": 5, "width": 832, "height": 480},
        "sora_2": {"duration": 15, "width": 1920, "height": 1080},
        "veo_3": {"duration": 8},
        "kling_25": {"duration": 10, "width": 1280, "height": 720},
    },
    "upscale_after_generate": True,
}


class FalClient:
    def __init__(self, api_key=""):
        self.api_key = api_key
        if api_key:
            os.environ["FAL_KEY"] = api_key

    def set_key(self, api_key):
        self.api_key = api_key
        os.environ["FAL_KEY"] = api_key

    def _ensure_key(self):
        if not self.api_key:
            raise ValueError("fal.ai API Key が設定されていません")
        os.environ["FAL_KEY"] = self.api_key

    def _extract_image_urls(self, result):
        """Extract image URLs from fal.ai result."""
        urls = []
        if isinstance(result, dict):
            for img in result.get("images", []):
                url = img.get("url", "")
                if url:
                    urls.append(url)
            if not urls:
                img = result.get("image", {})
                if isinstance(img, dict) and img.get("url"):
                    urls.append(img["url"])
                elif isinstance(img, str) and img.startswith("http"):
                    urls.append(img)
            if not urls:
                for key in ["output", "result", "url"]:
                    val = result.get(key)
                    if isinstance(val, str) and val.startswith("http"):
                        urls.append(val)
                    elif isinstance(val, dict) and val.get("url"):
                        urls.append(val["url"])
        return urls

    def _extract_video_url(self, result):
        """Extract video URL from fal.ai result."""
        if isinstance(result, dict):
            vid = result.get("video", {})
            if isinstance(vid, dict):
                return vid.get("url", "")
            elif isinstance(vid, str):
                return vid
            return result.get("url", "")
        return ""

    def generate_image(self, model_key, prompt, negative_prompt="",
                       width=1024, height=1024, steps=None, guidance=None,
                       seed=-1, num_images=1, safety_checker=False,
                       lora_urls=None):
        """Generate image via fal.ai. Returns list of image URLs.

        Args:
            lora_urls: list of (url, scale) tuples for LoRA weights.
                       e.g. [("https://storage.example.com/lora.safetensors", 0.8)]
        """
        self._ensure_key()

        model_info = FAL_MODELS.get(model_key)
        if not model_info:
            raise ValueError(f"不明なモデル: {model_key}")

        model_id = model_info["model_id"]

        # If LoRAs provided but model doesn't support them, auto-switch to flux-lora
        if lora_urls and not model_info.get("supports_lora"):
            model_id = "fal-ai/flux-lora"

        args = {
            "prompt": prompt,
            "image_size": {"width": width, "height": height},
            "num_images": num_images,
            "enable_safety_checker": False,
        }

        defaults = model_info.get("defaults", {})
        if "safety_tolerance" in defaults:
            args["safety_tolerance"] = defaults["safety_tolerance"]

        if negative_prompt:
            args["negative_prompt"] = negative_prompt

        if "num_inference_steps" in defaults:
            args["num_inference_steps"] = defaults["num_inference_steps"]
        if "guidance_scale" in defaults:
            args["guidance_scale"] = defaults["guidance_scale"]

        if steps is not None:
            args["num_inference_steps"] = steps
        if guidance is not None:
            args["guidance_scale"] = guidance
        if seed >= 0:
            args["seed"] = seed

        # Add LoRA weights
        if lora_urls:
            args["loras"] = [{"path": url, "scale": scale} for url, scale in lora_urls]

        result = _fal_run(self.api_key, model_id, args)
        return self._extract_image_urls(result)

    def generate_video(self, model_key, prompt, negative_prompt="",
                       duration=None, width=None, height=None, seed=-1):
        """Generate video via fal.ai. Returns video URL."""
        self._ensure_key()

        model_info = FAL_VIDEO_MODELS.get(model_key)
        if not model_info:
            raise ValueError(f"不明な動画モデル: {model_key}")

        model_id = model_info["model_id"]
        args = {
            "prompt": prompt,
            "enable_safety_checker": False,
        }

        if negative_prompt:
            args["negative_prompt"] = negative_prompt
        if seed >= 0:
            args["seed"] = seed

        # Duration (seconds)
        if model_info.get("supports_duration"):
            dur = duration or model_info.get("default_duration", 5)
            max_dur = model_info.get("max_duration", 10)
            dur = min(dur, max_dur)
            args["duration"] = dur

        # Resolution
        if model_info.get("supports_resolution") and width and height:
            args["image_size"] = {"width": int(width), "height": int(height)}

        result = _fal_run(self.api_key, model_id, args)
        return self._extract_video_url(result)

    def img2vid(self, model_key, image_url, prompt="", negative_prompt="",
                duration=None, seed=-1):
        """Generate video from image via fal.ai. Returns video URL."""
        self._ensure_key()

        model_info = FAL_IMG2VID_MODELS.get(model_key)
        if not model_info:
            raise ValueError(f"不明なimg2vidモデル: {model_key}")

        model_id = model_info["model_id"]
        args = {
            "image_url": image_url,
            "enable_safety_checker": False,
        }
        if prompt:
            args["prompt"] = prompt
        if negative_prompt:
            args["negative_prompt"] = negative_prompt
        if seed >= 0:
            args["seed"] = seed

        # Duration (seconds)
        if model_info.get("supports_duration"):
            dur = duration or model_info.get("default_duration", 5)
            max_dur = model_info.get("max_duration", 10)
            dur = min(dur, max_dur)
            args["duration"] = dur

        result = _fal_run(self.api_key, model_id, args)
        return self._extract_video_url(result)

    def style_transfer(self, image_url, style_key, custom_prompt="", strength=0.75):
        """Apply style transfer to an image. Returns list of image URLs."""
        self._ensure_key()

        style = STYLE_PRESETS.get(style_key)
        if not style:
            raise ValueError(f"不明なスタイル: {style_key}")

        prompt = style["prompt_prefix"]
        if custom_prompt:
            prompt += custom_prompt
        else:
            prompt += "high quality, detailed, artistic"

        args = {
            "prompt": prompt,
            "image_url": image_url,
            "strength": strength or style.get("strength", 0.75),
            "image_size": "square_hd",
            "num_inference_steps": 28,
            "guidance_scale": 3.5,
            "num_images": 1,
            "enable_safety_checker": False,
            "safety_tolerance": 6,
        }
        if style.get("negative"):
            args["negative_prompt"] = style["negative"]

        result = _fal_run(self.api_key, "fal-ai/flux/dev/image-to-image", args)
        return self._extract_image_urls(result)

    def inpaint(self, image_url, mask_url, prompt, negative_prompt="",
                width=1024, height=1024, steps=28, guidance=3.5, seed=-1):
        """Inpaint image region specified by mask. Returns list of image URLs."""
        self._ensure_key()

        args = {
            "prompt": prompt,
            "image_url": image_url,
            "mask_url": mask_url,
            "image_size": {"width": width, "height": height},
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "num_images": 1,
            "enable_safety_checker": False,
            "safety_tolerance": 6,
        }
        if negative_prompt:
            args["negative_prompt"] = negative_prompt
        if seed >= 0:
            args["seed"] = seed

        result = _fal_run(self.api_key, "fal-ai/flux/dev/inpainting", args)
        return self._extract_image_urls(result)

    def remove_background(self, image_url):
        """Remove background from image. Returns result image URL."""
        self._ensure_key()

        result = _fal_run(self.api_key, "fal-ai/birefnet", {
            "image_url": image_url,
        })
        return self._extract_image_urls(result)

    def upscale(self, image_url, scale=2):
        """Upscale image using AI. Returns result image URL."""
        self._ensure_key()

        if scale >= 4:
            # Use clarity upscaler for 4x
            result = _fal_run(self.api_key, "fal-ai/clarity-upscaler", {
                "image_url": image_url,
                "scale_factor": scale,
                "enable_safety_checker": False,
            })
        else:
            # Use Aura SR for 2x (faster, good quality)
            result = _fal_run(self.api_key, "fal-ai/aura-sr", {
                "image_url": image_url,
                "upscaling_factor": scale,
            })
        return self._extract_image_urls(result)

    def controlnet(self, control_image_url, prompt, control_type="canny",
                   control_strength=0.7, negative_prompt="",
                   width=1024, height=1024, steps=28, guidance=3.5, seed=-1):
        """Generate image guided by ControlNet. Returns list of image URLs."""
        self._ensure_key()

        args = {
            "prompt": prompt,
            "control_image_url": control_image_url,
            "controlnet_conditioning_scale": control_strength,
            "image_size": {"width": width, "height": height},
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "num_images": 1,
            "enable_safety_checker": False,
            "safety_tolerance": 6,
        }
        if negative_prompt:
            args["negative_prompt"] = negative_prompt
        if seed >= 0:
            args["seed"] = seed

        # Map control type to fal.ai model
        control_models = {
            "canny": "fal-ai/flux-general/v2",
            "depth": "fal-ai/flux-general/v2",
            "openpose": "fal-ai/flux-general/v2",
            "scribble": "fal-ai/flux-general/v2",
        }
        model_id = control_models.get(control_type, "fal-ai/flux-general/v2")

        result = _fal_run(self.api_key, model_id, args)
        return self._extract_image_urls(result)

    def vid2vid(self, video_url, prompt, model_key="Wan 2.1 vid2vid (NSFW OK)", strength=0.6):
        """Video-to-video style transfer via cloud. Returns video URL."""
        self._ensure_key()

        model_info = FAL_VID2VID_MODELS.get(model_key)
        if not model_info:
            raise ValueError(f"不明なvid2vidモデル: {model_key}")

        args = {
            "video_url": video_url,
            "prompt": prompt,
            "strength": strength,
            "enable_safety_checker": False,
        }

        result = _fal_run(self.api_key, model_info["model_id"], args)
        return self._extract_video_url(result)

    def face_swap(self, base_image_url, swap_image_url):
        """Face swap via fal.ai. Returns result image URL."""
        self._ensure_key()

        result = _fal_run(self.api_key, "fal-ai/face-swap", {
            "base_image_url": base_image_url,
            "swap_image_url": swap_image_url,
        })

        urls = self._extract_image_urls(result)
        return urls[0] if urls else ""

    def check_api_key(self):
        try:
            _fal_run(self.api_key, "fal-ai/flux/schnell",
                     {"prompt": "test", "image_size": {"width": 256, "height": 256}, "num_inference_steps": 1})
            return True
        except Exception:
            return False


# ── vid2vid models ──
FAL_VID2VID_MODELS = {
    "Wan 2.1 vid2vid (NSFW OK)": {
        "model_id": "fal-ai/wan-v2v",
        "cost": "~$0.10/本",
        "nsfw": True,
    },
    "Wan 2.6 vid2vid (最新・NSFW OK)": {
        "model_id": "fal-ai/wan-v2v",
        "cost": "~$0.15/本",
        "nsfw": True,
    },
    "Kling 2.5 vid2vid (高品質・SFW)": {
        "model_id": "fal-ai/kling-video/v2.5-turbo/pro/video-to-video",
        "cost": "~$0.15/本",
        "nsfw": False,
    },
}


def download_fal_image(url):
    """Download image from fal.ai URL and return PIL Image."""
    from PIL import Image
    req = urllib.request.Request(url, headers={"User-Agent": "AI-diffusion/1.0"})
    resp = urllib.request.urlopen(req, timeout=30)
    return Image.open(io.BytesIO(resp.read()))


def download_fal_video(url, output_path):
    """Download video from fal.ai URL and save to file."""
    req = urllib.request.Request(url, headers={"User-Agent": "AI-diffusion/1.0"})
    resp = urllib.request.urlopen(req, timeout=120)
    with open(output_path, "wb") as f:
        f.write(resp.read())
    return output_path
