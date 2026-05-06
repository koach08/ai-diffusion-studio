"""AI-diffusion Studio - Custom UI with ComfyUI backend."""
import base64
import datetime
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
import urllib.error

import gradio as gr
from PIL import Image

# ── Error logging to file ──
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "error.log")
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("ai-diffusion")

from config import load_config, save_config, get_available_models, get_available_loras, get_available_vaes, get_available_motion_models, get_available_upscale_models, get_available_unet_models, get_available_clip_models, get_icloud_only_models, SAMPLERS, SCHEDULERS
from comfyui_api import (
    ComfyUIClient, build_txt2img_workflow, build_img2img_workflow, build_refine_workflow,
    build_inpaint_workflow, build_animatediff_workflow, build_img2vid_workflow,
    build_vid2vid_workflow, build_flux_workflow, build_controlnet_workflow,
    build_ipadapter_workflow,
    build_wan22_t2v_workflow, build_wan22_i2v_workflow, WAN22_DEFAULTS,
)
from runpod_manager import RunPodManager, format_pod_status, format_pod_cost
from vast_ai_manager import (VastAIManager, RECOMMENDED_MODELS as VAST_MODELS,
                              STARTER_PACK, FULL_PACK, COMFYUI_IMAGES,
                              format_instance_status, format_cost_summary)
from replicate_api import ReplicateClient, MODELS as REPLICATE_MODELS, VIDEO_MODELS as REPLICATE_VIDEO_MODELS, download_url_to_pil
from civitai_api import CivitAIClient, format_search_results, CIVITAI_GENERATION_MODELS
from fal_api import (FalClient, FAL_MODELS, FAL_VIDEO_MODELS, FAL_IMG2VID_MODELS, FAL_VID2VID_MODELS,
                      STYLE_PRESETS, CONTROLNET_TYPES, ART_PRESETS, NSFW_PRESETS,
                      NSFW_VIDEO_PRESETS, HQ_DEFAULTS,
                      download_fal_image, download_fal_video)
from together_api import TogetherClient, TOGETHER_MODELS, decode_together_image
from dezgo_api import DezgoClient, DEZGO_IMAGE_MODELS, DEZGO_VIDEO_MODELS, decode_dezgo_image
from novita_api import NovitaClient, NOVITA_MODELS, download_novita_image
from guide import GUIDE_SECTIONS, PROMPT_TEMPLATES
from ai_assistant import chat_with_ai, QUICK_QUESTIONS, PROVIDERS
from adult_studio import (
    CHAR_ETHNICITY, CHAR_AGE, CHAR_BODY_TYPE, CHAR_BREAST, CHAR_BUTT,
    CHAR_HAIR_COLOR, CHAR_HAIR_STYLE, CHAR_SKIN, CHAR_EXPRESSION,
    CHAR_CLOTHING, CHAR_POSE, SEX_POSITIONS, CHAR_CAMERA, CHAR_SETTING,
    CHAR_STYLE, CHAR_PEOPLE_COUNT, SCENE_CATEGORIES, UNDRESS_MODES,
    ADULT_VIDEO_SCENES,
    ADULT_LORA_CATEGORIES, MODEL_OPTIMAL_SETTINGS,
    QUALITY_PRESETS, get_quality_preset, apply_quality_to_params,
    compose_character_prompt, compose_scene_prompt, compose_video_prompt,
    get_undress_params, filter_loras_by_category, get_model_settings,
)

config = load_config()

# ── Cinema Camera Presets ──
CINEMA_PRESETS = {
    "(なし)": "",
    "Cinematic Blockbuster": ", shot on ARRI Alexa 35, Cooke S7/i 50mm T2.0, shallow depth of field, anamorphic lens flare, 2.39:1 cinematic framing",
    "Indie / A24": ", shot on ARRI AMIRA, Zeiss Super Speed 35mm T1.3, natural lighting, 16mm film grain, muted desaturated palette, handheld camera",
    "Vintage 70s": ", shot on Panavision Panaflex, anamorphic C-Series lenses, heavy halation, warm color cast, film grain, soft focus edges",
    "Music Video / Fashion": ", shot on RED V-Raptor 8K, Sigma Cine 85mm T1.5, extreme shallow DOF, high contrast, teal and orange grade, slow motion",
    "Documentary": ", shot on Sony FX6, Sony 24-70mm f/2.8 GM, available light, slight camera shake, realistic skin tones, broadcast look",
    "Horror / Thriller": ", shot on RED Monstro 8K, Leica Summilux-C 29mm T1.4, dutch angle, underexposed, desaturated, green-tinted shadows, wide-angle distortion",
    "Wes Anderson": ", shot on ARRI Alexa Mini, Zeiss Master Prime 40mm T1.3, perfectly symmetrical composition, pastel color palette, flat lighting, centered framing",
    "IMAX Epic": ", shot on ARRI Alexa 65, Hasselblad Prime 65 50mm, IMAX aspect ratio, extreme resolution, vast depth of field f/8, sweeping crane shot",
    "Neon Noir / Cyberpunk": ", shot on Blackmagic URSA Mini Pro 12K, Sigma Art 35mm f/1.4, neon reflections on wet pavement, high contrast, deep blacks, cyan and magenta grade",
    "Dreamy / Ethereal": ", shot on Canon C500 Mark II, Canon CN-E 85mm T1.3, Pro-Mist diffusion filter, golden hour backlighting, lens flare, warm highlights, soft skin",
    "Tarantino": ", shot on 35mm film, Panavision Ultra Speed 40mm T1.1, trunk shot POV, low angle, saturated reds, 1970s exploitation film aesthetic, visible film scratches",
    "Spielberg Classic": ", shot on Panavision Millennium XL2, anamorphic G-Series, lens flare, magic hour lighting, 2.39:1 widescreen, dolly zoom",
    "Drone / Aerial": ", shot on DJI Inspire 3, Zenmuse X9 24mm, aerial establishing shot, golden hour, sweeping orbital movement, vast landscape, tilt-shift",
    "Found Footage / CCTV": ", shot on low-resolution CCTV camera, wide-angle fisheye lens, infrared night vision, timestamp overlay, VHS tracking lines, high ISO noise",
    "Anime Cinematic": ", anime feature film quality, Makoto Shinkai style lighting, detailed backgrounds, volumetric light rays, 2.39:1 cinematic letterbox, vibrant color palette",
}

# ── Color Grading Presets (Pillow-based) ──
COLOR_GRADE_PRESETS = {
    "(なし)": {},
    "Teal & Orange": {"contrast": 1.15, "saturation": 1.3, "brightness": 1.0},
    "Vintage Warm": {"contrast": 1.1, "brightness": 1.05, "saturation": 0.9, "sepia": 0.25},
    "Film Noir": {"contrast": 1.4, "brightness": 0.9, "saturation": 0.0},
    "Cold Blue": {"contrast": 1.05, "brightness": 1.05, "saturation": 0.8, "hue_shift": 15},
    "Golden Hour": {"contrast": 1.05, "brightness": 1.1, "saturation": 1.3, "sepia": 0.15},
    "Bleach Bypass": {"contrast": 1.5, "brightness": 0.95, "saturation": 0.4},
    "Neon Glow": {"contrast": 1.2, "brightness": 1.1, "saturation": 1.8},
    "Moody Dark": {"contrast": 1.3, "brightness": 0.85, "saturation": 0.7, "sepia": 0.1},
    "Pastel Dream": {"contrast": 0.9, "brightness": 1.15, "saturation": 0.6},
    "Matte Film": {"contrast": 0.95, "brightness": 1.08, "saturation": 0.85, "sepia": 0.08},
    "High Contrast B&W": {"contrast": 1.8, "brightness": 0.95, "saturation": 0.0},
    "Retro Fade": {"contrast": 0.95, "brightness": 1.1, "saturation": 0.7, "sepia": 0.35},
    "Cross Process": {"contrast": 1.15, "brightness": 1.05, "saturation": 1.4, "hue_shift": 25},
}


def apply_color_grade(pil_image, grade_name):
    """Apply color grading to a PIL Image using ImageEnhance."""
    from PIL import ImageEnhance, ImageFilter
    import numpy as np

    if grade_name == "(なし)" or grade_name not in COLOR_GRADE_PRESETS:
        return pil_image

    params = COLOR_GRADE_PRESETS[grade_name]
    img = pil_image.copy()

    # Saturation (0.0 = grayscale)
    if "saturation" in params:
        img = ImageEnhance.Color(img).enhance(params["saturation"])

    # Contrast
    if "contrast" in params:
        img = ImageEnhance.Contrast(img).enhance(params["contrast"])

    # Brightness
    if "brightness" in params:
        img = ImageEnhance.Brightness(img).enhance(params["brightness"])

    # Sepia tone
    if params.get("sepia", 0) > 0:
        arr = np.array(img, dtype=np.float32)
        sepia_matrix = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131],
        ])
        sepia_arr = arr[:, :, :3] @ sepia_matrix.T
        sepia_arr = np.clip(sepia_arr, 0, 255)
        strength = params["sepia"]
        blended = (1 - strength) * arr[:, :, :3] + strength * sepia_arr
        arr[:, :, :3] = np.clip(blended, 0, 255)
        img = Image.fromarray(arr.astype(np.uint8))

    return img


# ── Session save/restore ──
SESSION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "session.json")

def save_session(data):
    """Save current work state to session file."""
    try:
        with open(SESSION_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"Session save error: {e}")
        return False

def load_session():
    """Load saved session state."""
    try:
        if os.path.exists(SESSION_FILE):
            with open(SESSION_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Session load error: {e}")
    return {}

def shutdown_app():
    """Shutdown ComfyUI and Gradio safely."""
    import signal
    try:
        subprocess.run(["bash", "-c", "lsof -ti:8188 | xargs kill 2>/dev/null"], timeout=5)
    except Exception:
        pass
    os.kill(os.getpid(), signal.SIGTERM)

client = ComfyUIClient(config["comfyui_url"])
runpod = RunPodManager(config.get("runpod_api_key", ""))
vastai = VastAIManager(config.get("vast_api_key", ""))
civitai = CivitAIClient(config.get("civitai_api_key", ""))
replicate = ReplicateClient(config.get("replicate_api_key", ""))
fal = FalClient(config.get("fal_api_key", ""))
together = TogetherClient(config.get("together_api_key", ""))
dezgo = DezgoClient(config.get("dezgo_api_key", ""))
novita = NovitaClient(config.get("novita_api_key", ""))

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _get_models_for_backend():
    """Get model list: from ComfyUI API for vast/local+running, else local scan."""
    backend = config.get("backend", "local")
    if backend in ("vast", "local", "runpod"):
        try:
            remote = client.get_models()
            if remote:
                return remote
        except Exception:
            pass
    return get_available_models(config["models_dir"])


def _build_model_choices(models=None):
    """Build full model choices list based on current backend (with iCloud support)."""
    if models is None:
        models = _get_models_for_backend()
    backend = config.get("backend", "local")
    civitai_names = list(CIVITAI_GENERATION_MODELS.keys())

    if backend == "civitai":
        icloud_models = get_icloud_only_models()
        icloud_set = set(icloud_models)
        models = [m for m in models if m not in icloud_set]
        icloud_prefixed = [f"[iCloud] {m}" for m in icloud_models]
        return models + civitai_names + icloud_prefixed
    elif backend == "fal":
        fal_names = list(FAL_MODELS.keys())
        return models + fal_names + civitai_names
    else:
        return models + civitai_names


def refresh_models():
    all_models = _build_model_choices()
    loras = ["None"] + get_available_loras(config["models_dir"])
    vaes = ["None"] + get_available_vaes(config["models_dir"])
    return (
        gr.update(choices=all_models, value=all_models[0] if all_models else None),
        gr.update(choices=loras, value="None"),
        gr.update(choices=vaes, value="None"),
    )


def refresh_all_model_dropdowns():
    """Refresh model/lora/vae dropdowns across ALL tabs after a download."""
    all_models = _build_model_choices()
    loras = ["None"] + get_available_loras(config["models_dir"])
    vaes = ["None"] + get_available_vaes(config["models_dir"])
    motion = get_available_motion_models()
    m = gr.update(choices=all_models)
    l = gr.update(choices=loras)
    v = gr.update(choices=vaes)
    mm = gr.update(choices=motion)
    return (m, l, v, m, l, v, m, l, v, mm, m, v, mm, m, v, mm, get_model_summary())


def check_server_status():
    backend = config.get("backend", "local")
    if backend == "replicate":
        if replicate.api_key:
            return "🟢 Replicate API: Ready (通常画像向け・在庫切れなし)"
        return "🔴 Replicate API Key が未設定 — Settingsタブで設定してください"
    if backend == "fal":
        if fal.api_key:
            return "🟢 fal.ai Flux: Ready (Flux品質・NSFW OK・在庫切れなし)"
        return "🔴 fal.ai API Key が未設定 — Settingsタブで設定してください"
    if backend == "together":
        if together.api_key:
            return "🟢 Together.ai: Ready (Flux品質・NSFW OK・LoRA対応)"
        return "🔴 Together.ai API Key が未設定 — Settingsタブで設定してください"
    if backend == "dezgo":
        if dezgo.api_key:
            return "🟢 Dezgo: Ready (完全無検閲・画像+動画・安い)"
        return "🔴 Dezgo API Key が未設定 — Settingsタブで設定してください"
    if backend == "novita":
        if novita.api_key:
            return "🟢 Novita.ai: Ready (無検閲モデル・NSFW OK・安い)"
        return "🔴 Novita.ai API Key が未設定 — Settingsタブで設定してください"
    if backend == "civitai":
        if civitai.api_key:
            return "🟢 CivitAI Generation: Ready (NSFW OK・全モデル使える・在庫切れなし)"
        return "🔴 CivitAI API Key が未設定 — Settingsタブで設定してください"
    if backend == "vast":
        # vast.ai: check remote ComfyUI via configured URL
        if client.is_server_running():
            return f"🟢 vast.ai ComfyUI: Running ({config.get('vast_comfyui_url', config['comfyui_url'])})"
        return "🔴 vast.ai ComfyUI: Not Reachable — トンネルが切れている可能性。Settings確認"
    if client.is_server_running():
        label = "Local" if backend == "local" else "RunPod Cloud"
        status_line = f"🟢 ComfyUI Server: Running ({label})"
        # Add cost info for RunPod
        if backend == "runpod" and runpod.api_key:
            pod_id = config.get("runpod_pod_id", "")
            if pod_id:
                try:
                    pod = runpod.get_pod(pod_id)
                    if pod:
                        gpu = pod.get("machine", {}).get("gpuDisplayName", "")
                        from runpod_manager import GPU_HOURLY_RATES
                        rate = GPU_HOURLY_RATES.get(gpu, 0.30)
                        uptime_secs = 0
                        if pod.get("runtime") and pod["runtime"].get("uptimeInSeconds"):
                            uptime_secs = pod["runtime"]["uptimeInSeconds"]
                        hours = uptime_secs / 3600
                        cost = hours * rate
                        mins = uptime_secs // 60
                        status_line += f" | {gpu} | ${rate:.2f}/hr | {int(mins)}分 | 💰${cost:.3f} (¥{int(cost*150)})"
                except Exception:
                    pass
        return status_line
    if backend == "runpod":
        return "🔴 ComfyUI Server: Not Running — Backend を runpod に切り替えると自動接続します"
    return "🔴 ComfyUI Server: Not Running — launch.commandで起動してください"


def _gradio_safe_video(video_path):
    """Copy video to temp dir so Gradio can serve it (bypasses allowed_paths restrictions)."""
    import tempfile
    if not video_path or not os.path.exists(video_path):
        return video_path
    tmp = os.path.join(tempfile.gettempdir(), os.path.basename(video_path))
    shutil.copy2(video_path, tmp)
    return tmp


def _resolve_lora_to_url(lora_name: str) -> str | None:
    """Convert a local LoRA filename to a CivitAI download URL for cloud backends.

    Strategy:
    1. Search CivitAI for the filename
    2. Return the download URL if found
    3. Return None if not found (skip LoRA)
    """
    if not lora_name or lora_name == "None":
        return None

    # Strip .safetensors extension for search
    search_name = lora_name.replace(".safetensors", "").replace(".ckpt", "")

    try:
        if civitai.api_key:
            # Search CivitAI for this model
            results = civitai._get("/models", {
                "query": search_name,
                "types": "LORA",
                "limit": "5",
                "sort": "Most Downloaded",
            })
            for model in results.get("items", []):
                versions = model.get("modelVersions", [])
                for version in versions:
                    for f in version.get("files", []):
                        if f.get("name", "").lower() == lora_name.lower():
                            url = f.get("downloadUrl", "")
                            if url and civitai.api_key:
                                url += f"?token={civitai.api_key}"
                            return url

            # If exact match not found, try the first result's latest version
            if results.get("items"):
                first = results["items"][0]
                versions = first.get("modelVersions", [])
                if versions:
                    download_url = versions[0].get("downloadUrl", "")
                    if download_url and civitai.api_key:
                        download_url += f"?token={civitai.api_key}"
                    return download_url
    except Exception as e:
        logger.warning(f"CivitAI LoRA search failed for '{lora_name}': {e}")

    return None


def save_image_to_dir(image, output_dir, prefix="img"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    image.save(filepath)
    return filepath


def generate_image(prompt, negative_prompt, model, lora, lora_strength, vae,
                   width, height, steps, cfg, sampler, scheduler, seed,
                   batch_size, hires_fix=False, hires_scale=1.5, hires_denoise=0.5,
                   hires_steps=15, upscale_model="", mode="normal",
                   face_detailer=False, face_denoise=0.4, face_guide_size=512):
    """Generate image - routes to CivitAI / Replicate API / ComfyUI based on backend."""
    backend = config.get("backend", "local")

    # ── fal.ai backend: Flux quality, NSFW OK ──
    if backend == "fal":
        # CivitAIモデル or iCloudモデルが選ばれた場合、CivitAIバックエンドに自動ルーティング
        if model and (model in CIVITAI_GENERATION_MODELS or model.startswith("[iCloud] ")):
            backend = "civitai"
            # CivitAIの処理にフォールスルー
        elif mode == "adult" and novita.api_key:
            # Adult mode: prefer Novita.ai for fully uncensored CivitAI models
            backend = "novita"
            # Novitaの処理にフォールスルー
        else:
            if not fal.api_key:
                raise gr.Error("fal.ai API Key が未設定です。Settingsタブで設定してください。")
            if not prompt.strip():
                raise gr.Error("プロンプトを入力してください。")

            w, h = int(width), int(height)
            w = max(256, (w // 32) * 32)
            h = max(256, (h // 32) * 32)
            # Use selected model if it's a fal model, otherwise default
            if model and model in FAL_MODELS:
                model_key = model
            else:
                model_key = "Flux Dev (高品質・NSFW OK)"
            # Adult mode: avoid Pro 1.1 (blocks NSFW with black image)
            if mode == "adult" and not FAL_MODELS.get(model_key, {}).get("nsfw", True):
                model_key = "Flux Dev (高品質・NSFW OK)"

            # Resolve LoRA: local filename → CivitAI download URL for fal.ai
            lora_urls = None
            if lora and lora != "None":
                lora_url = _resolve_lora_to_url(lora)
                if lora_url:
                    lora_urls = [(lora_url, float(lora_strength))]
                    model_key = "Flux + LoRA (カスタムLoRA・NSFW OK)"
                    logger.info(f"fal.ai LoRA resolved: {lora} → {lora_url}")

            try:
                urls = fal.generate_image(
                    model_key=model_key,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=w, height=h,
                    num_images=int(batch_size),
                    seed=int(seed),
                    safety_checker=False,
                    lora_urls=lora_urls,
                )

                images = []
                saved_paths = []
                output_dir = config["output_dir_adult"] if mode == "adult" else config["output_dir_normal"]
                for url in urls:
                    img = download_fal_image(url)
                    images.append(img)
                    path = save_image_to_dir(img, output_dir, prefix=f"fal_{mode}")
                    saved_paths.append(path)

                lora_label = f" + LoRA: {lora}" if lora_urls else ""
                cost = FAL_MODELS.get(model_key, {}).get("cost", "?")
                return images, f"[fal.ai {model_key}{lora_label}] コスト: {cost}\n保存先: {', '.join(saved_paths)}"
            except Exception as e:
                logger.error(f"fal.ai生成エラー: {traceback.format_exc()}")
                raise gr.Error(f"fal.ai生成エラー: {e}")

    # ── Together.ai backend: Flux + NSFW OK ──
    if backend == "together":
        if not together.api_key:
            raise gr.Error("Together.ai API Key が未設定です。Settingsタブで設定してください。")
        if not prompt.strip():
            raise gr.Error("プロンプトを入力してください。")
        w, h = int(width), int(height)
        w = max(256, (w // 32) * 32)
        h = max(256, (h // 32) * 32)
        model_key = "Flux.1 Dev (高品質)"
        try:
            b64_images = together.generate_image(
                model_key=model_key, prompt=prompt, negative_prompt=negative_prompt,
                width=w, height=h, num_images=int(batch_size), seed=int(seed),
            )
            images = []
            saved_paths = []
            output_dir = config["output_dir_adult"] if mode == "adult" else config["output_dir_normal"]
            for b64 in b64_images:
                img = decode_together_image(b64)
                images.append(img)
                path = save_image_to_dir(img, output_dir, prefix=f"together_{mode}")
                saved_paths.append(path)
            cost = TOGETHER_MODELS.get(model_key, {}).get("cost", "?")
            return images, f"[Together.ai {model_key}] コスト: {cost}\n保存先: {', '.join(saved_paths)}"
        except Exception as e:
            logger.error(f"Together.ai生成エラー: {traceback.format_exc()}")
            raise gr.Error(f"Together.ai生成エラー: {e}")

    # ── Dezgo backend: uncensored ──
    if backend == "dezgo":
        if not dezgo.api_key:
            raise gr.Error("Dezgo API Key が未設定です。Settingsタブで設定してください。")
        if not prompt.strip():
            raise gr.Error("プロンプトを入力してください。")
        w, h = int(width), int(height)
        model_key = "Flux Dev (高品質)"
        try:
            png_bytes = dezgo.generate_image(
                model_key=model_key, prompt=prompt, negative_prompt=negative_prompt,
                width=w, height=h, seed=int(seed),
            )
            img = decode_dezgo_image(png_bytes)
            output_dir = config["output_dir_adult"] if mode == "adult" else config["output_dir_normal"]
            path = save_image_to_dir(img, output_dir, prefix=f"dezgo_{mode}")
            cost = DEZGO_IMAGE_MODELS.get(model_key, {}).get("cost", "?")
            return [img], f"[Dezgo {model_key}] コスト: {cost}\n保存先: {path}"
        except Exception as e:
            logger.error(f"Dezgo生成エラー: {traceback.format_exc()}")
            raise gr.Error(f"Dezgo生成エラー: {e}")

    # ── Novita.ai backend: uncensored models ──
    if backend == "novita":
        if not novita.api_key:
            raise gr.Error("Novita.ai API Key が未設定です。Settingsタブで設定してください。")
        if not prompt.strip():
            raise gr.Error("プロンプトを入力してください。")
        w, h = int(width), int(height)
        model_key = "Flux Dev (高品質)"
        try:
            urls = novita.generate_image(
                model_key=model_key, prompt=prompt, negative_prompt=negative_prompt,
                width=w, height=h, num_images=int(batch_size), seed=int(seed),
            )
            images = []
            saved_paths = []
            output_dir = config["output_dir_adult"] if mode == "adult" else config["output_dir_normal"]
            for url in urls:
                img = download_novita_image(url)
                images.append(img)
                path = save_image_to_dir(img, output_dir, prefix=f"novita_{mode}")
                saved_paths.append(path)
            cost = NOVITA_MODELS.get(model_key, {}).get("cost", "?")
            return images, f"[Novita.ai {model_key}] コスト: {cost}\n保存先: {', '.join(saved_paths)}"
        except Exception as e:
            logger.error(f"Novita.ai生成エラー: {traceback.format_exc()}")
            raise gr.Error(f"Novita.ai生成エラー: {e}")

    # ── CivitAI backend: cloud GPU, NSFW OK ──
    if backend == "civitai":
        if not civitai.api_key:
            raise gr.Error("CivitAI API Key が未設定です。Settingsタブで設定してください。")
        if not prompt.strip():
            raise gr.Error("プロンプトを入力してください。")

        # ── LoRA URN解決 (CivitAIクラウドでLoRAを使うにはURNが必要) ──
        lora_urns = None
        if lora and lora not in ("None", ""):
            lora_urn_info = civitai.resolve_icloud_model_urn(lora)
            if lora_urn_info and "lora" in lora_urn_info.get("type", "").lower():
                lora_urns = [(lora_urn_info["urn"], float(lora_strength))]
                logger.info(f"CivitAI LoRA URN解決: {lora} → {lora_urn_info['urn']}")

        # ── iCloudモデルが選ばれた場合: CivitAI APIでURNを解決してクラウド生成 ──
        if model and model.startswith("[iCloud] "):
            icloud_filename = model[len("[iCloud] "):]
            logger.info(f"iCloudモデル選択: {icloud_filename} → CivitAI URN解決中...")
            try:
                urn_info = civitai.resolve_icloud_model_urn(icloud_filename)
                if not urn_info:
                    raise gr.Error(
                        f"CivitAIでモデルが見つかりません: {icloud_filename}\n"
                        "モデルがCivitAIに存在するか確認してください。"
                    )
                if urn_info.get("type", "").lower() == "lora":
                    raise gr.Error(
                        f"{icloud_filename} はLoRAです。Checkpointモデルとして使えません。\n"
                        "LoRAドロップダウンから選択してください。"
                    )

                logger.info(f"URN解決成功: {urn_info['name']} → {urn_info['urn']}")
                urls = civitai.generate_image_by_urn(
                    model_urn=urn_info["urn"],
                    base_model=urn_info["base"],
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=int(width), height=int(height),
                    steps=int(steps), cfg_scale=float(cfg),
                    seed=int(seed), quantity=int(batch_size),
                    lora_urns=lora_urns,
                )

                images = []
                saved_paths = []
                output_dir = config["output_dir_adult"] if mode == "adult" else config["output_dir_normal"]
                for url in urls:
                    img = download_url_to_pil(url)
                    images.append(img)
                    path = save_image_to_dir(img, output_dir, prefix=f"civitai_icloud_{mode}")
                    saved_paths.append(path)

                cost = urn_info.get("cost", "~4 Buzz")
                model_label = f"{urn_info['name']} ({urn_info.get('version', '')})"
                lora_label = f" + LoRA: {lora}" if lora_urns else ""
                return images, f"[CivitAI iCloud: {model_label}{lora_label}] コスト: {cost}/枚\n保存先: {', '.join(saved_paths)}"
            except gr.Error:
                raise
            except Exception as e:
                logger.error(f"CivitAI iCloud生成エラー: {traceback.format_exc()}")
                raise gr.Error(f"CivitAI iCloudモデル生成エラー: {e}")

        # ── 既存のCivitAIクラウドモデル ──
        civitai_model_key = None
        if model and model in CIVITAI_GENERATION_MODELS:
            civitai_model_key = model
        else:
            civitai_model_key = "Juggernaut XL Ragnarok (SDXL 最高品質)"

        try:
            urls = civitai.generate_image(
                model_key=civitai_model_key,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=int(width), height=int(height),
                steps=int(steps), cfg_scale=float(cfg),
                seed=int(seed), quantity=int(batch_size),
                lora_urns=lora_urns,
            )

            images = []
            saved_paths = []
            output_dir = config["output_dir_adult"] if mode == "adult" else config["output_dir_normal"]
            for url in urls:
                img = download_url_to_pil(url)
                images.append(img)
                path = save_image_to_dir(img, output_dir, prefix=f"civitai_{mode}")
                saved_paths.append(path)

            cost = CIVITAI_GENERATION_MODELS.get(civitai_model_key, {}).get("cost", "?")
            lora_label = f" + LoRA: {lora}" if lora_urns else ""
            return images, f"[CivitAI: {civitai_model_key}{lora_label}] コスト: {cost}/枚\n保存先: {', '.join(saved_paths)}"
        except Exception as e:
            logger.error(f"CivitAI生成エラー: {traceback.format_exc()}")
            raise gr.Error(f"CivitAI生成エラー: {e}")

    # ── Replicate backend: use cloud API (no ComfyUI needed) ──
    if backend == "replicate":
        if not replicate.api_key:
            raise gr.Error("Replicate API Key が未設定です。Settingsタブで設定してください。")
        if not prompt.strip():
            raise gr.Error("プロンプトを入力してください。")

        # Auto-select best Replicate model and clamp params
        w, h = int(width), int(height)
        # Replicate models need dimensions in multiples of 32, min 256
        w = max(256, (w // 32) * 32)
        h = max(256, (h // 32) * 32)
        if w >= 1024 or h >= 1024:
            model_key = "Flux Dev (高品質・バランス)"
        else:
            model_key = "Flux Schnell (高速・安い)"

        try:
            # Don't pass steps/guidance that conflict with model defaults
            # Flux Schnell: max 4 steps, no guidance. Flux Dev: max 50 steps.
            urls = replicate.generate_image(
                model_key=model_key,
                prompt=prompt,
                width=w, height=h,
                num_outputs=int(batch_size),
                seed=int(seed),
            )

            images = []
            saved_paths = []
            output_dir = config["output_dir_adult"] if mode == "adult" else config["output_dir_normal"]
            for url in urls:
                img = download_url_to_pil(url)
                images.append(img)
                path = save_image_to_dir(img, output_dir, prefix=f"replicate_{mode}")
                saved_paths.append(path)

            from replicate_api import MODELS as REPLICATE_MODELS_MAP
            cost = REPLICATE_MODELS_MAP.get(model_key, {}).get("cost_per_image", "?")
            return images, f"[Replicate: {model_key}] コスト: {cost}/枚\n保存先: {', '.join(saved_paths)}"
        except Exception as e:
            logger.error(f"Replicate生成エラー: {traceback.format_exc()}")
            raise gr.Error(f"Replicate生成エラー: {e}")

    # ── Local / RunPod / vast.ai backend: use ComfyUI ──
    if not client.is_server_running():
        backend_label = {
            "vast": "vast.ai ComfyUI",
            "runpod": "RunPod ComfyUI",
            "local": "Local ComfyUI",
        }.get(backend, "ComfyUI Server")
        raise gr.Error(
            f"{backend_label} に接続できません ({client.server_url})\n"
            "💡 vast.ai: インスタンスとCloudflareトンネルが起動しているか確認\n"
            "💡 または Settings → Backend を 'fal' / 'novita' / 'replicate' に切り替え"
        )

    if not model:
        raise gr.Error("モデルが選択されていません。modelsフォルダにモデルを配置してください。")

    lora_name = "" if lora == "None" else lora
    vae_name = "" if vae == "None" else vae
    upscale_name = "" if not upscale_model or upscale_model == "None" else upscale_model

    workflow = build_txt2img_workflow(
        prompt=prompt,
        negative_prompt=negative_prompt,
        model=model,
        width=width,
        height=height,
        steps=steps,
        cfg=cfg,
        sampler=sampler,
        scheduler=scheduler,
        seed=int(seed),
        batch_size=int(batch_size),
        lora_name=lora_name,
        lora_strength=lora_strength,
        vae_name=vae_name,
        hires_fix=hires_fix,
        hires_scale=hires_scale,
        hires_denoise=hires_denoise,
        hires_steps=hires_steps,
        upscale_model=upscale_name,
        face_detailer=face_detailer,
        face_denoise=face_denoise,
        face_guide_size=face_guide_size,
    )

    timeout = 1200 if (hires_fix or face_detailer) else 900
    images = client.generate(workflow, timeout=timeout)

    output_dir = config["output_dir_adult"] if mode == "adult" else config["output_dir_normal"]
    saved_paths = []
    for img in images:
        path = save_image_to_dir(img, output_dir, prefix=mode)
        saved_paths.append(path)

    hires_info = f" [Hires Fix: {hires_scale}x, denoise={hires_denoise}]" if hires_fix else ""
    face_info = f" [FaceDetailer: denoise={face_denoise}]" if face_detailer else ""
    return images, f"保存先: {', '.join(saved_paths)}{hires_info}{face_info}"


# ──────────────────────────────────────────────
# Generate functions for each tab
# ──────────────────────────────────────────────

def _apply_cinema_and_grade(prompt, images, cinema_preset="(なし)", color_grade="(なし)"):
    """Apply cinema preset to prompt and color grading to generated images."""
    # Cinema preset: append suffix to prompt
    final_prompt = prompt
    if cinema_preset and cinema_preset != "(なし)" and cinema_preset in CINEMA_PRESETS:
        final_prompt = prompt + CINEMA_PRESETS[cinema_preset]

    # Color grading: apply Pillow filters to each image
    if color_grade and color_grade != "(なし)" and images:
        graded = []
        for img in images:
            if isinstance(img, Image.Image):
                graded.append(apply_color_grade(img, color_grade))
            else:
                graded.append(img)
        return final_prompt, graded
    return final_prompt, images


def generate_normal(prompt, neg, model, lora, lora_str, vae, w, h, steps, cfg, sampler, sched, seed, batch,
                    hires_fix, hires_scale, hires_denoise, hires_steps, upscale_model,
                    face_detailer=False, face_denoise=0.4, face_guide_size=512,
                    cinema_preset="(なし)", color_grade="(なし)"):
    final_prompt = prompt
    if cinema_preset and cinema_preset != "(なし)" and cinema_preset in CINEMA_PRESETS:
        final_prompt = prompt + CINEMA_PRESETS[cinema_preset]
    images, info = generate_image(final_prompt, neg, model, lora, lora_str, vae, w, h, steps, cfg, sampler, sched, seed, batch,
                                  hires_fix, hires_scale, hires_denoise, hires_steps, upscale_model, "normal",
                                  face_detailer, face_denoise, face_guide_size)
    if color_grade and color_grade != "(なし)" and images:
        images = [apply_color_grade(img, color_grade) if isinstance(img, Image.Image) else img for img in images]
    cinema_info = f" [Cinema: {cinema_preset}]" if cinema_preset != "(なし)" else ""
    grade_info = f" [Grade: {color_grade}]" if color_grade != "(なし)" else ""
    return images, info + cinema_info + grade_info


def generate_adult(prompt, neg, model, lora, lora_str, vae, w, h, steps, cfg, sampler, sched, seed, batch,
                   hires_fix, hires_scale, hires_denoise, hires_steps, upscale_model,
                   face_detailer=False, face_denoise=0.4, face_guide_size=512,
                   cinema_preset="(なし)", color_grade="(なし)"):
    """Adult image generation with robust fallback chain.

    Uses _adult_generate_image() → ComfyUI (vast.ai) → Novita → Dezgo → CivitAI → fal.ai.
    Never fails hard on transient network issues.
    """
    if not prompt or not prompt.strip():
        raise gr.Error("プロンプトを入力してください。")

    final_prompt = prompt
    if cinema_preset and cinema_preset != "(なし)" and cinema_preset in CINEMA_PRESETS:
        final_prompt = prompt + CINEMA_PRESETS[cinema_preset]

    # Resolve model override: (自動) or None → auto pick. Otherwise pass through.
    model_override = model if model and model not in ("(自動)", "None", "") else None

    # CivitAI cloud model or iCloud model selection routes via civitai_model argument
    civitai_model = None
    if model_override and (model_override in CIVITAI_GENERATION_MODELS or model_override.startswith("[iCloud] ")):
        civitai_model = model_override
        model_override = None

    images, info = _adult_generate_image(
        prompt=final_prompt,
        negative=neg or config.get("default_negative_prompt", ""),
        w=int(w), h=int(h),
        seed=int(seed) if seed is not None else -1,
        prefix="adult",
        civitai_model=civitai_model,
        model_override=model_override,
        lora_override=lora,
        lora_str=lora_str,
        vae_override=vae,
        steps_override=int(steps) if steps else None,
        cfg_override=float(cfg) if cfg else None,
        sampler_override=sampler,
        scheduler_override=sched,
        hires_fix=bool(hires_fix),
        hires_scale=float(hires_scale) if hires_scale else 1.5,
        hires_denoise=float(hires_denoise) if hires_denoise else 0.5,
        hires_steps=int(hires_steps) if hires_steps else 15,
        upscale_model=upscale_model if upscale_model and upscale_model != "None" else "",
        face_detailer=bool(face_detailer),
        face_denoise=float(face_denoise) if face_denoise else 0.4,
        face_guide_size=int(face_guide_size) if face_guide_size else 512,
    )

    if color_grade and color_grade != "(なし)" and images:
        images = [apply_color_grade(img, color_grade) if isinstance(img, Image.Image) else img for img in images]

    cinema_info = f" [Cinema: {cinema_preset}]" if cinema_preset != "(なし)" else ""
    grade_info = f" [Grade: {color_grade}]" if color_grade != "(なし)" else ""
    return images, info + cinema_info + grade_info


def refine_image(gallery_selection, prompt, negative, model, seed,
                 denoise=0.35, upscale_scale=1.5, face_fix=True, mode="adult"):
    """Refine a generated image: upscale + light denoise + face detailer.

    Takes image from gallery, runs through refine pipeline, returns enhanced version.
    """
    from PIL import Image as PILImage
    import numpy as np

    # Extract image from gallery selection
    if gallery_selection is None:
        raise gr.Error("リファインする画像を選択してください。ギャラリーの画像をクリックして選択。")

    # gallery_selection can be: filepath string, tuple (filepath, caption), or PIL Image
    if isinstance(gallery_selection, (list, tuple)):
        img_source = gallery_selection[0] if gallery_selection else None
    else:
        img_source = gallery_selection

    if img_source is None:
        raise gr.Error("画像が見つかりません。")

    # Get image as PIL
    if isinstance(img_source, str):
        img = PILImage.open(img_source)
    elif isinstance(img_source, np.ndarray):
        img = PILImage.fromarray(img_source)
    else:
        img = img_source

    output_dir = config["output_dir_adult"] if mode == "adult" else config["output_dir_normal"]
    os.makedirs(output_dir, exist_ok=True)
    s = int(seed) if seed is not None and seed >= 0 else -1

    # Save source image temporarily for ComfyUI
    ts = int(time.time())
    temp_path = os.path.join(output_dir, f"_refine_src_{ts}.png")
    img.save(temp_path)

    # Try ComfyUI refine workflow
    if client.is_server_running():
        try:
            # Upload image to ComfyUI
            img_name = client.upload_image(temp_path)
            if not img_name:
                img_name = os.path.basename(temp_path)

            models = client.get_models()
            if not models:
                models = _get_models_for_backend()

            chosen_model = model if model and model not in ("None", "(自動)", "") else (models[0] if models else None)
            if not chosen_model:
                raise gr.Error("モデルが見つかりません。")

            neg = negative or "ugly, deformed, blurry, low quality, bad anatomy, extra fingers, watermark, text"

            workflow = build_refine_workflow(
                prompt=prompt or "high quality, detailed, sharp focus, professional photography",
                negative_prompt=neg,
                model=chosen_model,
                image_path=img_name,
                width=img.width,
                height=img.height,
                steps=25,
                cfg=7.0,
                denoise=float(denoise),
                upscale_model="4x-UltraSharp.pth",
                upscale_scale=float(upscale_scale),
                face_detailer=bool(face_fix),
                face_denoise=0.35,
                face_guide_size=768,
                seed=s,
            )
            result = client.generate(workflow, timeout=1800)
            if result:
                saved = []
                for r_img in result:
                    saved.append(save_image_to_dir(r_img, output_dir, prefix="refined"))
                return result, f"[Refine完了] Upscale {upscale_scale}x + Denoise {denoise} + FaceDetailer\n保存先: {', '.join(saved)}"
        except Exception as e:
            # Clean up temp
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise gr.Error(f"リファイン失敗: {e}")
    else:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise gr.Error("ComfyUIが起動していません。リファインにはComfyUIが必要です。")

    # Clean up temp
    if os.path.exists(temp_path):
        os.remove(temp_path)
    return [img], "リファイン処理に失敗しました。"


# ──────────────────────────────────────────────
# Video generation
# ──────────────────────────────────────────────

def extract_last_frame(video_path):
    """Extract the last frame from a video file for clip chaining."""
    if not video_path:
        return None, "動画が指定されていません"
    try:
        from PIL import Image as PILImage
        import subprocess
        # Use ffmpeg to extract last frame
        out_path = os.path.join(config["output_dir_normal"], f"lastframe_{int(time.time())}.png")
        result = subprocess.run(
            ["ffmpeg", "-sseof", "-0.1", "-i", video_path, "-update", "1", "-q:v", "1", "-y", out_path],
            capture_output=True, timeout=30,
        )
        if result.returncode == 0 and os.path.exists(out_path):
            img = PILImage.open(out_path)
            return img, f"最終フレーム抽出完了: {out_path}"
        # Fallback: try with PIL for GIF
        if video_path.lower().endswith(".gif"):
            gif = PILImage.open(video_path)
            gif.seek(gif.n_frames - 1)
            frame = gif.copy().convert("RGB")
            frame.save(out_path)
            return frame, f"最終フレーム抽出完了: {out_path}"
        return None, f"フレーム抽出失敗: {result.stderr.decode()[:200]}"
    except Exception as e:
        return None, f"エラー: {e}"


def generate_video_txt2vid(prompt, neg, model, motion_model, lora, lora_str, vae,
                           w, h, steps, cfg, sampler, sched, seed,
                           frame_count, fps, output_format, mode="normal",
                           cloud_model=None, duration=None):
    """Generate video from text prompt - routes to cloud API or AnimateDiff.

    Duration: seconds (cloud only). For local AnimateDiff, duration is derived
    from frame_count/fps. If duration provided for local mode, frame_count is
    recomputed as int(duration * fps).
    """
    backend = config.get("backend", "local")

    # ── Cloud backends: use fal.ai for video ──
    if backend != "local":
        if not prompt.strip():
            raise gr.Error("プロンプトを入力してください。")
        if fal.api_key:
            fal_model = cloud_model or "LTX 2.3 (高速・安い)"
            dur = int(duration) if duration else None
            video_path, info = generate_fal_video(
                prompt, fal_model, mode, duration=dur,
                width=int(w) if w else None, height=int(h) if h else None,
                negative_prompt=neg, seed=int(seed) if seed is not None else -1,
            )
            return _gradio_safe_video(video_path), [], info
        else:
            raise gr.Error(
                "動画生成には fal.ai API Key が必要です。\n"
                "Settingsタブで設定してください。"
            )

    # Local AnimateDiff: derive frames from duration if specified
    if duration and duration > 0:
        frame_count = int(float(duration) * float(fps))

    if not client.is_server_running():
        raise gr.Error(
            "ComfyUI Server が起動していません。\n"
            "💡 Backend を fal / replicate / dezgo に切り替えると\n"
            "ComfyUI不要でクラウドから動画生成できます。"
        )
    if not model:
        raise gr.Error("モデルが選択されていません。")
    if not motion_model:
        raise gr.Error("Motion Model が選択されていません。AnimateDiffモデルをダウンロードしてください。")

    lora_name = "" if lora == "None" else lora
    vae_name = "" if vae == "None" else vae

    workflow = build_animatediff_workflow(
        prompt=prompt,
        negative_prompt=neg,
        model=model,
        motion_model=motion_model,
        width=int(w),
        height=int(h),
        steps=int(steps),
        cfg=cfg,
        sampler=sampler,
        scheduler=sched,
        seed=int(seed),
        frame_count=int(frame_count),
        fps=int(fps),
        lora_name=lora_name,
        lora_strength=lora_str,
        vae_name=vae_name,
        output_format=output_format,
    )

    output_dir = config["output_dir_adult"] if mode == "adult" else config["output_dir_normal"]
    frames, video_path = client.generate_video(workflow, output_dir, timeout=900)
    info = f"Frames: {len(frames)}"
    if video_path:
        info += f"\n保存先: {video_path}"
    return _gradio_safe_video(video_path), frames[:4] if frames else [], info


def generate_video_img2vid(image, prompt, neg, model, motion_model, vae,
                           w, h, steps, cfg, sampler, sched, seed,
                           frame_count, fps, denoise, output_format, mode="normal",
                           cloud_model=None, duration=None):
    """Generate video from input image using AnimateDiff img2vid or fal.ai cloud.

    Duration: seconds (cloud only). For local AnimateDiff, derived from frame_count/fps.
    If duration provided in local mode, frame_count is recomputed.
    """
    backend = config.get("backend", "local")

    # ── Cloud backends: use fal.ai for img2vid ──
    if backend != "local":
        if image is None:
            raise gr.Error("入力画像を選択してください。")
        if fal.api_key:
            fal_model = cloud_model or "Kling 2.5 Turbo Pro img2vid (高品質・SFW)"
            dur = int(duration) if duration else None
            video_path, info = generate_fal_img2vid(image, prompt, fal_model, mode, dur)
            return _gradio_safe_video(video_path), [], info
        else:
            raise gr.Error("fal.ai API Key が必要です。Settingsタブで設定してください。")

    # Local AnimateDiff: derive frames from duration if specified
    if duration and duration > 0:
        frame_count = int(float(duration) * float(fps))

    if not client.is_server_running():
        raise gr.Error("ComfyUI Server が起動していません。")
    if not model:
        raise gr.Error("モデルが選択されていません。")
    if image is None:
        raise gr.Error("入力画像を選択してください。")

    vae_name = "" if vae == "None" else vae

    # Save uploaded image to ComfyUI input dir
    from PIL import Image as PILImage
    comfyui_input = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "comfyui", "input")
    os.makedirs(comfyui_input, exist_ok=True)
    input_filename = f"img2vid_input_{int(time.time())}.png"
    input_path = os.path.join(comfyui_input, input_filename)
    if isinstance(image, str):
        shutil.copy(image, input_path)
    else:
        PILImage.fromarray(image).save(input_path)

    workflow = build_img2vid_workflow(
        image_path=input_path,
        model=model,
        motion_model=motion_model,
        width=int(w),
        height=int(h),
        steps=int(steps),
        cfg=cfg,
        sampler=sampler,
        scheduler=sched,
        seed=int(seed),
        frame_count=int(frame_count),
        fps=int(fps),
        denoise=denoise,
        prompt=prompt,
        negative_prompt=neg,
        output_format=output_format,
    )

    output_dir = config["output_dir_adult"] if mode == "adult" else config["output_dir_normal"]
    frames, video_path = client.generate_video(workflow, output_dir, timeout=900)
    info = f"Frames: {len(frames)}"
    if video_path:
        info += f"\n保存先: {video_path}"
    return _gradio_safe_video(video_path), frames[:4] if frames else [], info


# Flux default model names (pre-configured for RunPod templates)
FLUX_DEFAULTS = {
    "unet": "flux1-dev-fp8.safetensors",
    "clip_l": "clip_l.safetensors",
    "t5xxl": "t5xxl_fp8_e4m3fn.safetensors",
    "vae": "ae.safetensors",
}


def generate_flux(prompt, unet_model, clip_l, t5xxl, vae_name, lora, lora_strength,
                  width, height, steps, guidance, sampler, scheduler, seed, batch_size,
                  weight_dtype, mode="normal"):
    """Generate image using Flux.1 model via ComfyUI."""
    if not client.is_server_running():
        raise gr.Error("ComfyUI Server が起動していません。Settingsタブで RunPod を起動してください。")
    if config.get("backend") != "runpod":
        raise gr.Error("Flux は RunPod でのみ使用できます。\n① Settingsタブで RunPod を起動\n② 「状態確認/接続」で接続\n③ Backend を runpod に切り替え")

    # Use defaults if not specified
    unet_model = unet_model or FLUX_DEFAULTS["unet"]
    clip_l = clip_l or FLUX_DEFAULTS["clip_l"]
    t5xxl = t5xxl or FLUX_DEFAULTS["t5xxl"]
    vae_name = vae_name or FLUX_DEFAULTS["vae"]

    lora_name = "" if not lora or lora == "None" else lora

    workflow = build_flux_workflow(
        prompt=prompt,
        unet_model=unet_model,
        clip_l=clip_l,
        t5xxl=t5xxl,
        vae_name=vae_name,
        width=width,
        height=height,
        steps=steps,
        guidance=guidance,
        sampler=sampler,
        scheduler=scheduler,
        seed=int(seed),
        batch_size=int(batch_size),
        weight_dtype=weight_dtype,
        lora_name=lora_name,
        lora_strength=lora_strength,
    )

    images = client.generate(workflow, timeout=900)

    output_dir = config["output_dir_adult"] if mode == "adult" else config["output_dir_normal"]
    saved_paths = []
    for img in images:
        path = save_image_to_dir(img, output_dir, prefix=f"flux_{mode}")
        saved_paths.append(path)

    return images, f"[Flux] 保存先: {', '.join(saved_paths)}"


# ──────────────────────────────────────────────
# New advanced features (ai-studio parity)
# ──────────────────────────────────────────────

def _upload_image_to_fal(image, output_dir, prefix="upload"):
    """Helper: save image to temp file, upload to fal storage, return URL."""
    import fal_client as fc
    from PIL import Image as PILImage
    os.environ["FAL_KEY"] = fal.api_key
    os.makedirs(output_dir, exist_ok=True)
    ts = int(time.time())
    if isinstance(image, str):
        img_path = image
    else:
        img_path = os.path.join(output_dir, f"{prefix}_{ts}.png")
        PILImage.fromarray(image).save(img_path)
    return fc.upload_file(img_path)


def generate_style_transfer(image, style_key, custom_prompt, strength, mode="normal"):
    """Apply style transfer to an image via fal.ai."""
    if not fal.api_key:
        raise gr.Error("fal.ai API Key が未設定です。Settingsタブで設定してください。")
    if image is None:
        raise gr.Error("画像をアップロードしてください。")
    try:
        output_dir = config["output_dir_adult"] if mode == "adult" else config["output_dir_normal"]
        image_url = _upload_image_to_fal(image, output_dir, "style_input")
        urls = fal.style_transfer(image_url, style_key, custom_prompt, strength)
        if not urls:
            raise RuntimeError("スタイル転送結果が取得できませんでした")
        img = download_fal_image(urls[0])
        save_path = save_image_to_dir(img, output_dir, prefix=f"style_{mode}")
        style_name = style_key.split("(")[0].strip()
        return img, f"[Style Transfer: {style_name}] strength={strength}\n保存先: {save_path}"
    except Exception as e:
        logger.error(f"スタイル転送エラー: {traceback.format_exc()}")
        raise gr.Error(f"スタイル転送エラー: {e}")


def generate_inpaint(image, mask, prompt, negative_prompt, width, height, steps, guidance, seed, mode="normal"):
    """Inpaint image region via fal.ai."""
    if not fal.api_key:
        raise gr.Error("fal.ai API Key が未設定です。Settingsタブで設定してください。")
    if image is None:
        raise gr.Error("画像をアップロードしてください。")
    if mask is None:
        raise gr.Error("マスク画像をアップロードしてください（白=編集エリア）。")
    if not prompt.strip():
        raise gr.Error("プロンプトを入力してください。")
    try:
        output_dir = config["output_dir_adult"] if mode == "adult" else config["output_dir_normal"]
        image_url = _upload_image_to_fal(image, output_dir, "inpaint_image")
        mask_url = _upload_image_to_fal(mask, output_dir, "inpaint_mask")
        urls = fal.inpaint(
            image_url, mask_url, prompt, negative_prompt,
            int(width), int(height), int(steps), float(guidance), int(seed)
        )
        if not urls:
            raise RuntimeError("インペイント結果が取得できませんでした")
        img = download_fal_image(urls[0])
        save_path = save_image_to_dir(img, output_dir, prefix=f"inpaint_{mode}")
        return img, f"[Inpaint] 完了\n保存先: {save_path}"
    except Exception as e:
        logger.error(f"インペイントエラー: {traceback.format_exc()}")
        raise gr.Error(f"インペイントエラー: {e}")


def generate_remove_bg(image, mode="normal"):
    """Remove background via fal.ai."""
    if not fal.api_key:
        raise gr.Error("fal.ai API Key が未設定です。Settingsタブで設定してください。")
    if image is None:
        raise gr.Error("画像をアップロードしてください。")
    try:
        output_dir = config["output_dir_adult"] if mode == "adult" else config["output_dir_normal"]
        image_url = _upload_image_to_fal(image, output_dir, "rmbg_input")
        urls = fal.remove_background(image_url)
        if not urls:
            raise RuntimeError("背景除去結果が取得できませんでした")
        img = download_fal_image(urls[0])
        save_path = save_image_to_dir(img, output_dir, prefix=f"nobg_{mode}")
        return img, f"[Background Removal] 完了\n保存先: {save_path}"
    except Exception as e:
        logger.error(f"背景除去エラー: {traceback.format_exc()}")
        raise gr.Error(f"背景除去エラー: {e}")


def generate_upscale(image, scale, mode="normal"):
    """Upscale image via fal.ai."""
    if not fal.api_key:
        raise gr.Error("fal.ai API Key が未設定です。Settingsタブで設定してください。")
    if image is None:
        raise gr.Error("画像をアップロードしてください。")
    try:
        output_dir = config["output_dir_adult"] if mode == "adult" else config["output_dir_normal"]
        image_url = _upload_image_to_fal(image, output_dir, "upscale_input")
        urls = fal.upscale(image_url, int(scale))
        if not urls:
            raise RuntimeError("アップスケール結果が取得できませんでした")
        img = download_fal_image(urls[0])
        save_path = save_image_to_dir(img, output_dir, prefix=f"upscale_{scale}x_{mode}")
        return img, f"[Upscale {scale}x] 完了\n保存先: {save_path}"
    except Exception as e:
        logger.error(f"アップスケールエラー: {traceback.format_exc()}")
        raise gr.Error(f"アップスケールエラー: {e}")


def generate_controlnet(control_image, prompt, negative_prompt, control_type,
                        control_strength, width, height, steps, guidance, seed, mode="normal"):
    """Generate image guided by ControlNet via fal.ai."""
    if not fal.api_key:
        raise gr.Error("fal.ai API Key が未設定です。Settingsタブで設定してください。")
    if control_image is None:
        raise gr.Error("コントロール画像をアップロードしてください。")
    if not prompt.strip():
        raise gr.Error("プロンプトを入力してください。")
    try:
        output_dir = config["output_dir_adult"] if mode == "adult" else config["output_dir_normal"]
        control_url = _upload_image_to_fal(control_image, output_dir, "controlnet_input")
        ct = CONTROLNET_TYPES.get(control_type, "canny")
        urls = fal.controlnet(
            control_url, prompt, ct, float(control_strength),
            negative_prompt, int(width), int(height), int(steps), float(guidance), int(seed)
        )
        if not urls:
            raise RuntimeError("ControlNet結果が取得できませんでした")
        img = download_fal_image(urls[0])
        save_path = save_image_to_dir(img, output_dir, prefix=f"controlnet_{mode}")
        return img, f"[ControlNet: {control_type}] strength={control_strength}\n保存先: {save_path}"
    except Exception as e:
        logger.error(f"ControlNetエラー: {traceback.format_exc()}")
        raise gr.Error(f"ControlNetエラー: {e}")


def generate_vid2vid_cloud(video_file, prompt, model_key, strength, mode="normal"):
    """Video-to-video style transfer via cloud API."""
    if not fal.api_key:
        raise gr.Error("fal.ai API Key が未設定です。Settingsタブで設定してください。")
    if video_file is None:
        raise gr.Error("動画ファイルをアップロードしてください。")
    if not prompt.strip():
        raise gr.Error("プロンプトを入力してください。")
    try:
        import fal_client as fc
        os.environ["FAL_KEY"] = fal.api_key
        video_path = video_file if isinstance(video_file, str) else video_file.name
        video_url = fc.upload_file(video_path)

        result_url = fal.vid2vid(video_url, prompt, model_key, strength)
        if not result_url:
            raise RuntimeError("vid2vid結果が取得できませんでした")

        output_dir = config["output_dir_adult"] if mode == "adult" else config["output_dir_normal"]
        os.makedirs(output_dir, exist_ok=True)
        import datetime as dt
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(output_dir, f"vid2vid_cloud_{ts}.mp4")
        download_fal_video(result_url, out_path)

        cost = FAL_VID2VID_MODELS.get(model_key, {}).get("cost", "?")
        return out_path, f"[vid2vid Cloud: {model_key}] コスト: {cost}\n保存先: {out_path}"
    except Exception as e:
        logger.error(f"vid2vid Cloudエラー: {traceback.format_exc()}")
        raise gr.Error(f"vid2vid Cloudエラー: {e}")


def generate_long_video(prompt, model_key, num_clips, duration_per_clip,
                        negative_prompt="", mode="normal", progress=gr.Progress()):
    """Generate a long video by auto-chaining multiple clips via img2vid."""
    if not fal.api_key:
        raise gr.Error("fal.ai API Key が未設定です。Settingsタブで設定してください。")
    if not prompt.strip():
        raise gr.Error("プロンプトを入力してください。")

    num_clips = int(num_clips)
    dur = int(duration_per_clip)
    if num_clips < 2:
        raise gr.Error("2クリップ以上を指定してください。（1クリップなら通常の動画生成を使ってください）")
    if num_clips > 10:
        raise gr.Error("最大10クリップまでです。")

    output_dir = config["output_dir_adult"] if mode == "adult" else config["output_dir_normal"]
    os.makedirs(output_dir, exist_ok=True)
    clip_paths = []

    try:
        import fal_client as fc
        os.environ["FAL_KEY"] = fal.api_key
        import datetime as dt

        # Step 1: Generate first clip via txt2vid
        progress(0.0, desc=f"クリップ 1/{num_clips} を生成中...")
        first_url = fal.generate_video(model_key, prompt,
                                       negative_prompt=negative_prompt,
                                       duration=dur)
        if not first_url:
            raise RuntimeError("最初のクリップの生成に失敗しました")

        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        clip1_path = os.path.join(output_dir, f"longvid_clip01_{ts}.mp4")
        download_fal_video(first_url, clip1_path)
        clip_paths.append(clip1_path)

        # Steps 2..N: Extract last frame → img2vid for continuation
        for i in range(2, num_clips + 1):
            progress((i - 1) / num_clips, desc=f"クリップ {i}/{num_clips} を生成中...")

            # Extract last frame from previous clip
            last_frame_img, _ = extract_last_frame(clip_paths[-1])
            if last_frame_img is None:
                raise RuntimeError(f"クリップ {i-1} から最終フレーム抽出に失敗しました")

            # Save and upload last frame
            frame_path = os.path.join(output_dir, f"longvid_frame{i:02d}_{ts}.png")
            last_frame_img.save(frame_path)
            frame_url = fc.upload_file(frame_path)

            # Find matching img2vid model
            img2vid_model = _find_matching_img2vid_model(model_key)

            video_url = fal.img2vid(img2vid_model, frame_url,
                                    prompt=prompt, duration=dur)
            if not video_url:
                raise RuntimeError(f"クリップ {i} の生成に失敗しました")

            clip_path = os.path.join(output_dir, f"longvid_clip{i:02d}_{ts}.mp4")
            download_fal_video(video_url, clip_path)
            clip_paths.append(clip_path)

        # Step final: Concatenate all clips with ffmpeg
        progress(0.95, desc="クリップを結合中...")
        concat_path = os.path.join(output_dir, f"longvid_final_{ts}.mp4")
        concat_list_path = os.path.join(output_dir, f"longvid_list_{ts}.txt")

        with open(concat_list_path, "w") as f:
            for cp in clip_paths:
                f.write(f"file '{cp}'\n")

        result = subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", concat_list_path, "-c", "copy", concat_path],
            capture_output=True, timeout=60,
        )
        if result.returncode != 0:
            # Fallback: try re-encoding
            subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                 "-i", concat_list_path, "-c:v", "libx264", "-crf", "18",
                 "-preset", "fast", concat_path],
                capture_output=True, timeout=120,
            )

        # Cleanup temp list file
        try:
            os.remove(concat_list_path)
        except Exception:
            pass

        total_dur = num_clips * dur
        cost = FAL_VIDEO_MODELS.get(model_key, {}).get("cost", "?")
        info = (
            f"[Long Video] {num_clips}クリップ x {dur}秒 = 約{total_dur}秒\n"
            f"Model: {model_key} | コスト: {cost} x {num_clips}\n"
            f"結合動画: {concat_path}\n"
            f"個別クリップ: {', '.join(os.path.basename(p) for p in clip_paths)}"
        )
        progress(1.0, desc="完了!")
        return concat_path, info

    except Exception as e:
        logger.error(f"長尺動画生成エラー: {traceback.format_exc()}")
        # Return partial results if any clips were generated
        if clip_paths:
            return clip_paths[-1], f"エラー: {e}\n\n生成済みクリップ ({len(clip_paths)}本): {', '.join(clip_paths)}"
        raise gr.Error(f"長尺動画生成エラー: {e}")


def generate_extend_video(video_file, prompt, model_key, num_extensions,
                          duration_per_clip, negative_prompt="", mode="normal",
                          progress=gr.Progress()):
    """Extend an existing video by generating continuation clips from its last frame."""
    if not fal.api_key:
        raise gr.Error("fal.ai API Key が未設定です。Settingsタブで設定してください。")
    if video_file is None:
        raise gr.Error("元動画をアップロードしてください。")

    num_ext = int(num_extensions)
    dur = int(duration_per_clip)
    if num_ext < 1 or num_ext > 10:
        raise gr.Error("延長クリップ数は1-10の範囲で指定してください。")

    output_dir = config["output_dir_adult"] if mode == "adult" else config["output_dir_normal"]
    os.makedirs(output_dir, exist_ok=True)

    video_path = video_file if isinstance(video_file, str) else video_file.name
    clip_paths = [video_path]

    try:
        import fal_client as fc
        os.environ["FAL_KEY"] = fal.api_key
        import datetime as dt
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

        img2vid_model = _find_matching_img2vid_model(model_key)

        for i in range(1, num_ext + 1):
            progress((i - 1) / num_ext, desc=f"延長クリップ {i}/{num_ext} を生成中...")

            last_frame_img, _ = extract_last_frame(clip_paths[-1])
            if last_frame_img is None:
                raise RuntimeError(f"フレーム抽出失敗 (クリップ {len(clip_paths)})")

            frame_path = os.path.join(output_dir, f"extend_frame{i:02d}_{ts}.png")
            last_frame_img.save(frame_path)
            frame_url = fc.upload_file(frame_path)

            video_url = fal.img2vid(img2vid_model, frame_url,
                                    prompt=prompt or "smooth continuation, consistent style",
                                    duration=dur)
            if not video_url:
                raise RuntimeError(f"延長クリップ {i} の生成に失敗しました")

            clip_path = os.path.join(output_dir, f"extend_clip{i:02d}_{ts}.mp4")
            download_fal_video(video_url, clip_path)
            clip_paths.append(clip_path)

        # Concatenate
        progress(0.95, desc="クリップを結合中...")
        concat_path = os.path.join(output_dir, f"extended_final_{ts}.mp4")
        concat_list_path = os.path.join(output_dir, f"extend_list_{ts}.txt")

        with open(concat_list_path, "w") as f:
            for cp in clip_paths:
                f.write(f"file '{cp}'\n")

        result = subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", concat_list_path, "-c", "copy", concat_path],
            capture_output=True, timeout=60,
        )
        if result.returncode != 0:
            subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                 "-i", concat_list_path, "-c:v", "libx264", "-crf", "18",
                 "-preset", "fast", concat_path],
                capture_output=True, timeout=120,
            )

        try:
            os.remove(concat_list_path)
        except Exception:
            pass

        cost = FAL_IMG2VID_MODELS.get(img2vid_model, {}).get("cost", "?")
        info = (
            f"[Video Extend] {num_ext}クリップ追加 x {dur}秒\n"
            f"Model: {img2vid_model} | コスト: {cost} x {num_ext}\n"
            f"結合動画: {concat_path}"
        )
        progress(1.0, desc="完了!")
        return concat_path, info

    except Exception as e:
        logger.error(f"動画延長エラー: {traceback.format_exc()}")
        if len(clip_paths) > 1:
            return clip_paths[-1], f"エラー: {e}\n\n生成済みクリップ ({len(clip_paths)-1}本追加)"
        raise gr.Error(f"動画延長エラー: {e}")


def _find_matching_img2vid_model(txt2vid_key):
    """Find a matching img2vid model for a given txt2vid model."""
    key_lower = txt2vid_key.lower()
    if "wan 2.6" in key_lower:
        return "Wan 2.6 img2vid (最新・NSFW OK)"
    if "wan" in key_lower:
        return "Wan 2.1 img2vid (高品質・NSFW OK)"
    if "kling" in key_lower:
        return "Kling 2.5 Turbo Pro img2vid (高品質・SFW)"
    if "veo" in key_lower:
        return "Veo 3 img2vid (Google最高品質・SFW)"
    if "sora" in key_lower:
        return "Sora 2 img2vid (OpenAI・SFW)"
    if "ltx" in key_lower:
        return "LTX 2.3 img2vid (高速・安い)"
    # Default
    return list(FAL_IMG2VID_MODELS.keys())[0]


def generate_nsfw_with_fallback(prompt, preset_key, seed, auto_upscale=False):
    """Generate NSFW image with automatic fallback chain: fal.ai → Dezgo → Novita."""
    preset = NSFW_PRESETS.get(preset_key)
    if not preset:
        raise gr.Error(f"不明なプリセット: {preset_key}")

    full_prompt = preset["prompt_base"]
    if prompt.strip():
        full_prompt = full_prompt + ", " + prompt.strip()

    negative = preset["negative"]
    settings = preset["settings"]
    model_key = preset["recommended_models"][0]
    s = int(seed)

    # Try fal.ai first
    if fal.api_key:
        try:
            urls = fal.generate_image(
                model_key=model_key, prompt=full_prompt,
                negative_prompt=negative,
                width=settings["width"], height=settings["height"],
                steps=settings["steps"], guidance=settings["cfg"],
                seed=s, safety_checker=False,
            )
            images = []
            saved = []
            for url in urls:
                img = download_fal_image(url)
                images.append(img)
                path = save_image_to_dir(img, config["output_dir_adult"], prefix="nsfw_fal")
                saved.append(path)

            # Auto upscale if requested
            if auto_upscale and images:
                try:
                    upscaled_images = []
                    for img in images:
                        up_path = save_image_to_dir(img, config["output_dir_adult"], prefix="nsfw_pre_upscale")
                        up_url = _upload_image_to_fal(up_path, config["output_dir_adult"], "nsfw_upscale")
                        up_urls = fal.upscale(up_url, 2)
                        if up_urls:
                            up_img = download_fal_image(up_urls[0])
                            upscaled_images.append(up_img)
                            up_save = save_image_to_dir(up_img, config["output_dir_adult"], prefix="nsfw_hq")
                            saved.append(up_save)
                    if upscaled_images:
                        images = upscaled_images
                except Exception as ue:
                    logger.error(f"Auto upscale failed: {ue}")

            return images, f"[NSFW fal.ai: {preset_key}] Model: {model_key}\n保存先: {', '.join(saved)}"
        except Exception as e:
            if "content_policy_violation" not in str(e):
                raise gr.Error(f"fal.ai NSFW生成エラー: {e}")
            logger.info(f"fal.ai NSFWブロック → Dezgo/Novita にフォールバック")

    # Fallback to Dezgo (completely uncensored)
    if dezgo.api_key:
        try:
            png_bytes = dezgo.generate_image(
                "Flux Dev (高品質)", full_prompt,
                negative_prompt=negative,
                width=settings["width"], height=settings["height"],
                steps=settings.get("steps", 28), seed=s,
            )
            img = decode_dezgo_image(png_bytes)
            path = save_image_to_dir(img, config["output_dir_adult"], prefix="nsfw_dezgo")
            return [img], f"[NSFW Dezgo無検閲: {preset_key}]\n保存先: {path}"
        except Exception as e:
            logger.error(f"Dezgo NSFW fallback error: {e}")

    # Fallback to Novita (uncensored models)
    if novita.api_key:
        try:
            urls = novita.generate_image(
                "Realistic Vision (フォトリアル)", full_prompt,
                negative_prompt=negative,
                width=settings["width"], height=settings["height"],
                steps=settings.get("steps", 25), seed=s,
            )
            images = []
            saved = []
            for url in urls:
                img = download_novita_image(url)
                images.append(img)
                path = save_image_to_dir(img, config["output_dir_adult"], prefix="nsfw_novita")
                saved.append(path)
            return images, f"[NSFW Novita無検閲: {preset_key}]\n保存先: {', '.join(saved)}"
        except Exception as e:
            logger.error(f"Novita NSFW fallback error: {e}")

    raise gr.Error(
        "NSFW生成に対応するAPIキーが設定されていません。\n\n"
        "以下のいずれかを設定してください:\n"
        "1. fal.ai (推奨・高品質)\n"
        "2. Dezgo (完全無検閲)\n"
        "3. Novita.ai (無検閲・安い)"
    )


def generate_nsfw_video_preset(prompt, preset_key, from_image=None):
    """Generate NSFW video using preset with auto-fallback."""
    preset = NSFW_VIDEO_PRESETS.get(preset_key)
    if not preset:
        raise gr.Error(f"不明な動画プリセット: {preset_key}")

    full_prompt = preset["prompt_base"]
    if prompt.strip():
        full_prompt = full_prompt + ", " + prompt.strip()

    neg = preset.get("negative", "")
    dur = preset.get("duration", 5)

    # img2vid if image provided
    if from_image is not None:
        model_key = preset.get("img2vid_model", "Wan 2.6 img2vid (最新・NSFW OK)")
        return generate_fal_img2vid(from_image, full_prompt, model_key, "adult", dur)

    # txt2vid
    model_key = preset.get("model", "Wan 2.6 txt2vid (最新・NSFW OK)")
    video_path, info = generate_fal_video(full_prompt, model_key, "adult",
                                          duration=dur, negative_prompt=neg)
    return _gradio_safe_video(video_path), info


def generate_hq_image(prompt, negative_prompt, model_key, width, height,
                      steps, guidance, seed, batch_size, auto_upscale, mode="normal"):
    """Generate highest quality image with optional auto-upscale."""
    if not fal.api_key:
        raise gr.Error("fal.ai API Key が未設定です。")
    if not prompt.strip():
        raise gr.Error("プロンプトを入力してください。")

    try:
        w = max(256, (int(width) // 32) * 32)
        h = max(256, (int(height) // 32) * 32)
        urls = fal.generate_image(
            model_key=model_key, prompt=prompt,
            negative_prompt=negative_prompt,
            width=w, height=h,
            steps=int(steps), guidance=float(guidance),
            seed=int(seed), num_images=int(batch_size),
            safety_checker=False,
        )

        images = []
        saved = []
        output_dir = config["output_dir_adult"] if mode == "adult" else config["output_dir_normal"]
        for url in urls:
            img = download_fal_image(url)
            images.append(img)
            path = save_image_to_dir(img, output_dir, prefix=f"hq_{mode}")
            saved.append(path)

        # Auto upscale
        if auto_upscale and images:
            upscaled = []
            for img in images:
                try:
                    tmp_path = save_image_to_dir(img, output_dir, prefix="hq_pre_up")
                    up_url = _upload_image_to_fal(tmp_path, output_dir, "hq_upscale")
                    up_urls = fal.upscale(up_url, 2)
                    if up_urls:
                        up_img = download_fal_image(up_urls[0])
                        upscaled.append(up_img)
                        up_save = save_image_to_dir(up_img, output_dir, prefix=f"hq_upscaled_{mode}")
                        saved.append(up_save)
                except Exception as ue:
                    logger.error(f"HQ auto upscale failed: {ue}")
                    upscaled.append(img)
            if upscaled:
                images = upscaled

        cost = FAL_MODELS.get(model_key, {}).get("cost", "?")
        up_info = " + 2x Upscale" if auto_upscale else ""
        return images, f"[HQ {model_key}]{up_info}\n{w}x{h} → {'{}x{}'.format(w*2, h*2) if auto_upscale else f'{w}x{h}'}\nコスト: {cost}\n保存先: {', '.join(saved)}"
    except Exception as e:
        logger.error(f"HQ生成エラー: {traceback.format_exc()}")
        raise gr.Error(f"HQ生成エラー: {e}")


def generate_with_art_preset(prompt, preset_key, seed, mode="normal"):
    """Generate image using artistic preset settings."""
    if not fal.api_key:
        raise gr.Error("fal.ai API Key が未設定です。Settingsタブで設定してください。")
    if not prompt.strip():
        raise gr.Error("プロンプトを入力してください。")

    preset = ART_PRESETS.get(preset_key)
    if not preset:
        raise gr.Error(f"不明なプリセット: {preset_key}")

    full_prompt = prompt + preset["prompt_suffix"]
    model_key = preset.get("model", "Flux Dev (高品質・NSFW OK)")

    try:
        urls = fal.generate_image(
            model_key=model_key,
            prompt=full_prompt,
            negative_prompt=preset.get("negative", ""),
            width=preset.get("width", 1024),
            height=preset.get("height", 1024),
            steps=preset.get("steps", 28),
            guidance=preset.get("cfg", 3.5),
            seed=int(seed),
            safety_checker=False,
        )
        images = []
        saved_paths = []
        output_dir = config["output_dir_adult"] if mode == "adult" else config["output_dir_normal"]
        for url in urls:
            img = download_fal_image(url)
            images.append(img)
            path = save_image_to_dir(img, output_dir, prefix=f"art_{mode}")
            saved_paths.append(path)
        return images, f"[Art Preset: {preset_key}]\nModel: {model_key}\n保存先: {', '.join(saved_paths)}"
    except Exception as e:
        logger.error(f"Art Preset生成エラー: {traceback.format_exc()}")
        raise gr.Error(f"Art Preset生成エラー: {e}")


def generate_face_swap(base_image, swap_image, mode="normal"):
    """Face swap using fal.ai."""
    if not fal.api_key:
        raise gr.Error("fal.ai API Key が未設定です。Settingsタブで設定してください。")
    if base_image is None:
        raise gr.Error("ベース画像（体・シーン）をアップロードしてください。")
    if swap_image is None:
        raise gr.Error("顔画像（入れ替える顔）をアップロードしてください。")

    try:
        from PIL import Image as PILImage
        import base64
        output_dir = config["output_dir_adult"] if mode == "adult" else config["output_dir_normal"]
        os.makedirs(output_dir, exist_ok=True)

        # Upload images to fal storage
        import fal_client as fc
        os.environ["FAL_KEY"] = fal.api_key

        # Save temp files and upload
        ts = int(time.time())
        base_path = os.path.join(output_dir, f"faceswap_base_{ts}.png")
        swap_path = os.path.join(output_dir, f"faceswap_face_{ts}.png")

        if isinstance(base_image, str):
            base_path = base_image
        else:
            PILImage.fromarray(base_image).save(base_path)

        if isinstance(swap_image, str):
            swap_path = swap_image
        else:
            PILImage.fromarray(swap_image).save(swap_path)

        base_url = fc.upload_file(base_path)
        swap_url = fc.upload_file(swap_path)

        result_url = fal.face_swap(base_url, swap_url)
        if not result_url:
            raise RuntimeError("Face Swap結果のURLが取得できませんでした")

        result_img = download_fal_image(result_url)
        save_path = save_image_to_dir(result_img, output_dir, prefix=f"faceswap_{mode}")
        return result_img, f"[Face Swap] 完了!\n保存先: {save_path}"
    except Exception as e:
        logger.error(f"Face Swapエラー: {traceback.format_exc()}")
        raise gr.Error(f"Face Swapエラー: {e}")


def generate_fal_image_direct(prompt, model_key, width, height, seed, num_images, mode="normal"):
    """Generate image via fal.ai API with model selection."""
    if not fal.api_key:
        raise gr.Error("fal.ai API Key が未設定です。Settingsタブで設定してください。")
    if not prompt.strip():
        raise gr.Error("プロンプトを入力してください。")
    try:
        w, h = int(width), int(height)
        w = max(256, (w // 32) * 32)
        h = max(256, (h // 32) * 32)
        urls = fal.generate_image(
            model_key=model_key, prompt=prompt,
            width=w, height=h, num_images=int(num_images),
            seed=int(seed), safety_checker=False,
        )
        images = []
        saved_paths = []
        output_dir = config["output_dir_adult"] if mode == "adult" else config["output_dir_normal"]
        for url in urls:
            img = download_fal_image(url)
            images.append(img)
            path = save_image_to_dir(img, output_dir, prefix=f"fal_{mode}")
            saved_paths.append(path)
        cost = FAL_MODELS.get(model_key, {}).get("cost", "?")
        return images, f"[fal.ai {model_key}] コスト: {cost}\n保存先: {', '.join(saved_paths)}"
    except Exception as e:
        logger.error(f"fal.ai生成エラー: {traceback.format_exc()}")
        raise gr.Error(f"fal.ai生成エラー: {e}")


def generate_fal_img2vid(image, prompt, model_key, mode="normal", duration=None):
    """Generate video from image via fal.ai."""
    if not fal.api_key:
        raise gr.Error("fal.ai API Key が未設定です。Settingsタブで設定してください。")
    if image is None:
        raise gr.Error("画像をアップロードしてください。")

    try:
        import fal_client as fc
        os.environ["FAL_KEY"] = fal.api_key
        from PIL import Image as PILImage

        output_dir = config["output_dir_adult"] if mode == "adult" else config["output_dir_normal"]
        os.makedirs(output_dir, exist_ok=True)

        # Save and upload image
        ts = int(time.time())
        if isinstance(image, str):
            img_path = image
        else:
            img_path = os.path.join(output_dir, f"img2vid_input_{ts}.png")
            PILImage.fromarray(image).save(img_path)

        image_url = fc.upload_file(img_path)

        dur = int(duration) if duration else None
        video_url = fal.img2vid(model_key, image_url, prompt=prompt or "", duration=dur)
        if not video_url:
            raise RuntimeError("動画URLが取得できませんでした")

        video_path = os.path.join(output_dir, f"fal_img2vid_{ts}.mp4")
        download_fal_video(video_url, video_path)

        model_info = FAL_IMG2VID_MODELS.get(model_key, {})
        cost = model_info.get("cost", "?")
        dur_info = f" | {dur}秒" if dur else ""
        return _gradio_safe_video(video_path), f"[fal.ai img2vid {model_key}] コスト: {cost}{dur_info}\n保存先: {video_path}"
    except Exception as e:
        error_str = str(e)
        logger.error(f"fal.ai img2vidエラー: {traceback.format_exc()}")
        if "content_policy_violation" in error_str:
            raise gr.Error(
                "fal.aiがNSFWコンテンツをブロックしました。\n\n"
                "img2vidのNSFW → ローカル AnimateDiff (img2vidタブ) を使ってください。"
            )
        raise gr.Error(f"fal.ai img2vidエラー: {e}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Image to Video Pro — VLM-assisted (Preserve & Inspired modes)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from vision_analyzer import (
    analyze_for_motion,
    describe_for_inspiration,
    MOTION_PRESETS,
)


def _get_vision_keys():
    """Return (openai_key, anthropic_key) tuple from config."""
    return (
        config.get("openai_api_key", ""),
        config.get("anthropic_api_key", ""),
    )


def img2vid_analyze_motion(image):
    """Auto-analyze image for natural motion prompt (Preserve mode helper).

    Called by the 'Auto-analyze' button. Returns analyzed motion prompt.
    """
    if image is None:
        raise gr.Error("画像をアップロードしてください。")
    openai_key, anthropic_key = _get_vision_keys()
    if not openai_key and not anthropic_key:
        raise gr.Error(
            "OpenAI または Anthropic API Key が必要です。\n"
            "Settings → OpenAI API Key または Anthropic API Key を設定してください。"
        )
    try:
        motion_prompt = analyze_for_motion(image, openai_key, anthropic_key)
        return motion_prompt
    except Exception as e:
        logger.error(f"Image analysis (motion) failed: {traceback.format_exc()}")
        raise gr.Error(f"画像分析エラー: {e}")


def img2vid_analyze_inspiration(image):
    """Auto-analyze image for full scene description (Inspired mode helper).

    Called by the 'Auto-analyze' button. Returns scene description.
    """
    if image is None:
        raise gr.Error("画像をアップロードしてください。")
    openai_key, anthropic_key = _get_vision_keys()
    if not openai_key and not anthropic_key:
        raise gr.Error(
            "OpenAI または Anthropic API Key が必要です。\n"
            "Settings → OpenAI API Key または Anthropic API Key を設定してください。"
        )
    try:
        description = describe_for_inspiration(image, openai_key, anthropic_key)
        return description
    except Exception as e:
        logger.error(f"Image analysis (inspiration) failed: {traceback.format_exc()}")
        raise gr.Error(f"画像分析エラー: {e}")


def generate_img2vid_preserve(image, motion_prompt, motion_preset, fal_model_key,
                              auto_analyze, duration, mode="normal"):
    """Preserve mode: animate the image while keeping content exact.

    Workflow:
    1. If auto_analyze=True and motion_prompt is empty → run VLM analysis
    2. Append motion preset suffix
    3. Send to fal.ai img2vid model (Kling / Wan / Veo preserve the image)

    Args:
        image: numpy array or PIL image from Gradio
        motion_prompt: User-provided or auto-analyzed motion description
        motion_preset: Key from MOTION_PRESETS (appended to prompt)
        fal_model_key: Key into FAL_IMG2VID_MODELS
        auto_analyze: If True and prompt empty, run VLM
        duration: seconds
        mode: "normal" or "adult" (output folder)
    """
    if image is None:
        raise gr.Error("画像をアップロードしてください。")

    final_prompt = (motion_prompt or "").strip()

    # Auto-analyze if requested and prompt empty
    if auto_analyze and not final_prompt:
        try:
            openai_key, anthropic_key = _get_vision_keys()
            if openai_key or anthropic_key:
                final_prompt = analyze_for_motion(image, openai_key, anthropic_key)
                logger.info(f"[Preserve] auto motion: {final_prompt[:100]}")
        except Exception as e:
            logger.error(f"Auto-analyze failed, using empty prompt: {e}")

    # Append motion preset
    if motion_preset and motion_preset in MOTION_PRESETS and MOTION_PRESETS[motion_preset]:
        final_prompt = (final_prompt + MOTION_PRESETS[motion_preset]).strip(", ")

    if not final_prompt:
        final_prompt = "subtle natural motion, gentle breeze, cinematic"

    # Delegate to existing fal.ai img2vid pipeline
    video_out, info = generate_fal_img2vid(image, final_prompt, fal_model_key, mode, duration)
    full_info = (
        f"[Preserve Mode / {fal_model_key}]\n"
        f"Prompt: {final_prompt[:200]}\n"
        f"{info}"
    )
    return video_out, full_info


def generate_img2vid_inspired(image, description, style_hint, fal_t2v_model, duration, mode="normal"):
    """Inspired mode: analyze image, then generate a NEW video from the description.

    Workflow:
    1. If description is empty → analyze image with VLM
    2. Append style hint (e.g., "anime style", "cinematic", "dreamy")
    3. Send description to fal.ai txt2vid (no image conditioning)

    Args:
        image: numpy array or PIL (only for analysis, not sent to txt2vid)
        description: User-provided or auto-analyzed scene description
        style_hint: Optional style modifier
        fal_t2v_model: Key into FAL_VIDEO_MODELS (txt2vid)
        duration: seconds
        mode: "normal" or "adult"
    """
    if image is None and not description:
        raise gr.Error("画像または説明文のいずれかが必要です。")

    final_desc = (description or "").strip()

    # Auto-analyze if no description provided
    if not final_desc:
        try:
            openai_key, anthropic_key = _get_vision_keys()
            if not openai_key and not anthropic_key:
                raise gr.Error(
                    "OpenAI または Anthropic API Key が必要です (画像分析のため)。\n"
                    "Settingsタブで設定するか、説明文を手動入力してください。"
                )
            final_desc = describe_for_inspiration(image, openai_key, anthropic_key)
            logger.info(f"[Inspired] auto description: {final_desc[:100]}")
        except gr.Error:
            raise
        except Exception as e:
            logger.error(f"Inspired analyze failed: {traceback.format_exc()}")
            raise gr.Error(f"画像分析エラー: {e}")

    # Apply style hint
    if style_hint and style_hint.strip():
        final_desc = f"{final_desc}, {style_hint.strip()}"

    # Delegate to existing fal.ai txt2vid pipeline
    video_out, info = generate_fal_video(
        prompt=final_desc,
        model_key=fal_t2v_model,
        mode=mode,
        duration=int(duration) if duration else None,
    )
    full_info = (
        f"[Inspired Mode / {fal_t2v_model}]\n"
        f"Description: {final_desc[:300]}\n"
        f"{info}"
    )
    return video_out, full_info


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Wan 2.2 NSFW Lightning (dedicated high-VRAM backend)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _get_wan22_client():
    """Get ComfyUI client for Wan 2.2 dedicated backend.

    Priority: wan22_comfyui_url → vast_comfyui_url → main comfyui_url.
    This allows routing Wan 2.2 to a dedicated high-VRAM instance while
    keeping the regular image generation on a smaller backend.
    """
    url = (
        config.get("wan22_comfyui_url")
        or config.get("vast_comfyui_url")
        or config.get("comfyui_url")
    )
    return ComfyUIClient(url)


def generate_wan22_t2v(prompt, negative_prompt, width, height, duration,
                      steps, cfg, seed, mode="normal"):
    """Generate text-to-video using Wan 2.2 NSFW Lightning on dedicated backend.

    Requires:
    - ComfyUI-GGUF custom node installed
    - wan22_nsfw_fm_v2_{high,low}_Q4_K_M.gguf in models/unet/ (or wherever GGUF looks)
    - umt5_xxl_fp8_e4m3fn_scaled.safetensors in models/text_encoders/
    - wan_2.1_vae.safetensors in models/vae/
    """
    if not prompt or not prompt.strip():
        raise gr.Error("プロンプトを入力してください。")

    wan_client = _get_wan22_client()
    if not wan_client.is_server_running():
        raise gr.Error(
            f"Wan 2.2 ComfyUI に接続できません ({wan_client.server_url})\n"
            "Settingsタブで wan22_comfyui_url を設定してください。"
        )

    # 81 frames @ 16fps = 5 sec default. Compute from duration.
    fps = WAN22_DEFAULTS["fps"]
    frame_count = int(float(duration) * fps) if duration else WAN22_DEFAULTS["frame_count"]
    # Wan requires frame_count = 4N+1 (81, 65, 49, 33, ...)
    frame_count = ((frame_count - 1) // 4) * 4 + 1
    frame_count = max(17, min(frame_count, 241))  # 1-15 sec clamp (@16fps)

    workflow = build_wan22_t2v_workflow(
        prompt=prompt,
        negative_prompt=negative_prompt or "worst quality, blurry, static",
        width=int(width) if width else WAN22_DEFAULTS["width"],
        height=int(height) if height else WAN22_DEFAULTS["height"],
        frame_count=frame_count,
        fps=fps,
        steps=int(steps) if steps else WAN22_DEFAULTS["steps"],
        cfg=float(cfg) if cfg else WAN22_DEFAULTS["cfg"],
        seed=int(seed) if seed is not None else -1,
    )

    output_dir = config["output_dir_adult"] if mode == "adult" else config["output_dir_normal"]
    os.makedirs(output_dir, exist_ok=True)

    try:
        frames, video_path = wan_client.generate_video(workflow, output_dir, timeout=1800)
        if not video_path:
            raise RuntimeError("動画ファイルが生成されませんでした")
        info = (
            f"[Wan 2.2 NSFW Lightning T2V]\n"
            f"Prompt: {prompt[:150]}\n"
            f"Frames: {frame_count} @ {fps}fps = {frame_count/fps:.1f}秒\n"
            f"保存先: {video_path}"
        )
        return _gradio_safe_video(video_path), info
    except Exception as e:
        logger.error(f"Wan 2.2 T2V error: {traceback.format_exc()}")
        raise gr.Error(f"Wan 2.2 T2V エラー: {e}")


def generate_wan22_i2v(image, prompt, negative_prompt, width, height, duration,
                      steps, cfg, seed, mode="normal"):
    """Generate image-to-video using Wan 2.2 NSFW Lightning I2V on dedicated backend."""
    if image is None:
        raise gr.Error("画像をアップロードしてください。")

    wan_client = _get_wan22_client()
    if not wan_client.is_server_running():
        raise gr.Error(
            f"Wan 2.2 ComfyUI に接続できません ({wan_client.server_url})\n"
            "Settingsタブで wan22_comfyui_url を設定してください。"
        )

    # Save image and upload to ComfyUI
    from PIL import Image as PILImage
    ts = int(time.time())
    tmp_path = os.path.join("/tmp", f"wan22_i2v_input_{ts}.png")
    if isinstance(image, str):
        PILImage.open(image).save(tmp_path)
    else:
        PILImage.fromarray(image).save(tmp_path)

    upload_result = wan_client.upload_image(tmp_path)
    image_name = upload_result.get("name") if isinstance(upload_result, dict) else os.path.basename(tmp_path)

    fps = WAN22_DEFAULTS["fps"]
    frame_count = int(float(duration) * fps) if duration else WAN22_DEFAULTS["frame_count"]
    frame_count = ((frame_count - 1) // 4) * 4 + 1
    frame_count = max(17, min(frame_count, 241))

    workflow = build_wan22_i2v_workflow(
        prompt=prompt or "",
        image_name=image_name,
        negative_prompt=negative_prompt or "worst quality, blurry, static, distorted",
        width=int(width) if width else WAN22_DEFAULTS["width"],
        height=int(height) if height else WAN22_DEFAULTS["height"],
        frame_count=frame_count,
        fps=fps,
        steps=int(steps) if steps else WAN22_DEFAULTS["steps"],
        cfg=float(cfg) if cfg else WAN22_DEFAULTS["cfg"],
        seed=int(seed) if seed is not None else -1,
    )

    output_dir = config["output_dir_adult"] if mode == "adult" else config["output_dir_normal"]
    os.makedirs(output_dir, exist_ok=True)

    try:
        frames, video_path = wan_client.generate_video(workflow, output_dir, timeout=1800)
        if not video_path:
            raise RuntimeError("動画ファイルが生成されませんでした")
        info = (
            f"[Wan 2.2 NSFW Lightning I2V]\n"
            f"Motion Prompt: {prompt[:150] if prompt else '(なし)'}\n"
            f"Frames: {frame_count} @ {fps}fps = {frame_count/fps:.1f}秒\n"
            f"保存先: {video_path}"
        )
        return _gradio_safe_video(video_path), info
    except Exception as e:
        logger.error(f"Wan 2.2 I2V error: {traceback.format_exc()}")
        raise gr.Error(f"Wan 2.2 I2V エラー: {e}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Adult Studio Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _adult_generate_image(prompt, negative, w=832, h=1216, seed=-1, prefix="adult", civitai_model=None,
                          model_override=None, lora_override=None, lora_str=0.8, vae_override=None,
                          steps_override=None, cfg_override=None, sampler_override=None, scheduler_override=None,
                          hires_fix=False, hires_scale=1.5, hires_denoise=0.5, hires_steps=15, upscale_model="",
                          face_detailer=False, face_denoise=0.4, face_guide_size=512,
                          lora2_name=None, lora2_str=0.8, lora3_name=None, lora3_str=0.8):
    """Generate NSFW image using the best available uncensored backend.

    Priority: ComfyUI → Novita → Dezgo → CivitAI → fal.ai (last resort).
    If model_override is specified, ComfyUI uses that model directly.
    If civitai_model is specified, CivitAI is tried first.
    """
    output_dir = config["output_dir_adult"]
    os.makedirs(output_dir, exist_ok=True)
    backend = config.get("backend", "local")
    s = int(seed) if seed is not None else -1
    errors = []

    # Build LoRA list for multi-LoRA support
    loras_list = []
    if lora_override and lora_override not in ("None", "なし", ""):
        loras_list.append((lora_override, float(lora_str)))
    if lora2_name and lora2_name not in ("None", "なし", ""):
        loras_list.append((lora2_name, float(lora2_str)))
    if lora3_name and lora3_name not in ("None", "なし", ""):
        loras_list.append((lora3_name, float(lora3_str)))

    # Resolve VAE
    vae_name = vae_override if vae_override and vae_override not in ("None", "Auto", "") else ""

    # ── 0. CivitAI — 指定モデルがあれば最優先 ──
    if civitai_model and civitai.api_key:
        try:
            # iCloudモデルの場合はURN解決してクラウド生成
            if civitai_model.startswith("[iCloud] "):
                icloud_filename = civitai_model[len("[iCloud] "):]
                urn_info = civitai.resolve_icloud_model_urn(icloud_filename)
                if urn_info and urn_info.get("type", "").lower() != "lora":
                    urls = civitai.generate_image_by_urn(
                        model_urn=urn_info["urn"], base_model=urn_info["base"],
                        prompt=prompt, negative_prompt=negative, width=w, height=h)
                    if urls:
                        images = []
                        saved = []
                        for url in (urls if isinstance(urls, list) else [urls]):
                            img = download_fal_image(url) if isinstance(url, str) and url.startswith("http") else url
                            if img:
                                images.append(img)
                                saved.append(save_image_to_dir(img, output_dir, prefix=f"{prefix}_civitai_icloud"))
                        if images:
                            model_label = f"{urn_info['name']} ({urn_info.get('version', '')})"
                            return images, f"[CivitAI iCloud] {model_label}\nPrompt: {prompt[:150]}...\n保存先: {', '.join(saved)}"
            else:
                model_info = CIVITAI_GENERATION_MODELS.get(civitai_model)
                if model_info:
                    urls = civitai.generate_image(civitai_model, prompt, negative_prompt=negative,
                                                  width=w, height=h)
                    if urls:
                        images = []
                        saved = []
                        for url in (urls if isinstance(urls, list) else [urls]):
                            img = download_fal_image(url) if isinstance(url, str) and url.startswith("http") else url
                            if img:
                                images.append(img)
                                saved.append(save_image_to_dir(img, output_dir, prefix=f"{prefix}_civitai"))
                        if images:
                            return images, f"[CivitAI 指定モデル] {civitai_model}\nPrompt: {prompt[:150]}...\n保存先: {', '.join(saved)}"
        except Exception as e:
            errors.append(f"CivitAI ({civitai_model}): {e}")

    # ── 1. ComfyUI (Vast.ai / RunPod / Local) — 完全無検閲 ──
    if client.is_server_running():
        try:
            models = client.get_models()
            if not models:
                models = _get_models_for_backend()

            # Use model_override if specified, otherwise auto-select
            if model_override and model_override not in ("(自動)", ""):
                chosen = model_override
            else:
                chosen = _select_nsfw_model(models, prompt)

            # Determine generation parameters (override > default)
            gen_steps = int(steps_override) if steps_override else 28
            gen_cfg = float(cfg_override) if cfg_override else 7.5
            gen_sampler = sampler_override if sampler_override else "euler_ancestral"
            gen_scheduler = scheduler_override if scheduler_override else "normal"

            if models or model_override:
                workflow = build_txt2img_workflow(
                    prompt, negative, chosen, w, h, gen_steps, gen_cfg,
                    gen_sampler, gen_scheduler, s, 1,
                    loras=loras_list,
                    vae_name=vae_name,
                    hires_fix=hires_fix, hires_scale=hires_scale,
                    hires_denoise=hires_denoise, hires_steps=hires_steps,
                    upscale_model=upscale_model,
                    face_detailer=face_detailer, face_denoise=face_denoise,
                    face_guide_size=face_guide_size,
                )
                result = client.generate(workflow)
                if result:
                    saved = []
                    for img in result:
                        saved.append(save_image_to_dir(img, output_dir, prefix=prefix))
                    lora_info = f"\nLoRA: {', '.join(f'{n} ({s})' for n, s in loras_list)}" if loras_list else ""
                    return result, f"[ComfyUI 無検閲] Model: {chosen}{lora_info}\nPrompt: {prompt[:150]}...\n保存先: {', '.join(saved)}"
        except Exception as e:
            errors.append(f"ComfyUI: {e}")

    # ── 2. Novita.ai — 無検閲モデル ──
    if novita.api_key:
        try:
            model_key = "Realistic Vision (フォトリアル)" if "anime" not in prompt.lower() else "MeinaMix (アニメ)"
            results = novita.generate_image(model_key, prompt, negative_prompt=negative,
                                            width=min(w, 1024), height=min(h, 1024), seed=s)
            if results:
                images = []
                saved = []
                for img_b64 in results:
                    img = download_novita_image(img_b64) if isinstance(img_b64, str) and img_b64.startswith("http") else None
                    if img is None and isinstance(img_b64, str):
                        import base64
                        img = Image.open(io.BytesIO(base64.b64decode(img_b64)))
                    if img:
                        images.append(img)
                        saved.append(save_image_to_dir(img, output_dir, prefix=f"{prefix}_novita"))
                if images:
                    return images, f"[Novita 無検閲] Prompt: {prompt[:150]}...\n保存先: {', '.join(saved)}"
        except Exception as e:
            errors.append(f"Novita: {e}")

    # ── 3. Dezgo — 完全無検閲 ──
    if dezgo.api_key:
        try:
            img_bytes = dezgo.generate_image("Flux Dev (高品質)", prompt, negative_prompt=negative,
                                             width=w, height=h, seed=s)
            if img_bytes:
                img = decode_dezgo_image(img_bytes)
                if img:
                    path = save_image_to_dir(img, output_dir, prefix=f"{prefix}_dezgo")
                    return [img], f"[Dezgo 無検閲] Prompt: {prompt[:150]}...\n保存先: {path}"
        except Exception as e:
            errors.append(f"Dezgo: {e}")

    # ── 4. CivitAI Generation API ──
    if civitai.api_key:
        try:
            civitai_models = list(CIVITAI_GENERATION_MODELS.keys())
            if civitai_models:
                result = civitai.generate_image(civitai_models[0], prompt, negative_prompt=negative,
                                                width=w, height=h)
                if result:
                    images = []
                    saved = []
                    for img in (result if isinstance(result, list) else [result]):
                        if img:
                            images.append(img)
                            saved.append(save_image_to_dir(img, output_dir, prefix=f"{prefix}_civitai"))
                    if images:
                        return images, f"[CivitAI] Prompt: {prompt[:150]}...\n保存先: {', '.join(saved)}"
        except Exception as e:
            errors.append(f"CivitAI: {e}")

    # ── 5. fal.ai — 最終手段 (NSFWブロックの可能性あり) ──
    if fal.api_key:
        try:
            model_key = "Flux Dev (高品質・NSFW OK)"
            urls = fal.generate_image(model_key, prompt, negative_prompt=negative,
                                      width=w, height=h, seed=s)
            if urls:
                images = []
                saved = []
                for url in urls:
                    img = download_fal_image(url)
                    if img:
                        images.append(img)
                        saved.append(save_image_to_dir(img, output_dir, prefix=f"{prefix}_fal"))
                if images:
                    return images, f"[fal.ai] ⚠黒画像ならNovita/Dezgo/Vast.aiを使ってください\nPrompt: {prompt[:150]}...\n保存先: {', '.join(saved)}"
        except Exception as e:
            errors.append(f"fal.ai: {e}")

    error_detail = "\n".join(errors) if errors else "APIキーが未設定です"
    raise gr.Error(
        f"全バックエンドで生成失敗:\n{error_detail}\n\n"
        f"推奨: Novita.ai / Dezgo / Vast.ai ComfyUI のAPIキーを設定してください。"
    )


def generate_character_image(style_key, ethnicity, age, body_type, breast, butt,
                             hair_color, hair_style, skin, expression, clothing,
                             pose, position, camera, setting, people_count,
                             custom_prompt, seed, civitai_model="(自動)",
                             model_sel="(自動)", lora1="None", lora1_str=0.8,
                             lora2="None", lora2_str=0.8, lora3="None", lora3_str=0.8,
                             vae_sel="None",
                             steps=None, cfg=None, sampler=None, scheduler=None,
                             width=832, height=1216,
                             hires_fix=False, hires_scale=1.5, hires_denoise=0.5, hires_steps=15, upscale_model="None",
                             face_detailer=False, face_denoise=0.4, face_guide_size=512):
    """Generate image from character builder selections."""
    prompt, negative = compose_character_prompt(
        style_key, ethnicity, age, body_type, breast, butt, hair_color,
        hair_style, skin, expression, clothing, pose, position, camera,
        setting, people_count, custom_prompt
    )
    if not prompt.strip():
        raise gr.Error("少なくとも1つの項目を選択してください。")
    model = civitai_model if civitai_model and civitai_model != "(自動)" else None
    model_ov = model_sel if model_sel and model_sel != "(自動)" else None
    return _adult_generate_image(
        prompt, negative, w=int(width), h=int(height), seed=seed, prefix="char_builder",
        civitai_model=model, model_override=model_ov,
        lora_override=lora1, lora_str=lora1_str,
        lora2_name=lora2, lora2_str=lora2_str,
        lora3_name=lora3, lora3_str=lora3_str,
        vae_override=vae_sel,
        steps_override=steps, cfg_override=cfg, sampler_override=sampler, scheduler_override=scheduler,
        hires_fix=hires_fix, hires_scale=hires_scale, hires_denoise=hires_denoise,
        hires_steps=hires_steps, upscale_model=upscale_model if upscale_model != "None" else "",
        face_detailer=face_detailer, face_denoise=face_denoise, face_guide_size=face_guide_size,
    )


def generate_scene_category(category_key, custom_addition, seed, civitai_model="(自動)",
                            model_sel="(自動)", lora1="None", lora1_str=0.8,
                            lora2="None", lora2_str=0.8, lora3="None", lora3_str=0.8,
                            vae_sel="None",
                            steps=None, cfg=None, sampler=None, scheduler=None,
                            width=832, height=1216,
                            hires_fix=False, hires_scale=1.5, hires_denoise=0.5, hires_steps=15, upscale_model="None",
                            face_detailer=False, face_denoise=0.4, face_guide_size=512):
    """Generate image from a scene category preset."""
    prompt, negative = compose_scene_prompt(category_key, custom_addition)
    if not prompt.strip():
        raise gr.Error("カテゴリを選択してください。")
    model = civitai_model if civitai_model and civitai_model != "(自動)" else None
    model_ov = model_sel if model_sel and model_sel != "(自動)" else None
    return _adult_generate_image(
        prompt, negative, w=int(width), h=int(height), seed=seed, prefix="scene",
        civitai_model=model, model_override=model_ov,
        lora_override=lora1, lora_str=lora1_str,
        lora2_name=lora2, lora2_str=lora2_str,
        lora3_name=lora3, lora3_str=lora3_str,
        vae_override=vae_sel,
        steps_override=steps, cfg_override=cfg, sampler_override=sampler, scheduler_override=scheduler,
        hires_fix=hires_fix, hires_scale=hires_scale, hires_denoise=hires_denoise,
        hires_steps=hires_steps, upscale_model=upscale_model if upscale_model != "None" else "",
        face_detailer=face_detailer, face_denoise=face_denoise, face_guide_size=face_guide_size,
    )


def _select_nsfw_model(models, prompt_hint=""):
    """Select best NSFW-capable model from available models.

    Preference order (by actual model strength for NSFW):
    - Anime/hentai hint → cyberrealisticPony (Pony NSFW)
    - Asian/photo hint → chilloutmix (Asian NSFW SD1.5)
    - SDXL / high-res → realvisxlV50
    - Default photo NSFW → epicphotogasm (uncensored photo SD1.5)
    """
    if not models:
        return "realvisxlV50.safetensors"

    hint = (prompt_hint or "").lower()
    is_anime = any(k in hint for k in ["anime", "hentai", "manga", "アニメ", "pony"])
    is_asian = any(k in hint for k in ["asian", "japanese", "korean", "chinese", "日本人", "アジア"])

    if is_anime:
        priority = ["cyberrealisticpony", "pony", "epicphotogasm", "chilloutmix", "realvisxl"]
    elif is_asian:
        priority = ["chilloutmix", "epicphotogasm", "realvisxl", "cyberrealistic"]
    else:
        priority = ["epicphotogasm", "realvisxl", "chilloutmix", "cyberrealistic", "juggernaut", "epicrealism"]

    for pref in priority:
        for m in models:
            if pref.lower() in m.lower():
                return m
    # Skip base SD1.5 / SDXL models (not NSFW-tuned)
    for m in models:
        name = m.lower()
        if "v1-5-pruned" not in name and "sd_xl_base" not in name and "flux" not in name:
            return m
    return models[0]


def generate_undress_edit(editor_data, undress_mode, custom_prompt, seed, denoise_strength,
                          model_sel="(自動)", lora1="None", lora1_str=0.8):
    """Edit clothing using inpainting (mask) or img2img fallback."""
    if editor_data is None:
        raise gr.Error("画像をアップロードしてください。")

    from PIL import Image as PILImage
    import numpy as np

    edit_prompt, edit_negative, default_strength = get_undress_params(undress_mode)
    if custom_prompt and custom_prompt.strip():
        edit_prompt += ", " + custom_prompt.strip()

    # Use user-specified denoise or default from mode
    strength = denoise_strength if denoise_strength is not None else default_strength

    output_dir = config["output_dir_adult"]
    os.makedirs(output_dir, exist_ok=True)
    ts = int(time.time())
    s = int(seed) if seed is not None and seed >= 0 else -1

    # Extract image and mask from editor
    # Gradio ImageEditor returns dict: {"background": ndarray, "layers": [ndarray...], "composite": ndarray}
    if isinstance(editor_data, dict):
        bg = editor_data.get("background")
        layers = editor_data.get("layers", [])
        composite = editor_data.get("composite")

        # Save background image
        if bg is not None:
            img_pil = PILImage.fromarray(bg) if isinstance(bg, np.ndarray) else PILImage.open(bg)
        elif composite is not None:
            img_pil = PILImage.fromarray(composite) if isinstance(composite, np.ndarray) else PILImage.open(composite)
        else:
            raise gr.Error("画像が読み取れません。")

        # Extract mask from layers (white = edit area)
        mask_pil = None
        for layer in layers:
            if layer is not None:
                layer_arr = np.array(layer) if not isinstance(layer, np.ndarray) else layer
                # Check if layer has any non-transparent pixels (mask drawn)
                if layer_arr.ndim == 3 and layer_arr.shape[2] == 4:
                    alpha = layer_arr[:, :, 3]
                    if alpha.max() > 0:
                        mask_pil = PILImage.fromarray((alpha > 128).astype(np.uint8) * 255, mode="L")
                        break
                elif layer_arr.ndim == 3 and layer_arr.shape[2] == 3:
                    gray = np.mean(layer_arr, axis=2)
                    if gray.max() > 10:
                        mask_pil = PILImage.fromarray((gray > 10).astype(np.uint8) * 255, mode="L")
                        break
                elif layer_arr.ndim == 2:
                    if layer_arr.max() > 0:
                        mask_pil = PILImage.fromarray((layer_arr > 128).astype(np.uint8) * 255, mode="L")
                        break
    elif isinstance(editor_data, np.ndarray):
        img_pil = PILImage.fromarray(editor_data)
        mask_pil = None
    else:
        img_pil = PILImage.open(editor_data) if isinstance(editor_data, str) else PILImage.fromarray(editor_data)
        mask_pil = None

    img_path = os.path.join(output_dir, f"undress_input_{ts}.png")
    img_pil.save(img_path)

    has_mask = mask_pil is not None
    if has_mask:
        # Embed mask as alpha channel into the image for ComfyUI LoadImage
        # ComfyUI: alpha=0 → inpaint (regenerate), alpha=255 → keep
        # User paints white (255) on clothing = area to change → invert for alpha
        inverted_mask = PILImage.fromarray(255 - np.array(mask_pil), mode="L")
        img_rgba = img_pil.convert("RGBA")
        img_rgba.putalpha(inverted_mask)
        img_path_inpaint = os.path.join(output_dir, f"undress_inpaint_{ts}.png")
        img_rgba.save(img_path_inpaint)
        # Also save mask separately for debugging
        mask_path = os.path.join(output_dir, f"undress_mask_{ts}.png")
        mask_pil.save(mask_path)
        logger.info(f"[Undress] Mask detected, using inpainting mode")
    else:
        img_path_inpaint = img_path
        logger.info(f"[Undress] No mask, using img2img mode (denoise={strength})")

    errors = []

    # ── 1. ComfyUI Inpainting/img2img — 完全無検閲 ──
    if client.is_server_running():
        try:
            img_filename = f"undress_{ts}.png"
            client.upload_image(img_path, img_filename)

            remote_models = client.get_models()
            if not remote_models:
                remote_models = _get_models_for_backend()
            if model_sel and model_sel not in ("(自動)", ""):
                model_name = model_sel
            else:
                model_name = _select_nsfw_model(remote_models)
            # Build LoRA list for undress
            ud_loras = []
            if lora1 and lora1 not in ("None", "なし", ""):
                ud_loras.append((lora1, float(lora1_str)))
            logger.info(f"[Undress] ComfyUI model: {model_name}")

            if has_mask:
                # Inpainting: upload image with alpha mask embedded
                inpaint_filename = f"undress_inpaint_{ts}.png"
                client.upload_image(img_path_inpaint, inpaint_filename)
                workflow = build_inpaint_workflow(
                    edit_prompt, edit_negative, model_name,
                    image_path=inpaint_filename,
                    width=img_pil.width, height=img_pil.height,
                    steps=35, cfg=5.0,
                    sampler="euler_ancestral", scheduler="normal",
                    seed=s, denoise=strength,
                )
            else:
                # img2img: lower denoise to preserve more of original
                workflow = build_img2img_workflow(
                    edit_prompt, edit_negative, model_name,
                    image_path=img_filename,
                    width=img_pil.width, height=img_pil.height,
                    steps=30, cfg=7.0,
                    sampler="euler_ancestral", scheduler="normal",
                    seed=s, denoise=strength,
                )

            result = client.generate(workflow)
            logger.info(f"[Undress] ComfyUI result: {len(result) if result else 0} images")
            if result:
                saved = []
                for img in result:
                    saved.append(save_image_to_dir(img, output_dir, prefix="undress"))
                mode_str = "Inpaint (マスク)" if has_mask else "img2img"
                return result, f"[ComfyUI {mode_str} 無検閲] {undress_mode} | Model: {model_name} | denoise={strength}\n保存先: {', '.join(saved)}"
        except Exception as e:
            logger.error(f"[Undress] ComfyUI failed: {e}")
            errors.append(f"ComfyUI: {e}")

    # ── 2. Novita.ai img2img — 無検閲 ──
    if novita.api_key:
        try:
            model_key = "Realistic Vision (フォトリアル)" if "anime" not in edit_prompt.lower() else "MeinaMix (アニメ)"
            logger.info(f"[Undress] Trying Novita img2img with {model_key}")
            urls = novita.img2img(model_key, edit_prompt, img_path,
                                  negative_prompt=edit_negative, strength=strength, seed=s)
            if urls:
                images = []
                saved = []
                for url in urls:
                    img = download_novita_image(url)
                    if img:
                        images.append(img)
                        saved.append(save_image_to_dir(img, output_dir, prefix="undress_novita"))
                if images:
                    return images, f"[Novita 無検閲 img2img] {undress_mode}\n保存先: {', '.join(saved)}"
        except Exception as e:
            logger.error(f"[Undress] Novita failed: {e}")
            errors.append(f"Novita img2img: {e}")

    # ── 3. Dezgo img2img — 無検閲 ──
    if dezgo.api_key:
        try:
            logger.info(f"[Undress] Trying Dezgo img2img")
            img_bytes = dezgo.img2img(edit_prompt, img_path,
                                      negative_prompt=edit_negative, strength=strength, seed=s)
            if img_bytes:
                img = decode_dezgo_image(img_bytes)
                if img:
                    path = save_image_to_dir(img, output_dir, prefix="undress_dezgo")
                    return [img], f"[Dezgo 無検閲 img2img] {undress_mode}\n保存先: {path}"
        except Exception as e:
            errors.append(f"Dezgo img2img: {e}")

    # ── 4. Fallback: 元画像のプロンプトでtxt2img再生成 ──
    try:
        return _adult_generate_image(edit_prompt, edit_negative, seed=s, prefix="undress_regen")
    except Exception as e:
        errors.append(f"txt2img fallback: {e}")

    error_detail = "\n".join(errors)
    raise gr.Error(f"編集失敗:\n{error_detail}\n\n推奨: Novita.ai / Dezgo / Vast.ai ComfyUI のAPIキーを設定してください。")


def generate_undress_video(image, undress_mode, custom_prompt, duration):
    """Generate undress video/GIF from image. Auto-fallback: fal.ai → Dezgo → ComfyUI."""
    if image is None:
        raise gr.Error("画像をアップロードしてください。")

    edit_prompt, edit_negative, strength = get_undress_params(undress_mode)
    if custom_prompt and custom_prompt.strip():
        edit_prompt += ", " + custom_prompt.strip()

    motion_prompts = {
        "全脱衣 (ヌードに)": "slowly removing all clothing, undressing, strip tease, smooth motion",
        "トップレスに": "slowly removing top, taking off shirt, revealing breasts, smooth motion",
        "ランジェリーに変更": "changing into lingerie, putting on lace underwear, smooth motion",
        "ビキニに変更": "changing into bikini, putting on swimsuit, smooth motion",
        "服を透けさせる": "clothes becoming wet and transparent, water splashing, see through, smooth motion",
        "服を破る (ダメージ)": "clothes tearing apart, fabric ripping, battle damage reveal, dramatic motion",
        "エプロンのみに": "removing clothes leaving only apron, undressing to apron, smooth motion",
        "タオル一枚に": "stepping out of shower wrapped in towel, towel slowly slipping, smooth motion",
    }
    video_prompt = motion_prompts.get(undress_mode, "slowly undressing, smooth motion")
    full_prompt = f"{edit_prompt}, {video_prompt}"
    dur = int(duration) if duration else 5
    errors = []

    # ── 1. fal.ai (Wan NSFW OK) ──
    if fal.api_key:
        try:
            model_key = "Wan 2.6 img2vid (最新・NSFW OK)"
            return generate_fal_img2vid(image, full_prompt, model_key, "adult", dur)
        except Exception as e:
            errors.append(f"fal.ai: {e}")
            # Try Wan 2.1 as fallback
            try:
                model_key = "Wan 2.1 img2vid (高品質・NSFW OK)"
                return generate_fal_img2vid(image, full_prompt, model_key, "adult", dur)
            except Exception as e2:
                errors.append(f"fal.ai Wan2.1: {e2}")

    # ── 2. Dezgo txt2vid (画像は使えないがプロンプトで生成) ──
    if dezgo.api_key:
        try:
            return generate_dezgo_video(full_prompt, "Wan 2.6 (最新・高品質動画)", "adult")
        except Exception as e:
            errors.append(f"Dezgo: {e}")

    error_detail = "\n".join(errors) if errors else "動画生成に使えるAPIキーがありません"
    raise gr.Error(f"Undress動画生成失敗:\n{error_detail}\n\nfal.ai / Dezgo のAPIキーを設定してください。")


def generate_adult_video(scene_key, custom_addition, from_image, seed,
                         model_sel="(自動)", lora1="None", lora1_str=0.8):
    """Generate adult video from scene preset. Auto-fallback: fal.ai → Dezgo → Replicate → ComfyUI."""
    prompt, negative, duration = compose_video_prompt(scene_key, custom_addition)
    if not prompt.strip():
        raise gr.Error("シーンを選択してください。")

    errors = []
    s = int(seed) if seed is not None else -1

    # ── 1. fal.ai (Wan NSFW OK) ──
    if fal.api_key:
        try:
            if from_image is not None:
                model_key = "Wan 2.6 img2vid (最新・NSFW OK)"
                return generate_fal_img2vid(from_image, prompt, model_key, "adult", duration)
            else:
                model_key = "Wan 2.6 txt2vid (最新・NSFW OK)"
                video_path, info = generate_fal_video(prompt, model_key, "adult",
                                                       duration=duration, negative_prompt=negative, seed=s)
                return _gradio_safe_video(video_path), info
        except Exception as e:
            errors.append(f"fal.ai: {e}")
            logger.error(f"Adult video fal.ai failed: {e}")

    # ── 2. Dezgo (無検閲) ──
    if dezgo.api_key:
        try:
            return generate_dezgo_video(prompt, "Wan 2.6 (最新・高品質動画)", "adult")
        except Exception as e:
            errors.append(f"Dezgo: {e}")
            logger.error(f"Adult video Dezgo failed: {e}")

    # ── 3. Replicate ──
    if replicate.api_key:
        try:
            replicate_models = list(REPLICATE_VIDEO_MODELS.keys())
            if replicate_models:
                return generate_replicate_video(prompt, replicate_models[0], "adult")
        except Exception as e:
            errors.append(f"Replicate: {e}")
            logger.error(f"Adult video Replicate failed: {e}")

    # ── 4. ComfyUI AnimateDiff (ローカル/Vast.ai) ──
    if client.is_server_running() and from_image is None:
        try:
            remote_models = client.get_models()
            if not remote_models:
                remote_models = _get_models_for_backend()
            if model_sel and model_sel not in ("(自動)", ""):
                model_name = model_sel
            else:
                model_name = _select_nsfw_model(remote_models)
            # Build LoRA list
            vid_loras = []
            if lora1 and lora1 not in ("None", "なし", ""):
                vid_loras.append((lora1, float(lora1_str)))
            workflow = build_animatediff_workflow(
                prompt, negative, model_name,
                motion_model="mm_sd_v15_v2.ckpt",
                width=512, height=512, steps=20, cfg=7.0,
                sampler="euler_ancestral", scheduler="normal",
                seed=s, frame_count=16, fps=8,
                loras=vid_loras,
            )
            frames, video_path = client.generate_video(workflow, config["output_dir_adult"])
            if video_path:
                lora_info = f"\nLoRA: {', '.join(f'{n} ({s})' for n, s in vid_loras)}" if vid_loras else ""
                return _gradio_safe_video(video_path), f"[ComfyUI AnimateDiff 無検閲] {model_name}{lora_info}\n保存先: {video_path}"
        except Exception as e:
            errors.append(f"ComfyUI AnimateDiff: {e}")
            logger.error(f"Adult video ComfyUI failed: {e}")

    error_detail = "\n".join(errors) if errors else "動画生成に使えるAPIキーがありません"
    raise gr.Error(f"アダルト動画生成失敗:\n{error_detail}\n\nfal.ai / Dezgo / Replicate のAPIキーを設定してください。")


def generate_fal_video(prompt, model_key, mode="normal", duration=None,
                       width=None, height=None, negative_prompt="", seed=-1):
    """Generate video via fal.ai API."""
    if not fal.api_key:
        raise gr.Error("fal.ai API Key が未設定です。Settingsタブで設定してください。")
    if not prompt.strip():
        raise gr.Error("プロンプトを入力してください。")

    try:
        dur = int(duration) if duration else None
        w = int(width) if width else None
        h = int(height) if height else None
        s = int(seed) if seed is not None else -1
        video_url = fal.generate_video(model_key, prompt,
                                       negative_prompt=negative_prompt,
                                       duration=dur, width=w, height=h, seed=s)
        if not video_url:
            raise RuntimeError("動画URLが取得できませんでした")

        output_dir = config["output_dir_adult"] if mode == "adult" else config["output_dir_normal"]
        os.makedirs(output_dir, exist_ok=True)
        import datetime as dt
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(output_dir, f"fal_video_{ts}.mp4")
        download_fal_video(video_url, video_path)

        from fal_api import FAL_VIDEO_MODELS as FVM
        model_info = FVM.get(model_key, {})
        cost = model_info.get("cost", "?")
        dur_info = f" | {dur}秒" if dur else ""
        return _gradio_safe_video(video_path), f"[fal.ai {model_key}] コスト: {cost}{dur_info}\n保存先: {video_path}"
    except Exception as e:
        error_str = str(e)
        logger.error(f"fal.ai動画生成エラー: {traceback.format_exc()}")
        if "content_policy_violation" in error_str:
            if dezgo.api_key:
                logger.info("fal.ai NSFWブロック → Dezgo にフォールバック")
                return generate_dezgo_video(prompt, "Wan 2.6 (最新・高品質動画)", mode)
            raise gr.Error(
                "fal.aiがNSFWコンテンツをブロックしました。\n\n"
                "対策:\n"
                "1. Dezgo API Key を設定すると自動でDezgo (無検閲) にフォールバックします\n"
                "2. ローカル AnimateDiff を使う (遅いがNSFW制限なし)\n"
                "3. AI動画 (Dezgo) タブを直接使う"
            )
        raise gr.Error(f"fal.ai動画生成エラー: {e}")


def generate_dezgo_video(prompt, model_key, mode="normal"):
    """Generate video via Dezgo API."""
    if not dezgo.api_key:
        raise gr.Error("Dezgo API Key が未設定です。Settingsタブで設定してください。")
    if not prompt.strip():
        raise gr.Error("プロンプトを入力してください。")

    try:
        mp4_bytes = dezgo.generate_video(model_key=model_key, prompt=prompt)
        output_dir = config["output_dir_adult"] if mode == "adult" else config["output_dir_normal"]
        os.makedirs(output_dir, exist_ok=True)
        import datetime as dt
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(output_dir, f"dezgo_video_{ts}.mp4")
        with open(video_path, "wb") as f:
            f.write(mp4_bytes)
        cost = DEZGO_VIDEO_MODELS.get(model_key, {}).get("cost", "?")
        return _gradio_safe_video(video_path), f"[Dezgo {model_key}] コスト: {cost}\n保存先: {video_path}"
    except Exception as e:
        logger.error(f"Dezgo動画生成エラー: {traceback.format_exc()}")
        raise gr.Error(f"Dezgo動画生成エラー: {e}")


def generate_replicate_image(prompt, model_key, width, height, seed, mode="normal"):
    """Generate image via Replicate API."""
    if not replicate.api_key:
        raise gr.Error("Replicate API Key が未設定です。Settingsタブで設定してください。")
    if not prompt.strip():
        raise gr.Error("プロンプトを入力してください。")

    try:
        urls = replicate.generate_image(
            model_key=model_key,
            prompt=prompt,
            width=int(width),
            height=int(height),
            seed=int(seed),
        )

        images = []
        saved_paths = []
        output_dir = config["output_dir_adult"] if mode == "adult" else config["output_dir_normal"]
        for url in urls:
            img = download_url_to_pil(url)
            images.append(img)
            path = save_image_to_dir(img, output_dir, prefix=f"replicate_{mode}")
            saved_paths.append(path)

        model_info = REPLICATE_MODELS.get(model_key, {})
        cost = model_info.get("cost_per_image", "?")
        return images, f"[Replicate {model_key}] コスト: {cost}\n保存先: {', '.join(saved_paths)}"
    except Exception as e:
        raise gr.Error(f"Replicate生成エラー: {e}")


def generate_replicate_video(prompt, model_key, mode="normal"):
    """Generate video via Replicate API."""
    if not replicate.api_key:
        raise gr.Error("Replicate API Key が未設定です。Settingsタブで設定してください。")
    if not prompt.strip():
        raise gr.Error("プロンプトを入力してください。")

    try:
        urls = replicate.generate_video(model_key=model_key, prompt=prompt)

        output_dir = config["output_dir_adult"] if mode == "adult" else config["output_dir_normal"]
        video_path = None
        for url in urls:
            # Download video
            import datetime as dt
            ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = ".mp4"
            if ".webp" in url:
                ext = ".webp"
            elif ".gif" in url:
                ext = ".gif"
            video_path = os.path.join(output_dir, f"replicate_video_{ts}{ext}")
            os.makedirs(output_dir, exist_ok=True)
            req = urllib.request.Request(url, headers={"User-Agent": "AI-diffusion/1.0"})
            resp = urllib.request.urlopen(req, timeout=120)
            with open(video_path, "wb") as f:
                f.write(resp.read())
            break

        model_info = REPLICATE_VIDEO_MODELS.get(model_key, {})
        cost = model_info.get("cost_per_video", "?")
        return _gradio_safe_video(video_path), f"[Replicate {model_key}] コスト: {cost}\n保存先: {video_path}"
    except Exception as e:
        logger.error(f"Replicate動画生成エラー: {traceback.format_exc()}")
        raise gr.Error(f"Replicate動画生成エラー: {e}")


def generate_video_vid2vid(video_file, prompt, neg, model, motion_model, vae,
                           w, h, steps, cfg, sampler, sched, seed,
                           fps, denoise, frame_limit, output_format, mode="normal"):
    """Transform existing video using AnimateDiff vid2vid."""
    if not client.is_server_running():
        raise gr.Error("ComfyUI Server が起動していません。")
    if not model:
        raise gr.Error("モデルが選択されていません。")
    if video_file is None:
        raise gr.Error("動画ファイルをアップロードしてください。")

    vae_name = "" if vae == "None" else vae

    # video_file is a filepath string from Gradio
    video_path = video_file if isinstance(video_file, str) else video_file.name

    workflow = build_vid2vid_workflow(
        video_path=video_path,
        prompt=prompt,
        negative_prompt=neg,
        model=model,
        motion_model=motion_model,
        width=int(w),
        height=int(h),
        steps=int(steps),
        cfg=cfg,
        sampler=sampler,
        scheduler=sched,
        seed=int(seed),
        denoise=denoise,
        fps=int(fps),
        frame_limit=int(frame_limit),
        output_format=output_format,
    )

    output_dir = config["output_dir_adult"] if mode == "adult" else config["output_dir_normal"]
    frames, video_path_out = client.generate_video(workflow, output_dir, timeout=900)
    info = f"Frames: {len(frames)}"
    if video_path_out:
        info += f"\n保存先: {video_path_out}"
    return video_path_out, frames[:4] if frames else [], info


# ──────────────────────────────────────────────
# Settings functions
# ──────────────────────────────────────────────

def save_settings(output_normal, output_adult, gdrive_path, icloud_path, comfyui_url, runpod_api_key, backend_choice,
                   replicate_api_key, fal_api_key, together_api_key, dezgo_api_key, novita_api_key,
                   civitai_api_key, anthropic_api_key, openai_api_key, xai_api_key,
                   vast_api_key=""):
    config["output_dir_normal"] = output_normal
    config["output_dir_adult"] = output_adult
    config["google_drive_models_dir"] = gdrive_path
    config["icloud_models_dir"] = icloud_path
    config["comfyui_url"] = comfyui_url
    config["runpod_api_key"] = runpod_api_key
    config["backend"] = backend_choice
    config["replicate_api_key"] = replicate_api_key
    config["fal_api_key"] = fal_api_key
    config["together_api_key"] = together_api_key
    config["dezgo_api_key"] = dezgo_api_key
    config["novita_api_key"] = novita_api_key
    config["civitai_api_key"] = civitai_api_key
    config["anthropic_api_key"] = anthropic_api_key
    config["openai_api_key"] = openai_api_key
    config["xai_api_key"] = xai_api_key
    config["vast_api_key"] = vast_api_key
    save_config(config)
    client.server_url = comfyui_url.rstrip("/")
    runpod.api_key = runpod_api_key
    vastai.set_key(vast_api_key)
    replicate.api_key = replicate_api_key
    fal.set_key(fal_api_key)
    together.set_key(together_api_key)
    dezgo.set_key(dezgo_api_key)
    novita.set_key(novita_api_key)
    civitai.api_key = civitai_api_key
    return "設定を保存しました"


def switch_backend(choice):
    config["backend"] = choice
    if choice == "local":
        config["comfyui_url"] = "http://127.0.0.1:8188"
        client.server_url = "http://127.0.0.1:8188"
        save_config(config)
        return "ローカルに切り替えました", config["comfyui_url"]

    # ── Replicate (serverless, no pods needed) ──
    if choice == "replicate":
        save_config(config)
        if not replicate.api_key:
            return "Replicate API Key が未設定です。Settingsタブで設定してください。\nhttps://replicate.com/account/api-tokens", ""
        return (
            "Replicate に切り替えました!\n"
            "プロンプトを入力して生成できます。\n"
            "在庫切れなし・Pod管理不要・従量課金。"
        ), ""

    # ── Vast.ai (CivitAI models, full freedom) ──
    if choice == "vast":
        save_config(config)
        url = config.get("vast_comfyui_url")
        if url:
            config["comfyui_url"] = url
            client.server_url = url
            save_config(config)
            return (
                f"Vast.ai に切り替えました!\n"
                f"ComfyUI URL: {url}\n"
                f"CivitAIモデル自由使用可能。SDXL/Pony対応。"
            ), url
        return (
            "Vast.ai に切り替えましたが、ComfyUIが未接続です。\n"
            "Settingsタブで「状態確認/接続」を実行してください。"
        ), ""

    # ── fal.ai (Flux + NSFW OK) ──
    if choice == "fal":
        save_config(config)
        if not fal.api_key:
            return "fal.ai API Key が未設定です。Settingsタブで設定してください。", ""
        return (
            "fal.ai に切り替えました!\n"
            "Flux品質で生成できます。NSFW制限なし。\n"
            "コスト: Flux Dev ~$0.025/枚、Schnell ~$0.003/枚。"
        ), ""

    # ── Together.ai ──
    if choice == "together":
        save_config(config)
        if not together.api_key:
            return "Together.ai API Key が未設定です。Settingsタブで設定してください。\nhttps://api.together.ai/settings/api-keys", ""
        return (
            "Together.ai に切り替えました!\n"
            "Flux品質・NSFW OK・LoRA対応。\n"
            "コスト: Schnell ~$0.003/枚、Dev ~$0.025/枚。"
        ), ""

    # ── Dezgo (uncensored) ──
    if choice == "dezgo":
        save_config(config)
        if not dezgo.api_key:
            return "Dezgo API Key が未設定です。Settingsタブで設定してください。\nhttps://dezgo.com/account", ""
        return (
            "Dezgo に切り替えました!\n"
            "完全無検閲・画像+動画対応。\n"
            "コスト: Flux ~$0.02/枚、動画 ~$0.10/本。"
        ), ""

    # ── Novita.ai (uncensored models) ──
    if choice == "novita":
        save_config(config)
        if not novita.api_key:
            return "Novita.ai API Key が未設定です。Settingsタブで設定してください。\nhttps://novita.ai/dashboard/key", ""
        return (
            "Novita.ai に切り替えました!\n"
            "無検閲モデル・NSFW OK・最安級。\n"
            "コスト: ~$0.0015/枚〜。"
        ), ""

    # ── CivitAI (serverless, NSFW OK) ──
    if choice == "civitai":
        save_config(config)
        if not civitai.api_key:
            return "CivitAI API Key が未設定です。Settingsタブで設定してください。", ""
        return (
            "CivitAI に切り替えました!\n"
            "NSFW OK・CivitAI全モデル利用可能。\n"
            "コスト: SD1.5 ~1Buzz/枚、SDXL ~4Buzz/枚。"
        ), ""

    # ── RunPod auto-connect (予備) ──
    if choice == "runpod":
        if not runpod.api_key:
            return "RunPod API Key が未設定です。Settingsタブで設定してください。", config["comfyui_url"]

        # 1. Already have URL? Try it first
        if config.get("runpod_comfyui_url"):
            config["comfyui_url"] = config["runpod_comfyui_url"]
            client.server_url = config["runpod_comfyui_url"]
            config["backend"] = "runpod"
            save_config(config)
            if client.is_server_running():
                return f"RunPod 接続完了: {config['runpod_comfyui_url']}", config["comfyui_url"]

        # 2. Have pod ID? Check status and connect
        pod_id = config.get("runpod_pod_id", "")
        if pod_id:
            try:
                pod = runpod.get_pod(pod_id)
                if pod:
                    if pod["desiredStatus"] == "RUNNING":
                        url = runpod.get_comfyui_url(pod)
                        if url:
                            config["runpod_comfyui_url"] = url
                            config["comfyui_url"] = url
                            client.server_url = url
                            save_config(config)
                            cost_info = format_pod_cost(pod)
                            return f"RunPod 自動接続完了!\n{cost_info}\nURL: {url}", url
                    elif pod["desiredStatus"] in ("EXITED", "STOPPED"):
                        # Auto-restart stopped pod
                        runpod.start_pod(pod_id)
                        # Wait for it to come up
                        for _ in range(30):
                            time.sleep(3)
                            pod = runpod.get_pod(pod_id)
                            if pod and pod["desiredStatus"] == "RUNNING":
                                url = runpod.get_comfyui_url(pod)
                                if url:
                                    config["runpod_comfyui_url"] = url
                                    config["comfyui_url"] = url
                                    client.server_url = url
                                    save_config(config)
                                    cost_info = format_pod_cost(pod)
                                    return f"RunPod Pod再起動 → 接続完了!\n{cost_info}\nURL: {url}", url
                        return "Pod再起動中...数秒後にもう一度 runpod を選択してください。", config["comfyui_url"]
            except Exception as e:
                return f"RunPod接続エラー: {e}", config["comfyui_url"]

        # 3. No pod at all — create one with auto-retry
        max_retries = 5
        for attempt in range(max_retries):
            try:
                new_pod = runpod.create_pod(auto_fallback=True)
                config["runpod_pod_id"] = new_pod["id"]
                save_config(config)
                actual_gpu = new_pod.get("machine", {}).get("gpuDisplayName", "")
                # Wait for pod to come up
                for _ in range(40):
                    time.sleep(3)
                    pod = runpod.get_pod(new_pod["id"])
                    if pod and pod["desiredStatus"] == "RUNNING":
                        url = runpod.get_comfyui_url(pod)
                        if url:
                            config["runpod_comfyui_url"] = url
                            config["comfyui_url"] = url
                            client.server_url = url
                            save_config(config)
                            cost_info = format_pod_cost(pod)
                            return f"RunPod Pod新規作成 → 接続完了!\nGPU: {actual_gpu}\n{cost_info}\nURL: {url}", url
                return f"Pod作成済み（ID: {new_pod['id']}）、まだ起動中。数秒後に再度 runpod を選択してください。", config["comfyui_url"]
            except RuntimeError as e:
                if "在庫切れ" in str(e) and attempt < max_retries - 1:
                    # Wait and retry
                    time.sleep(30)
                    continue
                # All retries failed — fall back to local
                config["backend"] = "local"
                config["comfyui_url"] = "http://127.0.0.1:8188"
                client.server_url = "http://127.0.0.1:8188"
                save_config(config)
                return (
                    f"全GPU在庫切れ（{max_retries}回リトライ済み）\n"
                    f"→ ローカル (Mac MPS) にフォールバックしました。\n\n"
                    f"対処法:\n"
                    f"・数分〜数時間後に再試行（日本の朝〜昼が空きやすい）\n"
                    f"・Settingsタブの「空きGPU確認」で状況チェック\n"
                    f"・ローカルでも Quick/Adult タブの画像生成は可能（遅め）"
                ), config["comfyui_url"]
            except Exception as e:
                return f"RunPod Pod作成エラー: {e}", config["comfyui_url"]

    return "不明なバックエンド", config["comfyui_url"]


def runpod_start(gpu_choice="NVIDIA RTX A5000", template_choice="ComfyUI + Flux (推奨)"):
    if not runpod.api_key:
        return "RunPod API Key を Settings で設定してください"
    pod_id = config.get("runpod_pod_id", "")
    if pod_id:
        try:
            pod = runpod.get_pod(pod_id)
            if pod and pod["desiredStatus"] == "RUNNING":
                url = runpod.get_comfyui_url(pod)
                if url:
                    config["runpod_comfyui_url"] = url
                    config["comfyui_url"] = url
                    client.server_url = url
                    save_config(config)
                    return f"既に起動中: {format_pod_status(pod)}\nURL: {url}"
            if pod:
                runpod.start_pod(pod_id)
                return f"Pod再開リクエスト送信。「状態確認」ボタンで起動を確認してください。\nPod ID: {pod_id}"
        except Exception as e:
            return f"エラー: {e}"

    # No existing pod, create new one
    try:
        new_pod = runpod.create_pod(gpu_type_id=gpu_choice, auto_fallback=True, template_key=template_choice)
        actual_gpu = new_pod.get("machine", {}).get("gpuDisplayName", gpu_choice)
        config["runpod_pod_id"] = new_pod["id"]
        save_config(config)
        return (
            f"Pod作成完了! 起動中...\n"
            f"Pod ID: {new_pod['id']}\n"
            f"GPU: {actual_gpu}\n"
            f"「状態確認」ボタンで起動完了を確認し、Backend を runpod に切り替えてください。"
        )
    except Exception as e:
        return f"Pod作成エラー: {e}"


def runpod_connect():
    """Check pod status and connect if ready."""
    pod_id = config.get("runpod_pod_id", "")
    if not pod_id:
        return "Pod が未作成です"
    try:
        pod = runpod.get_pod(pod_id)
        if not pod:
            return "Pod が見つかりません（削除済み？）"
        if pod["desiredStatus"] != "RUNNING":
            return f"Pod はまだ起動中... Status: {pod['desiredStatus']}\n数秒待って再度「状態確認」を押してください。"
        url = runpod.get_comfyui_url(pod)
        if url:
            config["runpod_comfyui_url"] = url
            config["comfyui_url"] = url
            config["backend"] = "runpod"
            client.server_url = url
            save_config(config)
            return f"接続完了!\n{format_pod_status(pod)}\nURL: {url}\nBackend を runpod に自動切替しました。"
        return f"Pod起動中だがURLが未取得。もう少し待って再度確認してください。\n{format_pod_status(pod)}"
    except Exception as e:
        return f"確認エラー: {e}"


def runpod_stop():
    pod_id = config.get("runpod_pod_id", "")
    if not pod_id:
        return "Pod ID が設定されていません"
    try:
        # Get cost info before stopping
        cost_info = ""
        try:
            pod = runpod.get_pod(pod_id)
            if pod:
                cost_info = "\n" + format_pod_cost(pod)
        except Exception:
            pass
        runpod.stop_pod(pod_id)
        config["backend"] = "local"
        config["comfyui_url"] = "http://127.0.0.1:8188"
        client.server_url = "http://127.0.0.1:8188"
        save_config(config)
        return f"Pod を停止しました（ストレージは保持、課金停止）{cost_info}"
    except Exception as e:
        return f"停止エラー: {e}"


def runpod_check_status():
    """Check status and auto-connect if pod is ready. Show cost."""
    if not runpod.api_key:
        return "API Key が未設定です"
    pod_id = config.get("runpod_pod_id", "")
    if not pod_id:
        return "Pod が未作成です。Backend を runpod に切り替えると自動作成されます。"
    try:
        pod = runpod.get_pod(pod_id)
        if not pod:
            return "Pod が見つかりません（削除済み？）"
        cost_info = format_pod_cost(pod)
        if pod["desiredStatus"] == "RUNNING":
            url = runpod.get_comfyui_url(pod)
            if url:
                config["runpod_comfyui_url"] = url
                config["comfyui_url"] = url
                config["backend"] = "runpod"
                client.server_url = url
                save_config(config)
                return f"接続完了!\n{cost_info}\nURL: {url}"
            return f"Pod起動中だがURL未取得。もう少し待ってください。\n{cost_info}"
        return f"Pod Status: {pod['desiredStatus']}\n{cost_info}"
    except Exception as e:
        return f"確認エラー: {e}"


# ──────────────────────────────────────────────
# CivitAI functions
# ──────────────────────────────────────────────

def civitai_search(query, model_type, sort, nsfw):
    try:
        results = civitai.search_models(
            query=query,
            model_type=model_type,
            sort=sort,
            nsfw=nsfw,
            limit=10,
        )
        return format_search_results(results)
    except Exception as e:
        return f"検索エラー: {e}"


def civitai_generate_cloud(prompt, negative_prompt, model_key, version_id_custom,
                           width, height, steps, cfg, seed, quantity):
    """Generate image via CivitAI cloud GPU."""
    if not civitai.api_key:
        raise gr.Error("CivitAI API Key が未設定です。Settingsタブで設定してください。")
    if not prompt.strip():
        raise gr.Error("プロンプトを入力してください。")

    try:
        # iCloudモデルが選ばれた場合: URN解決してクラウド生成
        if model_key and model_key.startswith("[iCloud] "):
            icloud_filename = model_key[len("[iCloud] "):]
            urn_info = civitai.resolve_icloud_model_urn(icloud_filename)
            if not urn_info:
                raise gr.Error(f"CivitAIでモデルが見つかりません: {icloud_filename}")
            if urn_info.get("type", "").lower() == "lora":
                raise gr.Error(f"{icloud_filename} はLoRAです。Checkpointモデルを選択してください。")

            urls = civitai.generate_image_by_urn(
                model_urn=urn_info["urn"], base_model=urn_info["base"],
                prompt=prompt, negative_prompt=negative_prompt,
                width=int(width), height=int(height),
                steps=int(steps), cfg_scale=float(cfg),
                seed=int(seed), quantity=int(quantity),
            )
            model_name = f"{urn_info['name']} ({urn_info.get('version', '')})"

            images = []
            saved_paths = []
            output_dir = config["output_dir_adult"]
            for url in urls:
                img = download_url_to_pil(url)
                images.append(img)
                path = save_image_to_dir(img, output_dir, prefix="civitai_icloud")
                saved_paths.append(path)

            cost = urn_info.get("cost", "~4 Buzz")
            return images, f"[CivitAI iCloud: {model_name}] コスト: {cost}/枚\n保存先: {', '.join(saved_paths)}"

        # If custom version ID is provided, use it; otherwise use preset
        if version_id_custom and version_id_custom.strip():
            vid = version_id_custom.strip()
            # Fetch model info to determine base model
            try:
                model_data = civitai.get_model_version(int(vid))
                base = model_data.get("baseModel", "SD 1.5")
                model_id = model_data.get("modelId", 0)
                if "SDXL" in base.upper():
                    base_model = "SDXL"
                    urn = f"urn:air:sdxl:checkpoint:civitai:{model_id}@{vid}"
                else:
                    base_model = "SD_1_5"
                    urn = f"urn:air:sd1:checkpoint:civitai:{model_id}@{vid}"
                model_name = model_data.get("model", {}).get("name", vid)
            except Exception:
                base_model = "SD_1_5"
                urn = f"urn:air:sd1:checkpoint:civitai:0@{vid}"
                model_name = f"Version {vid}"

            urls = civitai.generate_image_by_urn(
                model_urn=urn, base_model=base_model,
                prompt=prompt, negative_prompt=negative_prompt,
                width=int(width), height=int(height),
                steps=int(steps), cfg_scale=float(cfg),
                seed=int(seed), quantity=int(quantity),
            )
        else:
            urls = civitai.generate_image(
                model_key=model_key,
                prompt=prompt, negative_prompt=negative_prompt,
                width=int(width), height=int(height),
                steps=int(steps), cfg_scale=float(cfg),
                seed=int(seed), quantity=int(quantity),
            )
            model_name = model_key

        images = []
        saved_paths = []
        output_dir = config["output_dir_adult"]
        for url in urls:
            img = download_url_to_pil(url)
            images.append(img)
            path = save_image_to_dir(img, output_dir, prefix="civitai")
            saved_paths.append(path)

        cost = CIVITAI_GENERATION_MODELS.get(model_key, {}).get("cost", "?")
        return images, f"[CivitAI: {model_name}] コスト: {cost}/枚\n保存先: {', '.join(saved_paths)}"
    except Exception as e:
        logger.error(f"CivitAI Cloud生成エラー: {traceback.format_exc()}")
        raise gr.Error(f"CivitAI生成エラー: {e}")


# Google Drive model directories (A1111/Forge format -> ComfyUI format)
GDRIVE_MODELS_BASE = os.path.expanduser(
    "~/Library/CloudStorage/GoogleDrive-japanesebusinessman4@gmail.com/マイドライブ/AI_PICS/models"
)
GDRIVE_DEST_MAP = {
    "Checkpoint": "Stable-diffusion",
    "LoRA": "Lora",
    "VAE": "VAE",
    "ControlNet": "ControlNet",
    "Upscaler": "ESRGAN",
    "Embedding": "embeddings",
}
LOCAL_DEST_MAP = {
    "Checkpoint": "checkpoints",
    "LoRA": "loras",
    "VAE": "vae",
    "ControlNet": "controlnet",
    "Upscaler": "upscale_models",
    "Embedding": "embeddings",
}


def civitai_download(version_id, model_type_dest):
    """Download model from CivitAI by version ID. Saves to Google Drive + creates symlink."""
    if not version_id:
        return ("Version ID を入力してください（検索結果に表示されます）",) + refresh_all_model_dropdowns()

    local_subdir = LOCAL_DEST_MAP.get(model_type_dest, "checkpoints")
    local_dir = os.path.join(config["models_dir"], local_subdir)
    os.makedirs(local_dir, exist_ok=True)

    # Determine download destination: Google Drive if available, else local
    gdrive_subdir = GDRIVE_DEST_MAP.get(model_type_dest, "Stable-diffusion")
    gdrive_dir = os.path.join(GDRIVE_MODELS_BASE, gdrive_subdir)
    use_gdrive = os.path.isdir(GDRIVE_MODELS_BASE)

    if use_gdrive:
        os.makedirs(gdrive_dir, exist_ok=True)
        dest_dir = gdrive_dir
    else:
        dest_dir = local_dir

    try:
        info = civitai.get_download_url(int(version_id))
        if not info:
            return ("ダウンロードURLが見つかりません",) + refresh_all_model_dropdowns()
        size_gb = info["size_kb"] / 1024 / 1024

        if use_gdrive:
            # Download to local temp first, then move to Google Drive (avoids sync conflicts)
            import tempfile
            tmp_dir = os.path.join(tempfile.gettempdir(), "ai-diffusion-dl")
            os.makedirs(tmp_dir, exist_ok=True)
            tmp_filepath = civitai.download_model(int(version_id), tmp_dir)
            filename = os.path.basename(tmp_filepath)
            filepath = os.path.join(gdrive_dir, filename)
            shutil.move(tmp_filepath, filepath)

            # Create symlinks
            symlink_path = os.path.join(local_dir, filename)
            if not os.path.exists(symlink_path):
                os.symlink(filepath, symlink_path)
            comfyui_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "comfyui", "models", local_subdir)
            os.makedirs(comfyui_dir, exist_ok=True)
            comfyui_link = os.path.join(comfyui_dir, filename)
            if not os.path.exists(comfyui_link):
                os.symlink(filepath, comfyui_link)
            storage_info = "💾 Google Drive に保存 (Mac容量を消費しません)"
        else:
            filepath = civitai.download_model(int(version_id), local_dir)
            filename = os.path.basename(filepath)
            storage_info = "⚠️ ローカルに保存 (Google Drive未接続)"

        msg = (
            f"✅ ダウンロード完了!\n"
            f"File: {filename}\n"
            f"Size: {size_gb:.1f}GB\n"
            f"Path: {filepath}\n"
            f"{storage_info}\n\n"
            f"全タブのモデルリストを自動更新しました"
        )
        return (msg,) + refresh_all_model_dropdowns()
    except Exception as e:
        return (f"ダウンロードエラー: {e}",) + refresh_all_model_dropdowns()


def install_to_vastai(version_id_or_url, model_type_dest):
    """Download CivitAI model directly to vast.ai ComfyUI instance.
    For early access / new models not yet on Novita.ai.
    """
    comfyui_url = config.get("vast_comfyui_url") or config.get("comfyui_url", "")
    if not comfyui_url or "127.0.0.1" in comfyui_url:
        return "vast.ai ComfyUI が未設定です。Settingsで設定してください。"

    # Determine download URL
    if str(version_id_or_url).startswith("http"):
        download_url = version_id_or_url
        filename = download_url.split("/")[-1].split("?")[0] or "model.safetensors"
    else:
        try:
            info = civitai.get_download_url(int(version_id_or_url))
            if not info:
                return "CivitAI Version IDが見つかりません"
            download_url = info.get("url", "")
            filename = info.get("name", "model.safetensors")
        except Exception as e:
            return f"CivitAI URL取得エラー: {e}"

    # Add CivitAI API key for early access models
    if "civitai.com" in download_url and civitai.api_key:
        sep = "&" if "?" in download_url else "?"
        download_url += f"{sep}token={civitai.api_key}"

    # Determine destination path on ComfyUI
    subdir_map = {
        "Checkpoint": "checkpoints",
        "LoRA": "loras",
        "VAE": "vae",
        "Embedding": "embeddings",
        "ControlNet": "controlnet",
    }
    subdir = subdir_map.get(model_type_dest, "checkpoints")
    dest_path = f"/opt/ComfyUI/models/{subdir}/{filename}"

    # Use vast.ai API to exec wget on the instance
    vast_key = config.get("vast_api_key", "")
    instance_id = config.get("vast_instance_id")
    if vast_key and instance_id:
        import urllib.request
        import json as json_mod
        try:
            cmd = f"wget -q -O '{dest_path}' '{download_url}'"
            req = urllib.request.Request(
                f"https://console.vast.ai/api/v0/instances/{instance_id}/exec/",
                data=json_mod.dumps({"command": cmd}).encode(),
                headers={
                    "Authorization": f"Bearer {vast_key}",
                    "Content-Type": "application/json",
                },
                method="PUT",
            )
            resp = urllib.request.urlopen(req, timeout=30)
            if resp.status in (200, 201, 202):
                return (
                    f"✅ vast.ai ComfyUIにダウンロード開始\n"
                    f"ファイル: {filename}\n"
                    f"保存先: {dest_path}\n"
                    f"大きいモデルは数分かかります。"
                )
        except Exception as e:
            return f"vast.ai exec エラー: {e}"

    # Fallback: try ComfyUI API upload
    return "vast.ai APIキーまたはインスタンスIDが未設定です。Settingsで設定してください。"


def download_from_url(url, model_type_dest, custom_filename):
    """Download model from direct URL. Saves to Google Drive + creates symlink."""
    if not url or not url.startswith("http"):
        return ("有効なURLを入力してください",) + refresh_all_model_dropdowns()

    local_subdir = LOCAL_DEST_MAP.get(model_type_dest, "checkpoints")
    if model_type_dest == "UNET / Diffusion Model":
        local_subdir = "diffusion_models"
    elif model_type_dest == "CLIP / Text Encoder":
        local_subdir = "clip"
    local_dir = os.path.join(config["models_dir"], local_subdir)
    os.makedirs(local_dir, exist_ok=True)

    # Determine filename
    if custom_filename and custom_filename.strip():
        filename = custom_filename.strip()
    else:
        filename = url.split("/")[-1].split("?")[0]
        if not filename or "." not in filename:
            filename = "downloaded_model.safetensors"

    # Google Drive destination
    gdrive_subdir = GDRIVE_DEST_MAP.get(model_type_dest, "Stable-diffusion")
    gdrive_dir = os.path.join(GDRIVE_MODELS_BASE, gdrive_subdir)
    use_gdrive = os.path.isdir(GDRIVE_MODELS_BASE)

    # Check if already exists at final destination
    final_path = os.path.join(gdrive_dir if use_gdrive else local_dir, filename)
    if os.path.exists(final_path):
        return (f"既にファイルが存在します: {final_path}",) + refresh_all_model_dropdowns()
    # Also check symlink
    local_link = os.path.join(local_dir, filename)
    if os.path.exists(local_link):
        return (f"既にファイルが存在します: {local_link}",) + refresh_all_model_dropdowns()

    try:
        import tempfile
        # Always download to local temp first (fast, no sync conflicts)
        tmp_dir = os.path.join(tempfile.gettempdir(), "ai-diffusion-dl")
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_filepath = os.path.join(tmp_dir, filename)

        req = urllib.request.Request(url, headers={"User-Agent": "AI-diffusion/1.0"})
        resp = urllib.request.urlopen(req, timeout=600)
        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB

        with open(tmp_filepath, "wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)

        # Move to final destination
        if use_gdrive:
            os.makedirs(gdrive_dir, exist_ok=True)
            filepath = os.path.join(gdrive_dir, filename)
            shutil.move(tmp_filepath, filepath)
            # Create symlinks
            if not os.path.exists(local_link):
                os.symlink(filepath, local_link)
            comfyui_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "comfyui", "models", local_subdir)
            os.makedirs(comfyui_dir, exist_ok=True)
            comfyui_link = os.path.join(comfyui_dir, filename)
            if not os.path.exists(comfyui_link):
                os.symlink(filepath, comfyui_link)
            storage_info = "💾 Google Drive に保存"
        else:
            filepath = os.path.join(local_dir, filename)
            shutil.move(tmp_filepath, filepath)
            storage_info = "⚠️ ローカルに保存"

        size_gb = downloaded / (1024 * 1024 * 1024)
        msg = f"✅ ダウンロード完了!\nFile: {filename}\nSize: {size_gb:.2f}GB\nPath: {filepath}\n{storage_info}\n\n全タブのモデルリストを自動更新しました"
        return (msg,) + refresh_all_model_dropdowns()
    except Exception as e:
        # Clean up partial download
        tmp_path = os.path.join(tempfile.gettempdir(), "ai-diffusion-dl", filename)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return (f"ダウンロードエラー: {e}",) + refresh_all_model_dropdowns()


def get_model_summary():
    """Get a summary of all installed models."""
    models_dir = config["models_dir"]
    categories = {
        "Checkpoints": "checkpoints",
        "LoRAs": "loras",
        "VAE": "vae",
        "ControlNet": "controlnet",
        "Upscale Models": "upscale_models",
        "Embeddings": "embeddings",
        "Diffusion Models (UNET)": "diffusion_models",
        "CLIP / Text Encoders": "clip",
    }
    exts = (".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf")

    lines = ["📁 インストール済みモデル一覧", "=" * 50]
    total_count = 0
    total_size = 0

    for label, subdir in categories.items():
        path = os.path.join(models_dir, subdir)
        if not os.path.isdir(path):
            lines.append(f"{label}: (フォルダなし)")
            continue
        files = [f for f in os.listdir(path) if f.lower().endswith(exts)]
        count = len(files)
        total_count += count
        # Calculate total size
        cat_size = 0
        for f in files:
            fp = os.path.join(path, f)
            try:
                cat_size += os.path.getsize(fp)
            except OSError:
                pass
        total_size += cat_size
        size_gb = cat_size / (1024 ** 3)
        lines.append(f"{label}: {count} 個 ({size_gb:.1f}GB)")
        if count > 0 and count <= 10:
            for f in sorted(files):
                lines.append(f"  - {f}")

    lines.append("=" * 50)
    lines.append(f"合計: {total_count} モデル ({total_size / (1024 ** 3):.1f}GB)")

    # Also check AnimateDiff motion models
    from config import get_available_motion_models
    motion = get_available_motion_models()
    if motion:
        lines.append(f"\nMotion Models (AnimateDiff): {len(motion)} 個")
        for m in motion:
            lines.append(f"  - {m}")

    return "\n".join(lines)


def civitai_upload_image(image_path, title, nsfw):
    """Upload generated image to CivitAI."""
    if not civitai.api_key:
        return "CivitAI API Key を Settings で設定してください"
    if not image_path:
        return "画像を選択してください"
    try:
        result = civitai.upload_image(image_path)
        return f"アップロード完了! Image ID: {result.get('id', 'N/A')}"
    except Exception as e:
        return f"アップロードエラー: {e}"


def link_google_drive(gdrive_path):
    """Link Google Drive models folder to local models directory."""
    if not gdrive_path or not os.path.isdir(gdrive_path):
        return "指定されたパスが存在しません"

    models_dir = config["models_dir"]
    linked = []
    for subdir in ["checkpoints", "loras", "vae", "controlnet", "embeddings", "upscale_models"]:
        src = os.path.join(gdrive_path, subdir)
        if os.path.isdir(src):
            for f in os.listdir(src):
                src_file = os.path.join(src, f)
                dst_file = os.path.join(models_dir, subdir, f)
                if not os.path.exists(dst_file):
                    os.symlink(src_file, dst_file)
                    linked.append(f)
    if linked:
        return f"リンク完了: {', '.join(linked)}"
    return "リンクするファイルが見つかりませんでした。Google Driveのフォルダにcheckpoints/等のサブフォルダを作成してモデルを配置してください。"


def link_icloud_models(icloud_path):
    """Link iCloud CivitAI models to local models directory via symlinks."""
    if not icloud_path or not os.path.isdir(icloud_path):
        return "指定されたパスが存在しません"

    models_dir = config["models_dir"]
    linked = []

    # 1. Scan subdirectories (checkpoints/, loras/, etc.) like Google Drive
    for subdir in ["checkpoints", "loras", "vae", "controlnet", "embeddings", "upscale_models"]:
        src = os.path.join(icloud_path, subdir)
        if os.path.isdir(src):
            os.makedirs(os.path.join(models_dir, subdir), exist_ok=True)
            for f in os.listdir(src):
                if f.lower().endswith((".safetensors", ".ckpt", ".pt", ".pth")):
                    src_file = os.path.join(src, f)
                    dst_file = os.path.join(models_dir, subdir, f)
                    if not os.path.exists(dst_file):
                        os.symlink(src_file, dst_file)
                        linked.append(f)

    # 2. Scan root directory (flat structure — CivitAI downloads often land here)
    os.makedirs(os.path.join(models_dir, "checkpoints"), exist_ok=True)
    for f in os.listdir(icloud_path):
        if f.lower().endswith((".safetensors", ".ckpt")):
            src_file = os.path.join(icloud_path, f)
            if os.path.isfile(src_file):
                dst_file = os.path.join(models_dir, "checkpoints", f)
                if not os.path.exists(dst_file):
                    os.symlink(src_file, dst_file)
                    linked.append(f)

    # パスを保存（リンク有無に関わらず、iCloudスキャンで使う）
    config["icloud_models_dir"] = icloud_path
    save_config(config)

    if linked:
        return f"✅ リンク完了 ({len(linked)}件): {', '.join(linked)}"

    # 既にリンク済みのファイル数をカウント
    existing = 0
    for f in os.listdir(icloud_path):
        if f.lower().endswith((".safetensors", ".ckpt")):
            existing += 1
    if existing > 0:
        return f"✅ 全てリンク済み ({existing}件)。Modelドロップダウンに[iCloud]付きで表示されます。"
    return "iCloudフォルダにsafetensorsファイルが見つかりません。CivitAIからダウンロードしてください。"


def open_folder(path):
    if os.path.isdir(path):
        subprocess.Popen(["open", path])
        return f"フォルダを開きました: {path}"
    return f"フォルダが存在しません: {path}"


# ──────────────────────────────────────────────
# Build generation controls (shared between tabs)
# ──────────────────────────────────────────────

STYLE_PRESETS = {
    "Realistic Portrait": {
        "prompt": "photo of a beautiful young woman, professional photography, natural skin texture, soft smile, looking at camera, studio lighting, shallow depth of field, 85mm lens, sharp focus, 8k uhd",
        "negative": "worst quality, low quality, blurry, deformed, ugly, bad anatomy, bad hands, bad face, distorted face, extra fingers, watermark, text, signature",
        "steps": 25, "cfg": 7.0, "sampler": "dpmpp_2m", "scheduler": "karras",
        "width": 512, "height": 768,
    },
    "Anime Character": {
        "prompt": "masterpiece, best quality, 1girl, beautiful detailed eyes, long flowing hair, smile, colorful, detailed background, anime style, illustration",
        "negative": "worst quality, low quality, blurry, deformed, ugly, bad anatomy, extra fingers",
        "steps": 25, "cfg": 7.0, "sampler": "euler_ancestral", "scheduler": "normal",
        "width": 512, "height": 768,
    },
    "Cyberpunk": {
        "prompt": "cyberpunk city street at night, neon lights, rain, futuristic, holographic signs, dark atmosphere, cinematic, blade runner style, 8k, ultra detailed",
        "negative": "worst quality, low quality, blurry, bright, daytime, watermark",
        "steps": 30, "cfg": 8.0, "sampler": "dpmpp_sde", "scheduler": "karras",
        "width": 768, "height": 512,
    },
    "Japanese Culture": {
        "prompt": "beautiful Japanese woman wearing traditional kimono, cherry blossoms, temple garden, spring, elegant, professional photography, soft lighting, 8k uhd",
        "negative": "worst quality, low quality, blurry, deformed, ugly, bad anatomy, watermark",
        "steps": 25, "cfg": 7.0, "sampler": "dpmpp_2m", "scheduler": "karras",
        "width": 512, "height": 768,
    },
    "NSFW Realistic": {
        "prompt": "beautiful woman, detailed skin texture, natural body, intimate, bedroom, soft lighting, photorealistic, 8k uhd, professional photography",
        "negative": "worst quality, low quality, deformed, ugly, bad anatomy, bad hands, cartoon, anime, watermark",
        "steps": 30, "cfg": 7.0, "sampler": "dpmpp_2m", "scheduler": "karras",
        "width": 512, "height": 768,
    },
    "NSFW Anime": {
        "prompt": "masterpiece, best quality, 1girl, beautiful detailed body, sensual pose, detailed skin, anime style, colorful, beautiful lighting",
        "negative": "worst quality, low quality, blurry, deformed, ugly, bad anatomy, extra limbs",
        "steps": 25, "cfg": 7.0, "sampler": "euler_ancestral", "scheduler": "normal",
        "width": 512, "height": 768,
    },
    "Landscape": {
        "prompt": "stunning landscape photography, mountains, lake reflection, golden hour, dramatic sky, nature, 8k uhd, professional photography, wide angle lens",
        "negative": "worst quality, low quality, blurry, watermark, text, people, person",
        "steps": 30, "cfg": 8.0, "sampler": "dpmpp_2m", "scheduler": "karras",
        "width": 768, "height": 512,
    },
}

# QUALITY_PRESETS は adult_studio.py から import 済み (Draft/Standard/High/Ultra)


def build_gen_controls(tab_name):
    """Build the common generation controls for a tab. Returns all input components."""
    models = _get_models_for_backend()
    loras = ["None"] + get_available_loras(config["models_dir"])
    vaes = ["None"] + get_available_vaes(config["models_dir"])
    upscale_models = ["None"] + get_available_upscale_models(config["models_dir"])

    # ── Style Presets ──
    with gr.Row():
        style_preset = gr.Dropdown(
            choices=["(自由入力)"] + list(STYLE_PRESETS.keys()),
            value="(自由入力)",
            label="Style Preset",
            info="選ぶとプロンプト+設定が自動入力。あとから編集もOK。",
            scale=2,
        )
        cinema_preset = gr.Dropdown(
            choices=list(CINEMA_PRESETS.keys()),
            value="(なし)",
            label="Cinema Preset",
            info="カメラ/レンズ情報をプロンプトに自動付加（ARRI, RED, Panavision等）",
            scale=2,
        )
        quality_mode = gr.Radio(
            choices=list(QUALITY_PRESETS.keys()),
            value="Standard (標準)",
            label="Quality",
            info="Draft=高速 / Standard=通常 / High=Hires+顔補正 / Ultra=最高画質",
            scale=2,
        )

    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(label="Prompt", lines=4, placeholder="Describe what you want to generate...")
            negative = gr.Textbox(label="Negative Prompt", lines=2, value=config["default_negative_prompt"])
        with gr.Column(scale=1):
            civitai_model_names = list(CIVITAI_GENERATION_MODELS.keys())
            fal_model_names = list(FAL_MODELS.keys())
            icloud_models = get_icloud_only_models()
            backend = config.get("backend", "local")
            if backend == "fal":
                cloud_choices = fal_model_names + civitai_model_names
                default_val = fal_model_names[0]
                model_info = "fal.ai: モデルを選択 / CivitAIバックエンドならCivitAIモデル"
            elif backend == "civitai":
                # iCloudモデルを[iCloud]プレフィックス付きで追加（重複排除）
                icloud_set = set(icloud_models)
                models = [m for m in models if m not in icloud_set]
                icloud_prefixed = [f"[iCloud] {m}" for m in icloud_models]
                cloud_choices = civitai_model_names + icloud_prefixed
                default_val = civitai_model_names[0]
                model_info = "CivitAIクラウド + [iCloud]モデル対応"
            else:
                cloud_choices = civitai_model_names
                default_val = models[0] if models else (civitai_model_names[0] if civitai_model_names else None)
                model_info = "ローカルモデル or クラウドモデル"
            model = gr.Dropdown(
                choices=models + cloud_choices,
                label="Model",
                value=default_val,
                info=model_info,
            )
            lora = gr.Dropdown(choices=loras, label="LoRA", value="None", info="ローカル/iCloud LoRA。fal.aiバックエンドではCivitAI URLから自動取得")
            lora_strength = gr.Slider(0, 2, value=0.8, step=0.05, label="LoRA Strength")
            vae = gr.Dropdown(choices=vaes, label="VAE (ローカル専用)", value="None")

    with gr.Row():
        width = gr.Slider(256, 2048, value=config["default_width"], step=64, label="Width")
        height = gr.Slider(256, 2048, value=config["default_height"], step=64, label="Height")

    with gr.Accordion("Advanced Settings", open=False):
        with gr.Row():
            steps = gr.Slider(1, 100, value=config["default_steps"], step=1, label="Steps")
            cfg = gr.Slider(1, 30, value=config["default_cfg"], step=0.5, label="CFG Scale")
            seed = gr.Number(value=-1, label="Seed (-1=random)", precision=0)
            batch_size = gr.Slider(1, 8, value=1, step=1, label="Batch Size")

        with gr.Row():
            sampler = gr.Dropdown(choices=SAMPLERS, value=config["default_sampler"], label="Sampler")
            scheduler = gr.Dropdown(choices=SCHEDULERS, value=config["default_scheduler"], label="Scheduler")

    with gr.Accordion("Hires Fix (顔・ディテール改善)", open=False):
        with gr.Row():
            hires_fix = gr.Checkbox(label="Hires Fix 有効", value=False)
            hires_scale = gr.Slider(1.25, 2.0, value=1.5, step=0.25, label="アップスケール倍率")
            hires_denoise = gr.Slider(0.2, 0.7, value=0.5, step=0.05, label="Denoise (低い=元に忠実)")
            hires_steps = gr.Slider(5, 30, value=15, step=1, label="Hires Steps")
        upscale_model_dd = gr.Dropdown(
            choices=upscale_models, value="4x-UltraSharp.pth" if "4x-UltraSharp.pth" in upscale_models else "None",
            label="Upscale Model (None=Latent Upscale)",
        )

    with gr.Accordion("FaceDetailer (顔の自動修正)", open=False):
        with gr.Row():
            face_detailer = gr.Checkbox(label="FaceDetailer 有効", value=False)
            face_denoise = gr.Slider(0.1, 0.7, value=0.4, step=0.05, label="顔Denoise (低い=元に忠実)")
            face_guide_size = gr.Slider(256, 1024, value=512, step=64, label="顔ガイドサイズ")

    with gr.Row():
        generate_btn = gr.Button(f"Generate ({tab_name})", variant="primary", size="lg")
        refresh_btn = gr.Button("Refresh Models", size="sm")

    with gr.Row():
        color_grade = gr.Dropdown(
            choices=list(COLOR_GRADE_PRESETS.keys()),
            value="(なし)",
            label="Color Grade (ポストプロダクション)",
            info="生成後の画像にカラーグレーディングを適用",
            scale=1,
        )
    with gr.Row():
        gallery = gr.Gallery(label="Generated Images", columns=2, height=512)
    info = gr.Textbox(label="Info", interactive=False)

    with gr.Accordion("Refine (生成後ワンクリック高画質化)", open=False):
        gr.Markdown("生成画像を選択 → Refineで **アップスケール + 軽いdenoise + 顔補正** を一括適用。")
        with gr.Row():
            refine_denoise = gr.Slider(0.15, 0.55, value=0.35, step=0.05, label="Refine Denoise (低い=元に忠実)")
            refine_scale = gr.Slider(1.25, 2.0, value=1.5, step=0.25, label="Upscale倍率")
            refine_face = gr.Checkbox(label="顔補正", value=True)
        refine_btn = gr.Button("Refine Selected Image", variant="secondary", size="lg")

    # ── Style Preset handler ──
    def apply_style(style_name):
        if style_name == "(自由入力)" or style_name not in STYLE_PRESETS:
            return [gr.update()] * 8
        p = STYLE_PRESETS[style_name]
        return (
            gr.update(value=p["prompt"]),
            gr.update(value=p["negative"]),
            gr.update(value=p["steps"]),
            gr.update(value=p["cfg"]),
            gr.update(value=p["sampler"]),
            gr.update(value=p["scheduler"]),
            gr.update(value=p["width"]),
            gr.update(value=p["height"]),
        )

    style_preset.change(
        fn=apply_style,
        inputs=[style_preset],
        outputs=[prompt, negative, steps, cfg, sampler, scheduler, width, height],
    )

    # ── Quality Mode handler ──
    def apply_quality(mode):
        p = QUALITY_PRESETS.get(mode, QUALITY_PRESETS["Standard (標準)"])
        return (
            gr.update(value=p.get("steps", 25)),
            gr.update(value=p.get("cfg", 7.0)),
            gr.update(value=p.get("sampler", "dpmpp_2m_sde")),
            gr.update(value=p.get("scheduler", "karras")),
            gr.update(value=p.get("hires_fix", False)),
            gr.update(value=p.get("hires_scale", 1.5)),
            gr.update(value=p.get("hires_denoise", 0.5)),
            gr.update(value=p.get("hires_steps", 15)),
            gr.update(value=p.get("upscale_model", "4x-UltraSharp.pth") if p.get("hires_fix") else "None"),
            gr.update(value=p.get("face_detailer", False)),
            gr.update(value=p.get("face_denoise", 0.4)),
            gr.update(value=p.get("face_guide_size", 512)),
        )

    quality_mode.change(
        fn=apply_quality,
        inputs=[quality_mode],
        outputs=[steps, cfg, sampler, scheduler,
                 hires_fix, hires_scale, hires_denoise, hires_steps, upscale_model_dd,
                 face_detailer, face_denoise, face_guide_size],
    )

    # ── Model auto-size ──
    def auto_size_for_model(model_name):
        if not model_name:
            return gr.update(), gr.update()
        name = model_name.lower()
        if any(x in name for x in ["xl", "sdxl", "flux", "turbo"]):
            return gr.update(value=1024), gr.update(value=1024)
        return gr.update(value=512), gr.update(value=768)

    model.change(fn=auto_size_for_model, inputs=[model], outputs=[width, height])

    refresh_btn.click(
        fn=refresh_models,
        outputs=[model, lora, vae],
    )

    # ── Refine handler ──
    _tab_mode = "adult" if "Adult" in tab_name else "normal"

    def _do_refine(gallery_sel, p, neg, mdl, sd, dn, sc, fc):
        return refine_image(gallery_sel, p, neg, mdl, sd, denoise=dn, upscale_scale=sc, face_fix=fc, mode=_tab_mode)

    refine_btn.click(
        fn=_do_refine,
        inputs=[gallery, prompt, negative, model, seed, refine_denoise, refine_scale, refine_face],
        outputs=[gallery, info],
    )

    return (prompt, negative, model, lora, lora_strength, vae,
            width, height, steps, cfg, sampler, scheduler, seed, batch_size,
            hires_fix, hires_scale, hires_denoise, hires_steps, upscale_model_dd,
            face_detailer, face_denoise, face_guide_size,
            cinema_preset, color_grade,
            generate_btn, gallery, info)


# ──────────────────────────────────────────────
# UI Layout
# ──────────────────────────────────────────────

with gr.Blocks(title="AI-diffusion Studio") as app:

    gr.HTML("<h1 class='main-title'>AI-diffusion Studio</h1>")

    with gr.Row():
        status = gr.Textbox(value=check_server_status(), label="Server Status", interactive=False, elem_classes="status-bar", scale=3)
        backend_radio = gr.Radio(
            choices=["vast", "fal", "replicate", "together", "dezgo", "novita", "civitai", "local"],
            value=config.get("backend", "local"),
            label="Backend",
            info="vast=CivitAIモデル自由・無検閲 / fal=Flux / replicate=万能 / dezgo=無検閲 / novita=最安 / civitai=SDXL / local=Mac",
            scale=1,
        )
        refresh_status_btn = gr.Button("Check Server", size="sm", scale=0)

    backend_switch_info = gr.Textbox(label="接続情報", interactive=False, lines=4)

    def switch_backend_ui(choice):
        msg, url = switch_backend(choice)
        return check_server_status(), msg

    backend_radio.change(fn=switch_backend_ui, inputs=[backend_radio], outputs=[status, backend_switch_info])
    refresh_status_btn.click(fn=check_server_status, outputs=[status])

    with gr.Tabs():
        # ── Quick (Normal) Tab ──
        with gr.Tab("Quick (Normal)"):
            gr.Markdown("**手軽に画像生成** — プロンプトを入力してGenerateを押すだけ")
            (q_prompt, q_neg, q_model, q_lora, q_lora_str, q_vae,
             q_w, q_h, q_steps, q_cfg, q_sampler, q_sched, q_seed, q_batch,
             q_hires, q_hires_scale, q_hires_denoise, q_hires_steps, q_upscale_model,
             q_face_detailer, q_face_denoise, q_face_guide_size,
             q_cinema, q_color_grade,
             q_gen_btn, q_gallery, q_info) = build_gen_controls("Normal")

            q_gen_btn.click(
                fn=generate_normal,
                inputs=[q_prompt, q_neg, q_model, q_lora, q_lora_str, q_vae,
                        q_w, q_h, q_steps, q_cfg, q_sampler, q_sched, q_seed, q_batch,
                        q_hires, q_hires_scale, q_hires_denoise, q_hires_steps, q_upscale_model,
                        q_face_detailer, q_face_denoise, q_face_guide_size,
                        q_cinema, q_color_grade],
                outputs=[q_gallery, q_info],
            )

        # ── Advanced (ComfyUI) Tab ──
        with gr.Tab("Advanced (ComfyUI)"):
            gr.Markdown("**ComfyUI ネイティブUI** — フル機能のノードエディタ")
            gr.HTML(
                '<iframe src="http://127.0.0.1:8188" '
                'style="width:100%; height:800px; border:none; border-radius:8px;"></iframe>'
            )

        # ── Adult (R18) Tab ──
        with gr.Tab("Adult (R18)", elem_id="adult-tab"):
            gr.Markdown("**R18 / Adult Studio** — 制限なし。キャラビルダー・シーン・体位選択・編集・動画。出力は専用フォルダ。")

            with gr.Tabs():
                # ── Sub-tab 1: Free (既存の自由入力) ──
                with gr.Tab("Free (自由入力)"):
                    (a_prompt, a_neg, a_model, a_lora, a_lora_str, a_vae,
                     a_w, a_h, a_steps, a_cfg, a_sampler, a_sched, a_seed, a_batch,
                     a_hires, a_hires_scale, a_hires_denoise, a_hires_steps, a_upscale_model,
                     a_face_detailer, a_face_denoise, a_face_guide_size,
                     a_cinema, a_color_grade,
                     a_gen_btn, a_gallery, a_info) = build_gen_controls("Adult")

                    a_gen_btn.click(
                        fn=generate_adult,
                        inputs=[a_prompt, a_neg, a_model, a_lora, a_lora_str, a_vae,
                                a_w, a_h, a_steps, a_cfg, a_sampler, a_sched, a_seed, a_batch,
                                a_hires, a_hires_scale, a_hires_denoise, a_hires_steps, a_upscale_model,
                                a_face_detailer, a_face_denoise, a_face_guide_size,
                                a_cinema, a_color_grade],
                        outputs=[a_gallery, a_info],
                    )

                # ── Sub-tab 2: Character Builder ──
                with gr.Tab("Character Builder"):
                    gr.Markdown(
                        "**キャラクタービルダー** — 各項目を選ぶだけでプロンプト自動生成。体位・シチュエーションも選択可。"
                    )

                    with gr.Row():
                        cb_style = gr.Dropdown(choices=list(CHAR_STYLE.keys()),
                                               value=list(CHAR_STYLE.keys())[0], label="スタイル")
                        cb_people = gr.Dropdown(choices=list(CHAR_PEOPLE_COUNT.keys()),
                                                value=list(CHAR_PEOPLE_COUNT.keys())[0], label="人数")

                    with gr.Accordion("外見 (Appearance)", open=True):
                        with gr.Row():
                            cb_ethnicity = gr.Dropdown(choices=list(CHAR_ETHNICITY.keys()),
                                                       value=list(CHAR_ETHNICITY.keys())[0], label="民族")
                            cb_age = gr.Dropdown(choices=list(CHAR_AGE.keys()),
                                                  value=list(CHAR_AGE.keys())[0], label="年齢")
                            cb_body = gr.Dropdown(choices=list(CHAR_BODY_TYPE.keys()),
                                                   value=list(CHAR_BODY_TYPE.keys())[1], label="体型")
                        with gr.Row():
                            cb_breast = gr.Dropdown(choices=list(CHAR_BREAST.keys()),
                                                     value=list(CHAR_BREAST.keys())[1], label="胸")
                            cb_butt = gr.Dropdown(choices=list(CHAR_BUTT.keys()),
                                                   value=list(CHAR_BUTT.keys())[0], label="お尻")
                            cb_skin = gr.Dropdown(choices=list(CHAR_SKIN.keys()),
                                                   value=list(CHAR_SKIN.keys())[1], label="肌")

                    with gr.Accordion("髪・表情 (Hair & Expression)", open=True):
                        with gr.Row():
                            cb_hair_color = gr.Dropdown(choices=list(CHAR_HAIR_COLOR.keys()),
                                                         value=list(CHAR_HAIR_COLOR.keys())[0], label="髪色")
                            cb_hair_style = gr.Dropdown(choices=list(CHAR_HAIR_STYLE.keys()),
                                                         value=list(CHAR_HAIR_STYLE.keys())[0], label="髪型")
                            cb_expression = gr.Dropdown(choices=list(CHAR_EXPRESSION.keys()),
                                                         value=list(CHAR_EXPRESSION.keys())[1], label="表情")

                    with gr.Accordion("服装・ポーズ (Clothing & Pose)", open=True):
                        with gr.Row():
                            cb_clothing = gr.Dropdown(choices=list(CHAR_CLOTHING.keys()),
                                                       value="全裸", label="服装 / 状態")
                            cb_pose = gr.Dropdown(choices=list(CHAR_POSE.keys()),
                                                   value="立ち (正面)", label="ポーズ")

                    with gr.Accordion("体位 (Sex Position)", open=False):
                        cb_position = gr.Dropdown(choices=list(SEX_POSITIONS.keys()),
                                                   value="なし (ソロ)", label="体位")
                        gr.Markdown("*体位を選択すると人数が自動的にカップル以上になります。*")

                    with gr.Accordion("カメラ・場所 (Camera & Setting)", open=False):
                        with gr.Row():
                            cb_camera = gr.Dropdown(choices=list(CHAR_CAMERA.keys()),
                                                     value=list(CHAR_CAMERA.keys())[0], label="カメラアングル")
                            cb_setting = gr.Dropdown(choices=list(CHAR_SETTING.keys()),
                                                      value=list(CHAR_SETTING.keys())[0], label="場所")

                    cb_custom = gr.Textbox(label="追加プロンプト (自由入力)", lines=2,
                                           placeholder="追加したい要素があれば入力。例: wet skin, biting lip, looking back")
                    with gr.Row():
                        cb_seed = gr.Number(value=-1, label="Seed (-1=random)", precision=0)
                        _cb_icloud = [f"[iCloud] {m}" for m in get_icloud_only_models()]
                        cb_civitai_model = gr.Dropdown(
                            choices=["(自動)"] + list(CIVITAI_GENERATION_MODELS.keys()) + _cb_icloud,
                            value="(自動)",
                            label="CivitAI モデル / iCloud (指定時CivitAI優先)",
                        )

                    # ── Model / LoRA / VAE Selection ──
                    _cb_models = _get_models_for_backend()
                    _cb_loras = ["None"] + get_available_loras(config["models_dir"])
                    _cb_vaes = ["None", "Auto"] + get_available_vaes(config["models_dir"])
                    _cb_upscalers = ["None"] + get_available_upscale_models(config["models_dir"])

                    with gr.Accordion("モデル設定 (Model / LoRA / VAE)", open=False):
                        with gr.Row():
                            cb_model_sel = gr.Dropdown(
                                choices=["(自動)"] + _cb_models,
                                value="(自動)",
                                label="チェックポイント",
                                info="(自動)=NSFW最適モデル自動選択。ローカル全モデルから選択可。",
                            )
                            cb_vae_sel = gr.Dropdown(choices=_cb_vaes, value="None", label="VAE")
                        with gr.Row():
                            cb_lora1 = gr.Dropdown(choices=_cb_loras, value="None", label="LoRA 1")
                            cb_lora1_str = gr.Slider(0, 2, value=0.8, step=0.05, label="LoRA 1 強度")
                        with gr.Row():
                            cb_lora2 = gr.Dropdown(choices=_cb_loras, value="None", label="LoRA 2")
                            cb_lora2_str = gr.Slider(0, 2, value=0.8, step=0.05, label="LoRA 2 強度")
                        with gr.Row():
                            cb_lora3 = gr.Dropdown(choices=_cb_loras, value="None", label="LoRA 3")
                            cb_lora3_str = gr.Slider(0, 2, value=0.8, step=0.05, label="LoRA 3 強度")

                    with gr.Accordion("生成設定 (Steps / CFG / Size)", open=False):
                        with gr.Row():
                            cb_width = gr.Slider(256, 2048, value=832, step=64, label="Width")
                            cb_height = gr.Slider(256, 2048, value=1216, step=64, label="Height")
                        with gr.Row():
                            cb_steps = gr.Slider(1, 100, value=28, step=1, label="Steps")
                            cb_cfg = gr.Slider(1, 30, value=7.5, step=0.5, label="CFG Scale")
                        with gr.Row():
                            cb_sampler = gr.Dropdown(choices=SAMPLERS, value="euler_ancestral", label="Sampler")
                            cb_scheduler = gr.Dropdown(choices=SCHEDULERS, value="normal", label="Scheduler")

                    cb_quality_preset = gr.Radio(
                        choices=list(QUALITY_PRESETS.keys()),
                        value="Standard (標準)",
                        label="画質プリセット",
                        info="Draft=高速 / Standard=通常 / High=Hires+顔補正 / Ultra=最高画質",
                    )

                    with gr.Accordion("Hires Fix / FaceDetailer", open=False):
                        with gr.Row():
                            cb_hires = gr.Checkbox(label="Hires Fix", value=False)
                            cb_hires_scale = gr.Slider(1.25, 2.0, value=1.5, step=0.25, label="倍率")
                            cb_hires_denoise = gr.Slider(0.2, 0.7, value=0.5, step=0.05, label="Hires Denoise")
                            cb_hires_steps = gr.Slider(5, 30, value=15, step=1, label="Hires Steps")
                        cb_upscale_model = gr.Dropdown(
                            choices=_cb_upscalers,
                            value="4x-UltraSharp.pth" if "4x-UltraSharp.pth" in _cb_upscalers else "None",
                            label="Upscale Model",
                        )
                        with gr.Row():
                            cb_face_detailer = gr.Checkbox(label="FaceDetailer", value=False)
                            cb_face_denoise = gr.Slider(0.1, 0.7, value=0.4, step=0.05, label="顔Denoise")
                            cb_face_guide = gr.Slider(256, 1024, value=512, step=64, label="顔ガイドサイズ")

                    # Preview generated prompt
                    cb_preview = gr.Textbox(label="生成されるプロンプト (プレビュー)", lines=3, interactive=False)

                    def preview_char_prompt(style, eth, age, body, breast, butt, hc, hs, skin, expr, cloth, pose, pos, cam, sett, ppl, custom):
                        p, n = compose_character_prompt(style, eth, age, body, breast, butt, hc, hs, skin, expr, cloth, pose, pos, cam, sett, ppl, custom)
                        return f"[Prompt]\n{p}\n\n[Negative]\n{n}"

                    _cb_inputs = [cb_style, cb_ethnicity, cb_age, cb_body, cb_breast, cb_butt,
                                  cb_hair_color, cb_hair_style, cb_skin, cb_expression, cb_clothing,
                                  cb_pose, cb_position, cb_camera, cb_setting, cb_people, cb_custom]

                    for inp in _cb_inputs:
                        inp.change(fn=preview_char_prompt, inputs=_cb_inputs, outputs=[cb_preview])

                    # ── Quality preset handler for Character Builder ──
                    def _cb_apply_quality(preset_key):
                        p = QUALITY_PRESETS.get(preset_key, QUALITY_PRESETS["Standard (標準)"])
                        return (
                            gr.update(value=p.get("steps", 25)),
                            gr.update(value=p.get("cfg", 7.0)),
                            gr.update(value=p.get("sampler", "dpmpp_2m_sde")),
                            gr.update(value=p.get("scheduler", "karras")),
                            gr.update(value=p.get("hires_fix", False)),
                            gr.update(value=p.get("hires_scale", 1.5)),
                            gr.update(value=p.get("hires_denoise", 0.5)),
                            gr.update(value=p.get("hires_steps", 15)),
                            gr.update(value=p.get("upscale_model", "4x-UltraSharp.pth") if p.get("hires_fix") else "None"),
                            gr.update(value=p.get("face_detailer", False)),
                            gr.update(value=p.get("face_denoise", 0.4)),
                            gr.update(value=p.get("face_guide_size", 512)),
                        )

                    cb_quality_preset.change(
                        fn=_cb_apply_quality,
                        inputs=[cb_quality_preset],
                        outputs=[cb_steps, cb_cfg, cb_sampler, cb_scheduler,
                                 cb_hires, cb_hires_scale, cb_hires_denoise, cb_hires_steps, cb_upscale_model,
                                 cb_face_detailer, cb_face_denoise, cb_face_guide],
                    )

                    cb_gen_btn = gr.Button("Generate Character", variant="primary", size="lg")
                    cb_gallery = gr.Gallery(label="Generated", columns=2, height=512)
                    cb_info = gr.Textbox(label="Info", interactive=False)

                    with gr.Accordion("Refine (生成後ワンクリック高画質化)", open=False):
                        gr.Markdown("生成画像を選択 → Refineで **アップスケール + denoise + 顔補正** を一括適用。")
                        with gr.Row():
                            cb_refine_denoise = gr.Slider(0.15, 0.55, value=0.35, step=0.05, label="Denoise")
                            cb_refine_scale = gr.Slider(1.25, 2.0, value=1.5, step=0.25, label="Upscale倍率")
                            cb_refine_face = gr.Checkbox(label="顔補正", value=True)
                        cb_refine_btn = gr.Button("Refine Selected Image", variant="secondary", size="lg")

                    cb_gen_btn.click(
                        fn=generate_character_image,
                        inputs=[cb_style, cb_ethnicity, cb_age, cb_body, cb_breast, cb_butt,
                                cb_hair_color, cb_hair_style, cb_skin, cb_expression, cb_clothing,
                                cb_pose, cb_position, cb_camera, cb_setting, cb_people,
                                cb_custom, cb_seed, cb_civitai_model,
                                cb_model_sel, cb_lora1, cb_lora1_str,
                                cb_lora2, cb_lora2_str, cb_lora3, cb_lora3_str,
                                cb_vae_sel,
                                cb_steps, cb_cfg, cb_sampler, cb_scheduler,
                                cb_width, cb_height,
                                cb_hires, cb_hires_scale, cb_hires_denoise, cb_hires_steps, cb_upscale_model,
                                cb_face_detailer, cb_face_denoise, cb_face_guide],
                        outputs=[cb_gallery, cb_info],
                    )

                    cb_refine_btn.click(
                        fn=lambda g, p, sd, dn, us, ff: refine_image(g, p, "", None, sd, dn, us, ff, "adult"),
                        inputs=[cb_gallery, cb_custom, cb_seed, cb_refine_denoise, cb_refine_scale, cb_refine_face],
                        outputs=[cb_gallery, cb_info],
                    )

                    # Auto-apply optimal settings when model changes
                    def _cb_apply_model_settings(model_name):
                        settings = get_model_settings(model_name)
                        if not settings:
                            return [gr.update()] * 6
                        return [
                            gr.update(value=settings.get("steps", 28)),
                            gr.update(value=settings.get("cfg", 7.5)),
                            gr.update(value=settings.get("sampler", "euler_ancestral")),
                            gr.update(value=settings.get("scheduler", "normal")),
                            gr.update(value=settings.get("w", 832)),
                            gr.update(value=settings.get("h", 1216)),
                        ]

                    cb_model_sel.change(
                        fn=_cb_apply_model_settings,
                        inputs=[cb_model_sel],
                        outputs=[cb_steps, cb_cfg, cb_sampler, cb_scheduler, cb_width, cb_height],
                    )

                # ── Sub-tab 3: Scene Categories ──
                with gr.Tab("Scene (ワンクリック)"):
                    gr.Markdown(
                        "**シーンカテゴリ** — カテゴリを選んでワンクリック生成。追加プロンプトでカスタマイズ可。"
                    )

                    sc_category = gr.Dropdown(
                        choices=list(SCENE_CATEGORIES.keys()),
                        value=list(SCENE_CATEGORIES.keys())[0],
                        label="シーンカテゴリ",
                    )
                    sc_detail = gr.Markdown("")

                    def show_scene_detail(key):
                        s = SCENE_CATEGORIES.get(key, {})
                        return f"**プロンプト**: {s.get('prompt', '')}\n**スタイル**: {s.get('style', '')}"

                    sc_category.change(fn=show_scene_detail, inputs=[sc_category], outputs=[sc_detail])

                    sc_custom = gr.Textbox(label="追加プロンプト (カスタム)", lines=2,
                                           placeholder="例: japanese woman, long black hair, big breasts")
                    with gr.Row():
                        sc_seed = gr.Number(value=-1, label="Seed (-1=random)", precision=0)
                        _sc_icloud = [f"[iCloud] {m}" for m in get_icloud_only_models()]
                        sc_civitai_model = gr.Dropdown(
                            choices=["(自動)"] + list(CIVITAI_GENERATION_MODELS.keys()) + _sc_icloud,
                            value="(自動)",
                            label="CivitAI モデル / iCloud (指定時CivitAI優先)",
                        )

                    # ── Scene Model / LoRA Selection ──
                    _sc_models = _get_models_for_backend()
                    _sc_loras = ["None"] + get_available_loras(config["models_dir"])
                    _sc_vaes = ["None", "Auto"] + get_available_vaes(config["models_dir"])
                    _sc_upscalers = ["None"] + get_available_upscale_models(config["models_dir"])

                    with gr.Accordion("モデル設定 (Model / LoRA / VAE)", open=False):
                        with gr.Row():
                            sc_model_sel = gr.Dropdown(choices=["(自動)"] + _sc_models, value="(自動)", label="チェックポイント")
                            sc_vae_sel = gr.Dropdown(choices=_sc_vaes, value="None", label="VAE")
                        with gr.Row():
                            sc_lora1 = gr.Dropdown(choices=_sc_loras, value="None", label="LoRA 1")
                            sc_lora1_str = gr.Slider(0, 2, value=0.8, step=0.05, label="LoRA 1 強度")
                        with gr.Row():
                            sc_lora2 = gr.Dropdown(choices=_sc_loras, value="None", label="LoRA 2")
                            sc_lora2_str = gr.Slider(0, 2, value=0.8, step=0.05, label="LoRA 2 強度")
                        with gr.Row():
                            sc_lora3 = gr.Dropdown(choices=_sc_loras, value="None", label="LoRA 3")
                            sc_lora3_str = gr.Slider(0, 2, value=0.8, step=0.05, label="LoRA 3 強度")

                    with gr.Accordion("生成設定 (Steps / CFG / Size)", open=False):
                        with gr.Row():
                            sc_width = gr.Slider(256, 2048, value=832, step=64, label="Width")
                            sc_height = gr.Slider(256, 2048, value=1216, step=64, label="Height")
                        with gr.Row():
                            sc_steps = gr.Slider(1, 100, value=28, step=1, label="Steps")
                            sc_cfg = gr.Slider(1, 30, value=7.5, step=0.5, label="CFG Scale")
                        with gr.Row():
                            sc_sampler = gr.Dropdown(choices=SAMPLERS, value="euler_ancestral", label="Sampler")
                            sc_scheduler = gr.Dropdown(choices=SCHEDULERS, value="normal", label="Scheduler")

                    sc_quality_preset = gr.Radio(
                        choices=list(QUALITY_PRESETS.keys()),
                        value="Standard (標準)",
                        label="画質プリセット",
                        info="Draft=高速 / Standard=通常 / High=Hires+顔補正 / Ultra=最高画質",
                    )

                    with gr.Accordion("Hires Fix / FaceDetailer", open=False):
                        with gr.Row():
                            sc_hires = gr.Checkbox(label="Hires Fix", value=False)
                            sc_hires_scale = gr.Slider(1.25, 2.0, value=1.5, step=0.25, label="倍率")
                            sc_hires_denoise = gr.Slider(0.2, 0.7, value=0.5, step=0.05, label="Hires Denoise")
                            sc_hires_steps = gr.Slider(5, 30, value=15, step=1, label="Hires Steps")
                        sc_upscale_model = gr.Dropdown(
                            choices=_sc_upscalers,
                            value="4x-UltraSharp.pth" if "4x-UltraSharp.pth" in _sc_upscalers else "None",
                            label="Upscale Model",
                        )
                        with gr.Row():
                            sc_face_detailer = gr.Checkbox(label="FaceDetailer", value=False)
                            sc_face_denoise = gr.Slider(0.1, 0.7, value=0.4, step=0.05, label="顔Denoise")
                            sc_face_guide = gr.Slider(256, 1024, value=512, step=64, label="顔ガイドサイズ")

                    # ── Quality preset handler for Scene ──
                    def _sc_apply_quality(preset_key):
                        p = QUALITY_PRESETS.get(preset_key, QUALITY_PRESETS["Standard (標準)"])
                        return (
                            gr.update(value=p.get("steps", 25)),
                            gr.update(value=p.get("cfg", 7.0)),
                            gr.update(value=p.get("sampler", "dpmpp_2m_sde")),
                            gr.update(value=p.get("scheduler", "karras")),
                            gr.update(value=p.get("hires_fix", False)),
                            gr.update(value=p.get("hires_scale", 1.5)),
                            gr.update(value=p.get("hires_denoise", 0.5)),
                            gr.update(value=p.get("hires_steps", 15)),
                            gr.update(value=p.get("upscale_model", "4x-UltraSharp.pth") if p.get("hires_fix") else "None"),
                            gr.update(value=p.get("face_detailer", False)),
                            gr.update(value=p.get("face_denoise", 0.4)),
                            gr.update(value=p.get("face_guide_size", 512)),
                        )

                    sc_quality_preset.change(
                        fn=_sc_apply_quality,
                        inputs=[sc_quality_preset],
                        outputs=[sc_steps, sc_cfg, sc_sampler, sc_scheduler,
                                 sc_hires, sc_hires_scale, sc_hires_denoise, sc_hires_steps, sc_upscale_model,
                                 sc_face_detailer, sc_face_denoise, sc_face_guide],
                    )

                    sc_gen_btn = gr.Button("Generate Scene", variant="primary", size="lg")
                    sc_gallery = gr.Gallery(label="Generated", columns=2, height=512)
                    sc_info = gr.Textbox(label="Info", interactive=False)

                    with gr.Accordion("Refine (生成後ワンクリック高画質化)", open=False):
                        gr.Markdown("生成画像を選択 → Refineで **アップスケール + denoise + 顔補正** を一括適用。")
                        with gr.Row():
                            sc_refine_denoise = gr.Slider(0.15, 0.55, value=0.35, step=0.05, label="Denoise")
                            sc_refine_scale = gr.Slider(1.25, 2.0, value=1.5, step=0.25, label="Upscale倍率")
                            sc_refine_face = gr.Checkbox(label="顔補正", value=True)
                        sc_refine_btn = gr.Button("Refine Selected Image", variant="secondary", size="lg")

                    sc_gen_btn.click(
                        fn=generate_scene_category,
                        inputs=[sc_category, sc_custom, sc_seed, sc_civitai_model,
                                sc_model_sel, sc_lora1, sc_lora1_str,
                                sc_lora2, sc_lora2_str, sc_lora3, sc_lora3_str,
                                sc_vae_sel,
                                sc_steps, sc_cfg, sc_sampler, sc_scheduler,
                                sc_width, sc_height,
                                sc_hires, sc_hires_scale, sc_hires_denoise, sc_hires_steps, sc_upscale_model,
                                sc_face_detailer, sc_face_denoise, sc_face_guide],
                        outputs=[sc_gallery, sc_info],
                    )

                    sc_refine_btn.click(
                        fn=lambda g, p, sd, dn, us, ff: refine_image(g, p, "", None, sd, dn, us, ff, "adult"),
                        inputs=[sc_gallery, sc_custom, sc_seed, sc_refine_denoise, sc_refine_scale, sc_refine_face],
                        outputs=[sc_gallery, sc_info],
                    )

                    # Auto-apply optimal settings when model changes
                    def _sc_apply_model_settings(model_name):
                        settings = get_model_settings(model_name)
                        if not settings:
                            return [gr.update()] * 6
                        return [
                            gr.update(value=settings.get("steps", 28)),
                            gr.update(value=settings.get("cfg", 7.5)),
                            gr.update(value=settings.get("sampler", "euler_ancestral")),
                            gr.update(value=settings.get("scheduler", "normal")),
                            gr.update(value=settings.get("w", 832)),
                            gr.update(value=settings.get("h", 1216)),
                        ]

                    sc_model_sel.change(
                        fn=_sc_apply_model_settings,
                        inputs=[sc_model_sel],
                        outputs=[sc_steps, sc_cfg, sc_sampler, sc_scheduler, sc_width, sc_height],
                    )

                # ── Sub-tab 4: Undress / Edit ──
                with gr.Tab("Undress / 衣服編集"):
                    gr.Markdown(
                        "**Undress / 衣服編集** — 画像をアップロード → **服の部分をブラシで塗る（マスク）** → 編集モード選択 → 生成\n\n"
                        "**マスクあり（推奨）**: 塗った部分だけ再生成。顔・体型・背景はそのまま保持。\n"
                        "**マスクなし**: img2img で全体を再生成（元画像の保持度はdenoiseで調整）。\n\n"
                        "ComfyUI Inpainting (Vast.ai) で完全無検閲。"
                    )

                    with gr.Row():
                        ud_image = gr.ImageEditor(
                            label="画像をアップロード → ブラシで服の部分を塗る",
                            type="numpy",
                            brush=gr.Brush(colors=["#FFFFFF"], default_size=30),
                            height=512,
                        )
                        ud_result = gr.Gallery(label="結果", columns=1, height=512)

                    ud_mode = gr.Dropdown(
                        choices=list(UNDRESS_MODES.keys()),
                        value=list(UNDRESS_MODES.keys())[0],
                        label="編集モード",
                    )
                    ud_detail = gr.Markdown("")

                    def show_undress_detail(key):
                        m = UNDRESS_MODES.get(key, {})
                        return f"**プロンプト**: {m.get('prompt', '')}\n**強度**: {m.get('strength', 0.7)}"

                    ud_mode.change(fn=show_undress_detail, inputs=[ud_mode], outputs=[ud_detail])

                    ud_custom = gr.Textbox(label="追加プロンプト", lines=1,
                                           placeholder="例: red lace lingerie, black stockings")
                    with gr.Row():
                        ud_seed = gr.Number(value=-1, label="Seed (-1=random)", precision=0)
                        ud_denoise = gr.Slider(0.2, 1.0, value=0.75, step=0.05,
                                               label="Denoise強度 (低い=元画像保持、高い=大きく変更)")

                    # ── Undress Model Selection ──
                    _ud_models = _get_models_for_backend()
                    _ud_loras = ["None"] + get_available_loras(config["models_dir"])

                    with gr.Accordion("モデル設定", open=False):
                        with gr.Row():
                            ud_model_sel = gr.Dropdown(choices=["(自動)"] + _ud_models, value="(自動)", label="チェックポイント")
                        with gr.Row():
                            ud_lora1 = gr.Dropdown(choices=_ud_loras, value="None", label="LoRA")
                            ud_lora1_str = gr.Slider(0, 2, value=0.8, step=0.05, label="LoRA 強度")

                    with gr.Row():
                        ud_gen_btn = gr.Button("Edit / Undress (画像)", variant="primary", size="lg")
                        ud_vid_btn = gr.Button("Undress動画 / GIF生成", variant="secondary", size="lg")

                    ud_info = gr.Textbox(label="Info", interactive=False)

                    with gr.Accordion("動画設定", open=False):
                        ud_duration = gr.Slider(3, 10, value=5, step=1, label="動画の長さ (秒)")
                        ud_video_out = gr.Video(label="Undress動画", height=400)
                        gr.Markdown(
                            "*画像をアップロード → 編集モード選択 → 「Undress動画」で脱衣アニメーション生成*\n"
                            "*fal.ai Wan 2.6 (NSFW OK) を使用。*"
                        )

                    ud_gen_btn.click(
                        fn=generate_undress_edit,
                        inputs=[ud_image, ud_mode, ud_custom, ud_seed, ud_denoise,
                                ud_model_sel, ud_lora1, ud_lora1_str],
                        outputs=[ud_result, ud_info],
                    )

                    ud_vid_btn.click(
                        fn=generate_undress_video,
                        inputs=[ud_image, ud_mode, ud_custom, ud_duration],
                        outputs=[ud_video_out, ud_info],
                    )

                # ── Sub-tab 5: Adult Video ──
                with gr.Tab("Adult Video"):
                    gr.Markdown(
                        "**アダルト動画生成** — シーンプリセットから動画生成。画像アップロードでimg2vidも対応。\n\n"
                        "fal.ai Wan 2.6 (NSFW OK) を使用。"
                    )

                    av_scene = gr.Dropdown(
                        choices=list(ADULT_VIDEO_SCENES.keys()),
                        value=list(ADULT_VIDEO_SCENES.keys())[0],
                        label="動画シーン",
                    )
                    av_detail = gr.Markdown("")

                    def show_av_detail(key):
                        s = ADULT_VIDEO_SCENES.get(key, {})
                        return f"**プロンプト**: {s.get('prompt', '')}\n**長さ**: {s.get('duration', 5)}秒"

                    av_scene.change(fn=show_av_detail, inputs=[av_scene], outputs=[av_detail])

                    av_custom = gr.Textbox(label="追加プロンプト", lines=2,
                                           placeholder="例: japanese woman, bedroom, passionate")
                    av_image = gr.Image(label="参照画像 (img2vid / 省略可)", type="numpy")
                    av_seed = gr.Number(value=-1, label="Seed (-1=random)", precision=0)

                    # ── Adult Video Model Selection ──
                    _av_models = _get_models_for_backend()
                    _av_loras = ["None"] + get_available_loras(config["models_dir"])

                    with gr.Accordion("モデル設定 (AnimateDiff用)", open=False):
                        with gr.Row():
                            av_model_sel = gr.Dropdown(choices=["(自動)"] + _av_models, value="(自動)",
                                                        label="チェックポイント (AnimateDiff用、SD1.5のみ)")
                        with gr.Row():
                            av_lora1 = gr.Dropdown(choices=_av_loras, value="None", label="LoRA")
                            av_lora1_str = gr.Slider(0, 2, value=0.8, step=0.05, label="LoRA 強度")
                        gr.Markdown("*AnimateDiffはSD1.5モデルのみ対応。SDXLモデルでは動作しません。*")

                    av_gen_btn = gr.Button("Generate Adult Video", variant="primary", size="lg")
                    av_video = gr.Video(label="Generated Video", height=400)
                    av_info = gr.Textbox(label="Info", interactive=False)

                    av_gen_btn.click(
                        fn=generate_adult_video,
                        inputs=[av_scene, av_custom, av_image, av_seed,
                                av_model_sel, av_lora1, av_lora1_str],
                        outputs=[av_video, av_info],
                    )

                # ── Sub-tab 5.5: Wan 2.2 NSFW Lightning (専用バックエンド) ──
                with gr.Tab("🔥 Wan 2.2 NSFW Lightning"):
                    gr.Markdown(
                        "**Wan 2.2 Enhanced NSFW (SVI Lightning Edition)** — 完全無検閲、カメラプロンプト対応の最新動画モデル。\n\n"
                        "• **2-stage MoE**: HIGH-noise → LOW-noise GGUF pair\n"
                        "• **Lightning**: 4 steps で生成（通常の5倍速）\n"
                        "• **専用バックエンド**: RTX 3090 24GB (vast.ai) — 重いので通常生成とは別\n"
                        "• **T2V / I2V** 両対応\n"
                        "• 1回あたり約30秒〜2分（5秒動画）\n\n"
                        "⚠️ セットアップ中の場合は Settings → `wan22_comfyui_url` を確認"
                    )

                    with gr.Tabs():
                        # ── Wan 2.2 T2V ──
                        with gr.Tab("T2V (テキスト→動画)"):
                            gr.Markdown("プロンプトから完全無検閲動画を生成。camera prompt対応（slow push-in, pan, orbit 等）")

                            with gr.Row():
                                with gr.Column(scale=2):
                                    w22_t_prompt = gr.Textbox(
                                        label="Prompt (英語推奨)",
                                        lines=4,
                                        placeholder="例: beautiful woman, undressing slowly, soft lighting, slow cinematic push-in camera, intimate",
                                    )
                                    w22_t_neg = gr.Textbox(
                                        label="Negative Prompt",
                                        lines=2,
                                        value="worst quality, blurry, static, jittery, distorted, deformed anatomy, bad hands",
                                    )
                                with gr.Column(scale=1):
                                    w22_t_duration = gr.Slider(
                                        1, 15, value=5, step=1,
                                        label="Duration (秒) / 動画の長さ",
                                        info="Wan2.2は4N+1フレーム構成 (16fps)",
                                    )
                                    w22_t_w = gr.Slider(512, 1280, value=832, step=16, label="Width")
                                    w22_t_h = gr.Slider(384, 720, value=480, step=16, label="Height")
                                    w22_t_steps = gr.Slider(2, 8, value=4, step=1, label="Steps (Lightning推奨=4)")
                                    w22_t_cfg = gr.Slider(1.0, 7.0, value=1.0, step=0.5, label="CFG (Lightning推奨=1.0)")
                                    w22_t_seed = gr.Number(value=-1, label="Seed (-1=random)", precision=0)
                                    w22_t_mode = gr.Radio(choices=["normal", "adult"], value="adult", label="Save to")

                            w22_t_btn = gr.Button("🎬 Wan 2.2 T2V 生成", variant="primary", size="lg")
                            with gr.Row():
                                w22_t_video = gr.Video(label="Generated Video", height=450)
                                w22_t_info = gr.Textbox(label="Info", interactive=False, lines=10)

                            w22_t_btn.click(
                                fn=generate_wan22_t2v,
                                inputs=[w22_t_prompt, w22_t_neg, w22_t_w, w22_t_h,
                                        w22_t_duration, w22_t_steps, w22_t_cfg, w22_t_seed, w22_t_mode],
                                outputs=[w22_t_video, w22_t_info],
                            )

                        # ── Wan 2.2 I2V ──
                        with gr.Tab("I2V (画像→動画)"):
                            gr.Markdown("画像を元に完全無検閲アニメーション生成。元画像を保持しつつ動かす。")

                            with gr.Row():
                                with gr.Column(scale=2):
                                    w22_i_image = gr.Image(label="Input Image", type="numpy", height=400)
                                    w22_i_prompt = gr.Textbox(
                                        label="Motion Prompt (optional, 英語推奨)",
                                        lines=3,
                                        placeholder="例: gentle body movement, breathing, slow push-in camera, intimate atmosphere",
                                    )
                                    w22_i_neg = gr.Textbox(
                                        label="Negative Prompt",
                                        lines=2,
                                        value="worst quality, blurry, static, jittery, distorted",
                                    )
                                with gr.Column(scale=1):
                                    w22_i_duration = gr.Slider(
                                        1, 15, value=5, step=1,
                                        label="Duration (秒) / 動画の長さ",
                                    )
                                    w22_i_w = gr.Slider(512, 1280, value=832, step=16, label="Width")
                                    w22_i_h = gr.Slider(384, 720, value=480, step=16, label="Height")
                                    w22_i_steps = gr.Slider(2, 8, value=4, step=1, label="Steps")
                                    w22_i_cfg = gr.Slider(1.0, 7.0, value=1.0, step=0.5, label="CFG")
                                    w22_i_seed = gr.Number(value=-1, label="Seed (-1=random)", precision=0)
                                    w22_i_mode = gr.Radio(choices=["normal", "adult"], value="adult", label="Save to")

                            w22_i_btn = gr.Button("🎬 Wan 2.2 I2V 生成", variant="primary", size="lg")
                            with gr.Row():
                                w22_i_video = gr.Video(label="Generated Video", height=450)
                                w22_i_info = gr.Textbox(label="Info", interactive=False, lines=10)

                            w22_i_btn.click(
                                fn=generate_wan22_i2v,
                                inputs=[w22_i_image, w22_i_prompt, w22_i_neg, w22_i_w, w22_i_h,
                                        w22_i_duration, w22_i_steps, w22_i_cfg, w22_i_seed, w22_i_mode],
                                outputs=[w22_i_video, w22_i_info],
                            )

                # ── Sub-tab 6: ControlNet (Pose/Depth/Lineart) ──
                with gr.Tab("ControlNet (ポーズ指定)"):
                    gr.Markdown(
                        "**ControlNet** — ポーズ画像・深度マップ・線画をアップロード → その構図でNSFW画像生成。\n"
                        "OpenPose棒人間、深度マップ、線画いずれかを使用。"
                    )
                    _cn_models = _get_models_for_backend()
                    _cn_loras = ["None"] + get_available_loras(config["models_dir"])

                    with gr.Row():
                        cn_image = gr.Image(label="コントロール画像 (ポーズ/深度/線画)", type="numpy", height=400)
                        cn_result = gr.Gallery(label="結果", columns=1, height=400)

                    with gr.Row():
                        cn_type = gr.Radio(["openpose", "depth", "lineart"], value="openpose", label="コントロールタイプ")
                        cn_strength = gr.Slider(0.1, 2.0, value=1.0, step=0.05, label="ControlNet強度")

                    cn_prompt = gr.Textbox(label="プロンプト", lines=3, placeholder="ポーズに合わせたNSFWプロンプト")
                    cn_negative = gr.Textbox(label="ネガティブ", lines=1, value="worst quality, low quality, deformed")

                    with gr.Accordion("モデル設定", open=False):
                        cn_model_sel = gr.Dropdown(choices=["(自動)"] + _cn_models, value="(自動)", label="チェックポイント")
                        with gr.Row():
                            cn_lora = gr.Dropdown(choices=_cn_loras, value="None", label="LoRA")
                            cn_lora_str = gr.Slider(0, 2, value=0.8, step=0.05, label="LoRA強度")
                        with gr.Row():
                            cn_width = gr.Slider(256, 2048, value=512, step=64, label="Width")
                            cn_height = gr.Slider(256, 2048, value=768, step=64, label="Height")
                        with gr.Row():
                            cn_steps = gr.Slider(1, 100, value=28, step=1, label="Steps")
                            cn_cfg = gr.Slider(1, 30, value=7, step=0.5, label="CFG")
                        cn_seed = gr.Number(value=-1, label="Seed", precision=0)

                    cn_gen_btn = gr.Button("Generate with ControlNet", variant="primary", size="lg")
                    cn_info = gr.Textbox(label="Info", interactive=False)

                    def generate_controlnet_image(image, ctrl_type, strength, prompt_text, neg, model, lora, lora_s, w, h, steps_v, cfg_v, seed_v):
                        if image is None:
                            raise gr.Error("コントロール画像をアップロードしてください。")
                        if not prompt_text.strip():
                            raise gr.Error("プロンプトを入力してください。")
                        if not client.is_server_running():
                            raise gr.Error("ComfyUIが起動していません。")

                        from PIL import Image as PILImage
                        import numpy as np
                        img_pil = PILImage.fromarray(image) if isinstance(image, np.ndarray) else image
                        ts = int(time.time())
                        img_path = os.path.join(config["output_dir_adult"], f"cn_input_{ts}.png")
                        img_pil.save(img_path)
                        img_filename = f"cn_input_{ts}.png"
                        client.upload_image(img_path, img_filename)

                        models = client.get_models() or _get_models_for_backend()
                        chosen = model if model and model != "(自動)" else _select_nsfw_model(models)
                        loras_list = [(lora, float(lora_s))] if lora and lora != "None" else []
                        s = int(seed_v) if seed_v and seed_v >= 0 else -1

                        workflow = build_controlnet_workflow(
                            prompt_text, neg, chosen, img_filename, ctrl_type, float(strength),
                            int(w), int(h), int(steps_v), float(cfg_v), "euler_ancestral", "normal", s, loras=loras_list,
                        )
                        result = client.generate(workflow)
                        if result:
                            saved = [save_image_to_dir(img, config["output_dir_adult"], prefix="controlnet") for img in result]
                            return result, f"[ControlNet {ctrl_type}] Model: {chosen}\n保存先: {', '.join(saved)}"
                        raise gr.Error("ControlNet生成に失敗しました。")

                    cn_gen_btn.click(
                        fn=generate_controlnet_image,
                        inputs=[cn_image, cn_type, cn_strength, cn_prompt, cn_negative,
                                cn_model_sel, cn_lora, cn_lora_str, cn_width, cn_height, cn_steps, cn_cfg, cn_seed],
                        outputs=[cn_result, cn_info],
                    )

                # ── Sub-tab 7: IP-Adapter (顔・雰囲気保持) ──
                with gr.Tab("IP-Adapter (顔保持)"):
                    gr.Markdown(
                        "**IP-Adapter** — 参照画像の顔・雰囲気を保持したまま新しい画像を生成。\n"
                        "同じ人物で異なるポーズ・衣装・シチュエーションを作れます。"
                    )
                    _ip_models = _get_models_for_backend()
                    _ip_loras = ["None"] + get_available_loras(config["models_dir"])

                    with gr.Row():
                        ip_image = gr.Image(label="参照画像 (この顔/雰囲気を保持)", type="numpy", height=400)
                        ip_result = gr.Gallery(label="結果", columns=1, height=400)

                    ip_weight = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="IP-Adapter強度 (高い=参照に忠実)")
                    ip_prompt = gr.Textbox(label="プロンプト (新しいポーズ・状況)", lines=3,
                                           placeholder="同じ人物で: different pose, wearing bikini, on the beach")
                    ip_negative = gr.Textbox(label="ネガティブ", lines=1, value="worst quality, low quality, deformed")

                    with gr.Accordion("モデル設定", open=False):
                        ip_model_sel = gr.Dropdown(choices=["(自動)"] + _ip_models, value="(自動)", label="チェックポイント")
                        with gr.Row():
                            ip_lora = gr.Dropdown(choices=_ip_loras, value="None", label="LoRA")
                            ip_lora_str = gr.Slider(0, 2, value=0.8, step=0.05, label="LoRA強度")
                        with gr.Row():
                            ip_width = gr.Slider(256, 2048, value=512, step=64, label="Width")
                            ip_height = gr.Slider(256, 2048, value=768, step=64, label="Height")
                        with gr.Row():
                            ip_steps = gr.Slider(1, 100, value=28, step=1, label="Steps")
                            ip_cfg = gr.Slider(1, 30, value=7, step=0.5, label="CFG")
                        ip_seed = gr.Number(value=-1, label="Seed", precision=0)

                    ip_gen_btn = gr.Button("Generate with IP-Adapter", variant="primary", size="lg")
                    ip_info = gr.Textbox(label="Info", interactive=False)

                    def generate_ipadapter_image(image, weight, prompt_text, neg, model, lora, lora_s, w, h, steps_v, cfg_v, seed_v):
                        if image is None:
                            raise gr.Error("参照画像をアップロードしてください。")
                        if not prompt_text.strip():
                            raise gr.Error("プロンプトを入力してください。")
                        if not client.is_server_running():
                            raise gr.Error("ComfyUIが起動していません。")

                        from PIL import Image as PILImage
                        import numpy as np
                        img_pil = PILImage.fromarray(image) if isinstance(image, np.ndarray) else image
                        ts = int(time.time())
                        img_path = os.path.join(config["output_dir_adult"], f"ip_ref_{ts}.png")
                        img_pil.save(img_path)
                        img_filename = f"ip_ref_{ts}.png"
                        client.upload_image(img_path, img_filename)

                        models = client.get_models() or _get_models_for_backend()
                        chosen = model if model and model != "(自動)" else _select_nsfw_model(models)
                        loras_list = [(lora, float(lora_s))] if lora and lora != "None" else []
                        s = int(seed_v) if seed_v and seed_v >= 0 else -1

                        workflow = build_ipadapter_workflow(
                            prompt_text, neg, chosen, img_filename, float(weight),
                            int(w), int(h), int(steps_v), float(cfg_v), "euler_ancestral", "normal", s, loras=loras_list,
                        )
                        result = client.generate(workflow)
                        if result:
                            saved = [save_image_to_dir(img, config["output_dir_adult"], prefix="ipadapter") for img in result]
                            return result, f"[IP-Adapter] Model: {chosen}, Weight: {weight}\n保存先: {', '.join(saved)}"
                        raise gr.Error("IP-Adapter生成に失敗しました。")

                    ip_gen_btn.click(
                        fn=generate_ipadapter_image,
                        inputs=[ip_image, ip_weight, ip_prompt, ip_negative,
                                ip_model_sel, ip_lora, ip_lora_str, ip_width, ip_height, ip_steps, ip_cfg, ip_seed],
                        outputs=[ip_result, ip_info],
                    )

        # ── Video Tab ──
        with gr.Tab("Video"):
            _is_cloud = config.get("backend", "local") != "local"
            gr.Markdown(
                "**動画生成** — クラウド (fal.ai) またはローカル (AnimateDiff)。\n"
                "クラウド: 高速・NSFW OK (Kling, Wan, LTX等)。ローカル: SD1.5 + AnimateDiff。"
                if _is_cloud else
                "**動画生成** — AnimateDiff で txt2vid / img2vid / vid2vid。SD1.5モデル対応。\n"
                "クラウド利用はSettingsでbackendをfal等に切り替えてください。"
            )

            # GPU recommendation warning
            video_gpu_warning = gr.Markdown(
                "**GPU推奨**: 動画生成はRunPod (Cloud GPU) での利用を推奨します。"
                "ローカルMacでは1本30-40分かかります。"
                if config.get("backend") == "local" else
                "**クラウドモード**: fal.aiで高速動画生成。モデルを選択してください。"
            )

            models = _get_models_for_backend()
            loras = ["None"] + get_available_loras(config["models_dir"])
            vaes = ["None"] + get_available_vaes(config["models_dir"])
            motion_models = get_available_motion_models()

            VIDEO_PRESETS = {
                "Quick Test": {"steps": 10, "cfg": 7.0, "frames": 8, "fps": 8, "w": 384, "h": 384, "sampler": "euler_ancestral"},
                "Standard": {"steps": 20, "cfg": 7.5, "frames": 16, "fps": 8, "w": 512, "h": 512, "sampler": "euler_ancestral"},
                "High Quality": {"steps": 28, "cfg": 7.5, "frames": 16, "fps": 12, "w": 512, "h": 768, "sampler": "dpmpp_2m"},
                "Long (24f)": {"steps": 20, "cfg": 7.5, "frames": 24, "fps": 12, "w": 512, "h": 512, "sampler": "euler_ancestral"},
            }

            with gr.Tabs():
                # ─ txt2vid ─
                with gr.Tab("Text to Video (txt2vid)"):
                    gr.Markdown(
                        "テキストから動画を生成。クラウド時はfal.aiモデルを選択、ローカル時はSD1.5チェックポイント使用。"
                    )

                    # Cloud model selector (shown when using cloud backend)
                    v_cloud_model = gr.Dropdown(
                        choices=list(FAL_VIDEO_MODELS.keys()),
                        value=list(FAL_VIDEO_MODELS.keys())[0],
                        label="Cloud動画モデル (fal.ai)",
                        info="クラウドバックエンド時に使用。Kling=最高品質 / LTX=高速安い / Veo3.1=Google最新",
                        visible=_is_cloud,
                    )

                    v_quality = gr.Radio(
                        choices=list(VIDEO_PRESETS.keys()),
                        value="Standard",
                        label="Quality Preset",
                        info="Quick=速い / Standard=標準 / HQ=高品質 / Long=長尺",
                    )

                    with gr.Row():
                        with gr.Column(scale=2):
                            v_prompt = gr.Textbox(label="Prompt", lines=3, placeholder="1girl, walking, wind blowing hair, outdoor, sunny day")
                            v_neg = gr.Textbox(label="Negative Prompt", lines=2, value=config["default_negative_prompt"])
                        with gr.Column(scale=1):
                            v_model = gr.Dropdown(choices=models, label="Checkpoint (SD1.5推奨)", value=models[0] if models else None)
                            v_motion = gr.Dropdown(choices=motion_models, label="Motion Model", value=motion_models[0] if motion_models else None)
                            v_lora = gr.Dropdown(choices=loras, label="LoRA", value="None")
                            v_lora_str = gr.Slider(0, 2, value=0.8, step=0.05, label="LoRA Strength")
                            v_vae = gr.Dropdown(choices=vaes, label="VAE", value="None")

                    with gr.Row():
                        v_w = gr.Slider(256, 1024, value=512, step=64, label="Width")
                        v_h = gr.Slider(256, 1024, value=512, step=64, label="Height")
                        v_frames = gr.Slider(8, 32, value=16, step=1, label="Frames (ローカル時)")
                        v_fps = gr.Slider(4, 30, value=8, step=1, label="FPS (ローカル時)")

                    with gr.Row():
                        v_duration = gr.Slider(
                            2, 20, value=5, step=1,
                            label="🎬 Duration / 動画の長さ (秒)",
                            info="クラウド時=API送信値 / ローカル時=frames=duration×fps で自動計算",
                        )

                    with gr.Row():
                        v_steps = gr.Slider(1, 50, value=20, step=1, label="Steps")
                        v_cfg = gr.Slider(1, 20, value=7.5, step=0.5, label="CFG")
                        v_seed = gr.Number(value=-1, label="Seed (-1=random)", precision=0)

                    with gr.Row():
                        v_sampler = gr.Dropdown(choices=SAMPLERS, value="euler_ancestral", label="Sampler")
                        v_sched = gr.Dropdown(choices=SCHEDULERS, value="normal", label="Scheduler")
                        v_format = gr.Dropdown(choices=["gif", "mp4", "webp"], value="gif", label="Output Format")
                        v_mode = gr.Radio(choices=["normal", "adult"], value="normal", label="Save to")

                    def apply_video_preset(preset):
                        p = VIDEO_PRESETS.get(preset, VIDEO_PRESETS["Standard"])
                        return (
                            gr.update(value=p["steps"]),
                            gr.update(value=p["cfg"]),
                            gr.update(value=p["frames"]),
                            gr.update(value=p["fps"]),
                            gr.update(value=p["w"]),
                            gr.update(value=p["h"]),
                            gr.update(value=p["sampler"]),
                        )

                    v_quality.change(
                        fn=apply_video_preset,
                        inputs=[v_quality],
                        outputs=[v_steps, v_cfg, v_frames, v_fps, v_w, v_h, v_sampler],
                    )

                    v_gen_btn = gr.Button("Generate Video", variant="primary", size="lg")

                    with gr.Row():
                        v_video = gr.Video(label="Generated Video", height=400)
                        v_preview = gr.Gallery(label="Preview Frames", columns=4, height=256)
                    v_info = gr.Textbox(label="Info", interactive=False)

                    v_gen_btn.click(
                        fn=generate_video_txt2vid,
                        inputs=[v_prompt, v_neg, v_model, v_motion, v_lora, v_lora_str, v_vae,
                                v_w, v_h, v_steps, v_cfg, v_sampler, v_sched, v_seed,
                                v_frames, v_fps, v_format, v_mode, v_cloud_model, v_duration],
                        outputs=[v_video, v_preview, v_info],
                    )

                # ─ img2vid ─
                with gr.Tab("Image to Video (img2vid)"):
                    gr.Markdown(
                        "画像をアニメーションに変換。クラウド時はfal.ai (Kling/Wan/LTX)、ローカル時はAnimateDiff。"
                    )

                    i2v_cloud_model = gr.Dropdown(
                        choices=list(FAL_IMG2VID_MODELS.keys()),
                        value=list(FAL_IMG2VID_MODELS.keys())[0],
                        label="Cloud img2vidモデル (fal.ai)",
                        info="Kling=最高品質 / Wan=高品質 / LTX=高速安い",
                        visible=_is_cloud,
                    )

                    with gr.Row():
                        with gr.Column(scale=2):
                            i2v_image = gr.Image(label="Input Image", type="numpy")
                            i2v_prompt = gr.Textbox(label="Motion Prompt (optional)", lines=2, placeholder="gentle wind, hair moving, blinking")
                            i2v_neg = gr.Textbox(label="Negative Prompt", lines=1, value="worst quality, static, blurry, distorted")
                        with gr.Column(scale=1):
                            i2v_model = gr.Dropdown(choices=models, label="Checkpoint (SD1.5推奨)", value=models[0] if models else None)
                            i2v_motion = gr.Dropdown(choices=motion_models, label="Motion Model", value=motion_models[0] if motion_models else None)
                            i2v_vae = gr.Dropdown(choices=vaes, label="VAE", value="None")
                            i2v_denoise = gr.Slider(0.1, 1.0, value=0.65, step=0.05, label="Denoise (低い=元画像に忠実)")

                    with gr.Row():
                        i2v_w = gr.Slider(256, 1024, value=512, step=64, label="Width")
                        i2v_h = gr.Slider(256, 1024, value=512, step=64, label="Height")
                        i2v_frames = gr.Slider(8, 32, value=16, step=1, label="Frames (ローカル時)")
                        i2v_fps = gr.Slider(4, 30, value=8, step=1, label="FPS (ローカル時)")

                    with gr.Row():
                        i2v_duration = gr.Slider(
                            2, 20, value=5, step=1,
                            label="🎬 Duration / 動画の長さ (秒)",
                            info="クラウド時=API送信値 / ローカル時=frames=duration×fps で自動計算",
                        )

                    with gr.Row():
                        i2v_steps = gr.Slider(1, 50, value=20, step=1, label="Steps")
                        i2v_cfg = gr.Slider(1, 20, value=7.5, step=0.5, label="CFG")
                        i2v_seed = gr.Number(value=-1, label="Seed", precision=0)

                    with gr.Row():
                        i2v_sampler = gr.Dropdown(choices=SAMPLERS, value="euler_ancestral", label="Sampler")
                        i2v_sched = gr.Dropdown(choices=SCHEDULERS, value="normal", label="Scheduler")
                        i2v_format = gr.Dropdown(choices=["gif", "mp4", "webp"], value="gif", label="Output Format")
                        i2v_mode = gr.Radio(choices=["normal", "adult"], value="normal", label="Save to")

                    i2v_gen_btn = gr.Button("Generate Video from Image", variant="primary", size="lg")

                    with gr.Row():
                        i2v_video = gr.Video(label="Generated Video", height=400)
                        i2v_preview = gr.Gallery(label="Preview Frames", columns=4, height=256)
                    i2v_info = gr.Textbox(label="Info", interactive=False)

                    i2v_gen_btn.click(
                        fn=generate_video_img2vid,
                        inputs=[i2v_image, i2v_prompt, i2v_neg, i2v_model, i2v_motion, i2v_vae,
                                i2v_w, i2v_h, i2v_steps, i2v_cfg, i2v_sampler, i2v_sched, i2v_seed,
                                i2v_frames, i2v_fps, i2v_denoise, i2v_format, i2v_mode, i2v_cloud_model, i2v_duration],
                        outputs=[i2v_video, i2v_preview, i2v_info],
                    )

                # ─ Image to Video PRO (VLM-assisted) ─
                with gr.Tab("🪄 img2vid Pro (VLM分析)"):
                    gr.Markdown(
                        "**画像→動画 Pro** — AIが画像を分析して最適な動きを生成。\n\n"
                        "- **Preserve (保持)**: 元画像を保ったまま自然にアニメーション化（Canva/Firefly/Kling風）\n"
                        "- **Inspired (連想)**: 画像の内容を分析して新しい動画を生成（別解釈・スタイル変更OK）\n\n"
                        "⚠️ 画像分析には OpenAI または Anthropic API Key が必要です（Settingsで設定）"
                    )

                    with gr.Tabs():
                        # ── Preserve Mode ──
                        with gr.Tab("① Preserve (保持してアニメ化)"):
                            gr.Markdown("画像をそのまま保ちつつ、自然な動きを付ける。Kling/Wan/Veo/Sora/LTX対応。")

                            with gr.Row():
                                with gr.Column(scale=2):
                                    pres_image = gr.Image(label="入力画像", type="numpy", height=380)
                                    pres_prompt = gr.Textbox(
                                        label="Motion Prompt (空欄OK — AI自動生成)",
                                        lines=3,
                                        placeholder="例: soft breeze moving her hair, subtle smile, slow push-in (空欄で自動分析)",
                                    )
                                    with gr.Row():
                                        pres_analyze_btn = gr.Button("🔍 AI画像分析", variant="secondary", size="sm")
                                        pres_auto_analyze = gr.Checkbox(value=True, label="空欄時に自動分析", scale=0)
                                with gr.Column(scale=1):
                                    pres_model = gr.Dropdown(
                                        choices=list(FAL_IMG2VID_MODELS.keys()),
                                        value=list(FAL_IMG2VID_MODELS.keys())[0],
                                        label="img2vid モデル",
                                        info="Kling=最高品質 / Wan=NSFW OK / Veo/Sora=映画品質 / LTX=高速安い",
                                    )
                                    pres_preset = gr.Dropdown(
                                        choices=list(MOTION_PRESETS.keys()),
                                        value="(なし)",
                                        label="Motion Preset (プロンプトに追加)",
                                        info="微細/ズーム/パン/オービット等",
                                    )
                                    pres_duration = gr.Slider(
                                        2, 20, value=5, step=1,
                                        label="🎬 Duration (秒) / 動画の長さ",
                                        info="最大20秒 (Wan2.6/Sora2=15-20s, Kling=10s, Wan2.1/LTX=5s)",
                                    )
                                    pres_mode = gr.Radio(choices=["normal", "adult"], value="normal", label="Save to")

                            pres_gen_btn = gr.Button("✨ 保持アニメ化 生成", variant="primary", size="lg")

                            with gr.Row():
                                pres_video = gr.Video(label="Generated Video", height=400)
                                pres_info = gr.Textbox(label="Info", interactive=False, lines=8)

                            # Wire events
                            pres_analyze_btn.click(
                                fn=img2vid_analyze_motion,
                                inputs=[pres_image],
                                outputs=[pres_prompt],
                            )
                            pres_gen_btn.click(
                                fn=generate_img2vid_preserve,
                                inputs=[pres_image, pres_prompt, pres_preset, pres_model,
                                        pres_auto_analyze, pres_duration, pres_mode],
                                outputs=[pres_video, pres_info],
                            )

                        # ── Inspired Mode ──
                        with gr.Tab("② Inspired (連想して新規生成)"):
                            gr.Markdown(
                                "画像からインスピレーション → AIが内容を分析 → **新しい動画を生成**。\n"
                                "スタイル変更、別解釈、雰囲気だけ継承、等に使えます。"
                            )

                            with gr.Row():
                                with gr.Column(scale=2):
                                    insp_image = gr.Image(label="参照画像 (分析用)", type="numpy", height=380)
                                    insp_desc = gr.Textbox(
                                        label="Scene Description (空欄OK — AI自動生成)",
                                        lines=6,
                                        placeholder="例: A woman standing by the window at sunset, warm golden light, cinematic, slow camera push-in...",
                                    )
                                    with gr.Row():
                                        insp_analyze_btn = gr.Button("🔍 AI画像分析", variant="secondary", size="sm")
                                        insp_style = gr.Dropdown(
                                            choices=[
                                                "(なし - そのまま)",
                                                "cinematic, 8k, film grain, dramatic lighting",
                                                "anime style, vibrant colors, detailed illustration",
                                                "dreamy, ethereal, soft focus, pastel colors",
                                                "photorealistic, 8k, sharp focus, professional photo",
                                                "oil painting, artistic, brush strokes",
                                                "cyberpunk, neon, futuristic, sci-fi",
                                                "watercolor, soft pastels, artistic",
                                                "noir, black and white, high contrast, dramatic",
                                            ],
                                            value="(なし - そのまま)",
                                            label="Style Hint (追加スタイル)",
                                        )
                                with gr.Column(scale=1):
                                    insp_model = gr.Dropdown(
                                        choices=list(FAL_VIDEO_MODELS.keys()),
                                        value=list(FAL_VIDEO_MODELS.keys())[0],
                                        label="txt2vid モデル",
                                        info="画像は参照のみ。ここのモデルで新規生成します",
                                    )
                                    insp_duration = gr.Slider(
                                        2, 20, value=5, step=1,
                                        label="🎬 Duration (秒) / 動画の長さ",
                                        info="最大20秒 (モデル依存)",
                                    )
                                    insp_mode = gr.Radio(choices=["normal", "adult"], value="normal", label="Save to")

                            insp_gen_btn = gr.Button("🎬 連想 新規生成", variant="primary", size="lg")

                            with gr.Row():
                                insp_video = gr.Video(label="Generated Video", height=400)
                                insp_info = gr.Textbox(label="Info", interactive=False, lines=8)

                            # Process style hint: filter "(なし)"
                            def _inspired_wrapper(image, desc, style, model, duration, mode):
                                style_clean = "" if style.startswith("(なし") else style
                                return generate_img2vid_inspired(image, desc, style_clean, model, duration, mode)

                            insp_analyze_btn.click(
                                fn=img2vid_analyze_inspiration,
                                inputs=[insp_image],
                                outputs=[insp_desc],
                            )
                            insp_gen_btn.click(
                                fn=_inspired_wrapper,
                                inputs=[insp_image, insp_desc, insp_style, insp_model, insp_duration, insp_mode],
                                outputs=[insp_video, insp_info],
                            )

                # ─ vid2vid ─
                with gr.Tab("Video to Video (vid2vid)"):
                    gr.Markdown("**既存動画をAIで変換** — スタイル変換、アニメ化、品質向上。元動画の動きを保ちつつAIが再描画します。")

                    V2V_STYLE_PRESETS = {
                        "(自由入力)": {"prompt": "", "neg": "", "denoise": 0.55},
                        "アニメ化": {"prompt": "anime style, colorful, detailed, cel shading, vibrant colors, illustration", "neg": "photorealistic, photo, 3d render, blurry, low quality", "denoise": 0.65},
                        "シネマティック": {"prompt": "cinematic, film grain, dramatic lighting, professional color grading, movie scene, 8k", "neg": "cartoon, anime, low quality, blurry, overexposed", "denoise": 0.45},
                        "油絵風": {"prompt": "oil painting style, brush strokes, artistic, rich colors, impressionist, canvas texture", "neg": "photo, realistic, digital, low quality", "denoise": 0.70},
                        "水彩画風": {"prompt": "watercolor painting, soft colors, flowing paint, artistic, delicate, beautiful watercolor", "neg": "photo, realistic, sharp, digital render", "denoise": 0.65},
                        "サイバーパンク": {"prompt": "cyberpunk style, neon lights, futuristic, glowing, high tech, dark atmosphere, sci-fi", "neg": "natural, daylight, simple, low quality", "denoise": 0.60},
                        "品質向上 (軽め)": {"prompt": "high quality, detailed, sharp focus, professional, best quality, masterpiece", "neg": "worst quality, low quality, blurry, noise, artifacts", "denoise": 0.35},
                    }

                    v2v_style = gr.Dropdown(
                        choices=list(V2V_STYLE_PRESETS.keys()),
                        value="(自由入力)",
                        label="Style Preset",
                        info="よく使うスタイル変換をワンクリックで設定",
                    )

                    with gr.Row():
                        with gr.Column(scale=2):
                            v2v_video_in = gr.Video(label="入力動画", height=300)
                            v2v_prompt = gr.Textbox(label="Style Prompt", lines=2, placeholder="anime style, colorful, detailed / photorealistic, cinematic, 8k")
                            v2v_neg = gr.Textbox(label="Negative Prompt", lines=1, value="worst quality, blurry, distorted, deformed")
                        with gr.Column(scale=1):
                            v2v_model = gr.Dropdown(choices=models, label="Checkpoint (SD1.5推奨)", value=models[0] if models else None)
                            v2v_motion = gr.Dropdown(choices=motion_models, label="Motion Model", value=motion_models[0] if motion_models else None)
                            v2v_vae = gr.Dropdown(choices=vaes, label="VAE", value="None")
                            v2v_denoise = gr.Slider(0.2, 0.9, value=0.55, step=0.05, label="Denoise (低い=元動画に忠実)")

                    with gr.Row():
                        v2v_w = gr.Slider(256, 1024, value=512, step=64, label="Width")
                        v2v_h = gr.Slider(256, 1024, value=512, step=64, label="Height")
                        v2v_fps = gr.Slider(4, 30, value=8, step=1, label="FPS")
                        v2v_frame_limit = gr.Slider(8, 64, value=32, step=1, label="Max Frames")

                    with gr.Row():
                        v2v_steps = gr.Slider(1, 50, value=20, step=1, label="Steps")
                        v2v_cfg = gr.Slider(1, 20, value=7.5, step=0.5, label="CFG")
                        v2v_seed = gr.Number(value=-1, label="Seed", precision=0)

                    with gr.Row():
                        v2v_sampler = gr.Dropdown(choices=SAMPLERS, value="euler_ancestral", label="Sampler")
                        v2v_sched = gr.Dropdown(choices=SCHEDULERS, value="normal", label="Scheduler")
                        v2v_format = gr.Dropdown(choices=["gif", "mp4", "webp"], value="mp4", label="Output Format")
                        v2v_mode = gr.Radio(choices=["normal", "adult"], value="normal", label="Save to")

                    def apply_v2v_style(style_name):
                        if style_name == "(自由入力)" or style_name not in V2V_STYLE_PRESETS:
                            return gr.update(), gr.update(), gr.update()
                        s = V2V_STYLE_PRESETS[style_name]
                        return gr.update(value=s["prompt"]), gr.update(value=s["neg"]), gr.update(value=s["denoise"])

                    v2v_style.change(
                        fn=apply_v2v_style,
                        inputs=[v2v_style],
                        outputs=[v2v_prompt, v2v_neg, v2v_denoise],
                    )

                    v2v_gen_btn = gr.Button("Transform Video", variant="primary", size="lg")

                    with gr.Row():
                        v2v_video_out = gr.Video(label="Transformed Video", height=400)
                        v2v_preview = gr.Gallery(label="Preview Frames", columns=4, height=256)
                    v2v_info = gr.Textbox(label="Info", interactive=False)

                    v2v_gen_btn.click(
                        fn=generate_video_vid2vid,
                        inputs=[v2v_video_in, v2v_prompt, v2v_neg, v2v_model, v2v_motion, v2v_vae,
                                v2v_w, v2v_h, v2v_steps, v2v_cfg, v2v_sampler, v2v_sched, v2v_seed,
                                v2v_fps, v2v_denoise, v2v_frame_limit, v2v_format, v2v_mode],
                        outputs=[v2v_video_out, v2v_preview, v2v_info],
                    )

                    gr.Markdown("""
**vid2vid のコツ:**
- **Denoise 0.3-0.5**: 元動画に忠実（色調補正、軽いスタイル変換）
- **Denoise 0.5-0.7**: バランス良い変換（アニメ化など）
- **Denoise 0.7-0.9**: 大幅な変換（元動画の動きだけ保持）
- **短い動画推奨**: 最初は8-16フレームでテスト
""")

                # ─ Video tips ─
                with gr.Tab("Tips"):
                    gr.Markdown("""
## AnimateDiff 動画生成のコツ

### 基本設定
- **Motion Model**: `mm_sd_v15_v2.ckpt` が最も安定
- **Checkpoint**: SD1.5ベースのモデルのみ対応（SDXLは非対応）
- **サイズ**: 512x512 が最も安定。512x768もOK
- **Frames**: 16フレームが標準。増やすとVRAM消費が増大

### 推奨設定
| パラメータ | テスト | 標準 | 高品質 |
|-----------|-------|------|--------|
| Steps | 10-15 | 20 | 25-30 |
| CFG | 7 | 7.5 | 7-8 |
| Frames | 8 | 16 | 24 |
| FPS | 8 | 8 | 12 |
| Size | 384x384 | 512x512 | 512x768 |

### img2vid のコツ
- **Denoise 0.5-0.7**: 元画像に近い動き（推奨）
- **Denoise 0.8-1.0**: 大きな変化、元画像から離れる
- **Motion Prompt**: `wind, hair moving, blinking, breathing` などの動きを指示

### VRAM使用量の目安
| 設定 | Mac MPS (18GB) | RunPod (24GB) |
|------|---------------|---------------|
| 512x512, 16f | 可能 (遅い) | 快適 |
| 512x768, 16f | ギリギリ | 快適 |
| 512x512, 24f | 不可 | 快適 |
| 512x768, 24f | 不可 | 可能 |

### 注意事項
- Mac MPSでの動画生成は**非常に遅い**（1本5-15分）
- 本格的な動画生成には**RunPod推奨**
- 初回は16フレーム/512x512でテストしてから拡大
""")

                # ─ Clip Chain (クリップ連結) ─
                with gr.Tab("Clip Chain (連結)"):
                    gr.Markdown("""
## クリップチェーン — 短いクリップを繋いで長い動画を作成

**ワークフロー:**
1. txt2vid / img2vid でクリップを生成（8-16フレーム）
2. 「最終フレーム抽出」で最後のフレームを取得
3. 抽出したフレームを img2vid に入力して次のクリップを生成
4. 繰り返してシーンを構築
5. 生成したクリップを外部ツール（CapCut, DaVinci Resolve等）で結合

**長い動画を作るコツ:**
- 各クリップは同じモデル・同じ解像度で統一感を保つ
- Denoise 0.5-0.6 で前のクリップとの一貫性を維持
- MP4フォーマットで出力（編集ソフトとの互換性が高い）
""")
                    with gr.Row():
                        with gr.Column(scale=2):
                            cc_video_in = gr.Video(label="生成済みクリップ（最終フレームを抽出）", height=300)
                        with gr.Column(scale=1):
                            cc_extract_btn = gr.Button("最終フレーム抽出", variant="primary", size="lg")
                            cc_info = gr.Textbox(label="Info", interactive=False)

                    cc_last_frame = gr.Image(label="抽出した最終フレーム → img2vid タブに入力として使用", height=300)

                    cc_extract_btn.click(
                        fn=extract_last_frame,
                        inputs=[cc_video_in],
                        outputs=[cc_last_frame, cc_info],
                    )

                    gr.Markdown("""
---
## 外部ツール連携ガイド

生成したクリップを外部の動画生成・編集ツールと組み合わせることで、より長く高品質な動画が作れます。

| ツール | 用途 | 連携方法 |
|--------|------|----------|
| **CapCut** | クリップ結合・テロップ・BGM | MP4出力をインポート、タイムラインで結合 |
| **DaVinci Resolve** | プロ編集・カラグレ | MP4をメディアプールに追加 |
| **Runway Gen-3** | AI動画拡張 | 最終フレームをimg2vidに入力、長尺化 |
| **Pika** | AI動画変換 | 画像 or 短いクリップを入力 |
| **Kling AI** | AI動画生成 | 画像から5秒動画を生成 |
| **Luma Dream Machine** | 高品質AI動画 | 画像+テキストで動画生成 |

**推奨ワークフロー:**
1. このアプリで短いAIクリップ（2-4秒）を複数生成
2. 外部ツール（CapCut等）でクリップを結合・BGM追加
3. 必要に応じて Runway / Pika で特定シーンを拡張
""")

                # ─ Long Video (自動長尺生成) ─
                with gr.Tab("Long Video (長尺)"):
                    gr.Markdown("""
## 長尺動画生成 — 自動クリップ連結で長い動画を作成

**仕組み:**
1. 最初のクリップをtxt2vidで生成
2. 最終フレームを抽出 → img2vidで次のクリップを自動生成
3. 指定回数繰り返し → ffmpegで自動結合

**例:** 5クリップ x 5秒 = 約25秒の動画
""")
                    lv_model = gr.Dropdown(
                        choices=list(FAL_VIDEO_MODELS.keys()),
                        value="Wan 2.6 txt2vid (最新・NSFW OK)",
                        label="動画モデル (最初のクリップ用)",
                    )
                    lv_prompt = gr.Textbox(
                        label="プロンプト (全クリップ共通)",
                        lines=3,
                        placeholder="例: A woman walking through a beautiful garden, cinematic, smooth motion, consistent lighting",
                    )
                    lv_neg = gr.Textbox(label="Negative Prompt (任意)", lines=1, placeholder="blurry, distorted, low quality")
                    with gr.Row():
                        lv_clips = gr.Slider(2, 10, value=3, step=1, label="クリップ数")
                        lv_duration = gr.Slider(3, 10, value=5, step=1, label="1クリップの秒数")
                    lv_mode = gr.Radio(choices=["normal", "adult"], value="normal", label="保存先")
                    lv_btn = gr.Button("長尺動画を生成", variant="primary", size="lg")
                    lv_video = gr.Video(label="結合動画", height=400)
                    lv_info = gr.Textbox(label="Info (進捗・コスト)", lines=5, interactive=False)

                    # Estimated cost/time display
                    lv_estimate = gr.Markdown("")
                    def estimate_long_video(model_key, clips, dur):
                        m = FAL_VIDEO_MODELS.get(model_key, {})
                        cost_str = m.get("cost", "?")
                        total_sec = int(clips) * int(dur)
                        return (
                            f"**見積もり**: {int(clips)}クリップ x {int(dur)}秒 = 約**{total_sec}秒**の動画\n"
                            f"コスト: {cost_str} x {int(clips)} = (最初のクリップ) + img2vid x {int(clips)-1}"
                        )
                    for _inp in [lv_model, lv_clips, lv_duration]:
                        _inp.change(fn=estimate_long_video, inputs=[lv_model, lv_clips, lv_duration], outputs=[lv_estimate])

                    lv_btn.click(
                        fn=generate_long_video,
                        inputs=[lv_prompt, lv_model, lv_clips, lv_duration, lv_neg, lv_mode],
                        outputs=[lv_video, lv_info],
                    )

                # ─ Extend Video (動画延長) ─
                with gr.Tab("Extend Video (延長)"):
                    gr.Markdown("""
## 動画延長 — 既存動画の続きを自動生成

**仕組み:**
1. アップロードした動画の最終フレームを自動抽出
2. そのフレームからimg2vidで続きを生成
3. 元動画 + 生成クリップをffmpegで結合

**用途:** 短い動画を長くしたい / 外部ツールで作った動画を延長したい
""")
                    ev_video_in = gr.Video(label="元動画（延長したい動画）")
                    ev_prompt = gr.Textbox(
                        label="延長部分のプロンプト（任意）",
                        lines=2,
                        placeholder="例: continue walking, smooth motion (空欄の場合は自動で 'smooth continuation' を使用)",
                    )
                    ev_model = gr.Dropdown(
                        choices=list(FAL_VIDEO_MODELS.keys()),
                        value="Wan 2.6 txt2vid (最新・NSFW OK)",
                        label="モデル (img2vidが自動選択される)",
                    )
                    with gr.Row():
                        ev_extensions = gr.Slider(1, 10, value=2, step=1, label="追加クリップ数")
                        ev_duration = gr.Slider(3, 10, value=5, step=1, label="1クリップの秒数")
                    ev_neg = gr.Textbox(label="Negative Prompt (任意)", lines=1)
                    ev_mode = gr.Radio(choices=["normal", "adult"], value="normal", label="保存先")
                    ev_btn = gr.Button("動画を延長", variant="primary", size="lg")
                    ev_video_out = gr.Video(label="延長済み動画", height=400)
                    ev_info = gr.Textbox(label="Info", lines=4, interactive=False)

                    ev_btn.click(
                        fn=generate_extend_video,
                        inputs=[ev_video_in, ev_prompt, ev_model, ev_extensions, ev_duration, ev_neg, ev_mode],
                        outputs=[ev_video_out, ev_info],
                    )

        # ── Flux / Cloud AI Tab ──
        with gr.Tab("Flux / Cloud AI"):
            gr.Markdown(
                "**Flux & クラウドAI生成** — 最高品質の画像生成。プロンプトを入力して押すだけ。\n"
                "Backend を `replicate` に切り替えると、在庫切れなし・Pod管理不要で即座に使えます。"
            )

            FLUX_SCENE_PRESETS = {
                "(自由入力)": "",
                "ポートレート (リアル)": "A beautiful young woman, natural skin texture, soft smile, looking at camera, professional portrait photography, studio lighting, shallow depth of field, 85mm lens, sharp focus, 8k uhd",
                "和風ポートレート": "A beautiful Japanese woman wearing traditional kimono, cherry blossoms in background, spring sunlight, elegant pose, professional photography, soft natural lighting",
                "アニメキャラ": "anime style illustration, beautiful girl, detailed eyes, colorful hair, dynamic pose, vibrant colors, detailed background, masterpiece quality",
                "風景 (ドラマチック)": "stunning landscape photography, dramatic mountains at golden hour, lake reflection, epic sky with clouds, nature, wide angle lens, 8k uhd, National Geographic style",
                "サイバーパンク": "cyberpunk city at night, neon lights reflecting on wet streets, futuristic architecture, holographic advertisements, rain, cinematic atmosphere, blade runner style",
                "ファンタジー": "epic fantasy scene, magical forest with glowing crystals, ethereal lighting, mystical atmosphere, detailed environment, concept art, digital painting, 8k",
                "NSFW ポートレート": "beautiful woman, detailed skin texture, intimate setting, soft bedroom lighting, photorealistic, professional boudoir photography, 8k uhd",
            }

            with gr.Tabs():
                # ── Replicate (メイン) ──
                with gr.Tab("Replicate (推奨)"):
                    gr.Markdown("**Replicate API** — 在庫切れなし・Pod管理不要。Backend を `replicate` に切り替えてください。")

                    with gr.Row():
                        r_model = gr.Dropdown(
                            choices=list(REPLICATE_MODELS.keys()),
                            value="Flux Schnell (高速・安い)",
                            label="モデル",
                            info="Schnell=安い($0.003) / Dev=バランス($0.03) / Pro=最高品質($0.04)",
                            scale=2,
                        )
                        r_scene = gr.Dropdown(
                            choices=list(FLUX_SCENE_PRESETS.keys()),
                            value="(自由入力)",
                            label="シーンプリセット",
                            scale=2,
                        )

                    r_prompt = gr.Textbox(
                        label="Prompt（英語推奨・日本語も可）",
                        lines=4,
                        placeholder="例: A beautiful Japanese woman in a cherry blossom garden, spring sunlight, professional photography\n例: cyberpunk city at night, neon lights, rain",
                    )

                    with gr.Row():
                        r_size = gr.Radio(
                            choices=["1024x1024 (正方形)", "768x1360 (縦長)", "1360x768 (横長)"],
                            value="1024x1024 (正方形)",
                            label="サイズ",
                            scale=2,
                        )
                        r_seed = gr.Number(value=-1, label="Seed (-1=random)", precision=0, scale=1)
                        r_mode = gr.Radio(choices=["normal", "adult"], value="normal", label="保存先", scale=1)

                    r_gen_btn = gr.Button("生成 (Replicate)", variant="primary", size="lg")

                    with gr.Row():
                        r_gallery = gr.Gallery(label="Generated Images", columns=2, height=512)
                    r_info = gr.Textbox(label="Info (コスト表示あり)", interactive=False)

                    # Hidden width/height
                    r_width = gr.Number(value=1024, visible=False, precision=0)
                    r_height = gr.Number(value=1024, visible=False, precision=0)

                    r_scene.change(
                        fn=lambda s: gr.update(value=FLUX_SCENE_PRESETS.get(s, "")) if s != "(自由入力)" else gr.update(),
                        inputs=[r_scene], outputs=[r_prompt],
                    )

                    def apply_r_size(size_str):
                        if "768x1360" in size_str:
                            return gr.update(value=768), gr.update(value=1360)
                        elif "1360x768" in size_str:
                            return gr.update(value=1360), gr.update(value=768)
                        return gr.update(value=1024), gr.update(value=1024)

                    r_size.change(fn=apply_r_size, inputs=[r_size], outputs=[r_width, r_height])

                    r_gen_btn.click(
                        fn=generate_replicate_image,
                        inputs=[r_prompt, r_model, r_width, r_height, r_seed, r_mode],
                        outputs=[r_gallery, r_info],
                    )

                    gr.Markdown("""
**料金の目安 (1枚あたり):**
| モデル | 品質 | 速度 | 料金 |
|--------|------|------|------|
| Flux Schnell | 良い | 最速(2-3秒) | ~$0.003 (~¥0.5) |
| Flux Dev | 高い | 普通(10-15秒) | ~$0.03 (~¥5) |
| Flux 1.1 Pro | 最高 | 普通(10-15秒) | ~$0.04 (~¥6) |
| SDXL | 良い | 普通 | ~$0.01 (~¥1.5) |
""")

                # ── fal.ai 画像生成 ──
                with gr.Tab("fal.ai 画像 (推奨)"):
                    gr.Markdown("**fal.ai画像生成** — Flux品質・NSFW完全対応・在庫切れなし。")
                    fi_model = gr.Dropdown(
                        choices=list(FAL_MODELS.keys()),
                        value=list(FAL_MODELS.keys())[0],
                        label="モデル",
                        info="Pro=最高品質 / Dev=バランス / Schnell=高速・安い / Realism=フォトリアル",
                    )
                    fi_scene = gr.Dropdown(
                        choices=list(FLUX_SCENE_PRESETS.keys()),
                        value="(自由入力)",
                        label="シーンプリセット",
                    )
                    fi_prompt = gr.Textbox(
                        label="Prompt（英語推奨・日本語も可）",
                        lines=4,
                        placeholder="例: A beautiful Japanese woman in cherry blossom garden, professional photography, 8k",
                    )
                    with gr.Row():
                        fi_size = gr.Radio(
                            choices=["1024x1024 (正方形)", "768x1360 (縦長)", "1360x768 (横長)"],
                            value="1024x1024 (正方形)",
                            label="サイズ",
                            scale=2,
                        )
                        fi_num = gr.Slider(1, 4, value=1, step=1, label="枚数", scale=1)
                        fi_seed = gr.Number(value=-1, label="Seed (-1=random)", precision=0, scale=1)
                        fi_mode = gr.Radio(choices=["normal", "adult"], value="normal", label="保存先", scale=1)
                    fi_gen_btn = gr.Button("生成 (fal.ai)", variant="primary", size="lg")
                    with gr.Row():
                        fi_gallery = gr.Gallery(label="Generated Images", columns=2, height=512)
                    fi_info = gr.Textbox(label="Info (コスト表示あり)", interactive=False)

                    fi_width = gr.Number(value=1024, visible=False, precision=0)
                    fi_height = gr.Number(value=1024, visible=False, precision=0)

                    fi_scene.change(
                        fn=lambda s: gr.update(value=FLUX_SCENE_PRESETS.get(s, "")) if s != "(自由入力)" else gr.update(),
                        inputs=[fi_scene], outputs=[fi_prompt],
                    )

                    def apply_fi_size(size_str):
                        if "768x1360" in size_str:
                            return gr.update(value=768), gr.update(value=1360)
                        elif "1360x768" in size_str:
                            return gr.update(value=1360), gr.update(value=768)
                        return gr.update(value=1024), gr.update(value=1024)

                    fi_size.change(fn=apply_fi_size, inputs=[fi_size], outputs=[fi_width, fi_height])

                    fi_gen_btn.click(
                        fn=generate_fal_image_direct,
                        inputs=[fi_prompt, fi_model, fi_width, fi_height, fi_seed, fi_num, fi_mode],
                        outputs=[fi_gallery, fi_info],
                    )

                    gr.Markdown("""
**fal.ai料金の目安 (1枚あたり):**
| モデル | 品質 | 速度 | 料金 | NSFW |
|--------|------|------|------|------|
| Flux Dev | 高い | 普通 | ~$0.025 (~¥4) | **完全OK** |
| Flux Schnell | 良い | 最速 | ~$0.003 (~¥0.5) | **完全OK** |
| Flux Realism | 最高(写真) | 普通 | ~$0.025 (~¥4) | **完全OK** |
| Flux Pro 1.1 | 最高 | 普通 | ~$0.05 (~¥8) | ❌ SFWのみ |

⚠️ **Flux Pro 1.1はNSFWで真っ黒になります。NSFW→ Dev / Realism / Schnell を使ってください。**
""")

                # ── AI動画生成 (Replicate) ──
                with gr.Tab("AI動画生成 (Replicate)"):
                    gr.Markdown("**AI動画生成** — テキストから動画を生成。Backend を `replicate` に切り替えてください。")

                    rv_model = gr.Dropdown(
                        choices=list(REPLICATE_VIDEO_MODELS.keys()),
                        value="LTX Video (高速)",
                        label="動画モデル",
                        info="LTX=安い / Minimax=高品質 / Hunyuan=テンセント製",
                    )
                    rv_prompt = gr.Textbox(
                        label="Prompt",
                        lines=3,
                        placeholder="例: A woman walking through cherry blossoms, gentle wind, cinematic",
                    )
                    rv_mode = gr.Radio(choices=["normal", "adult"], value="normal", label="保存先")
                    rv_gen_btn = gr.Button("動画生成 (Replicate)", variant="primary", size="lg")
                    rv_video = gr.Video(label="Generated Video", height=400)
                    rv_info = gr.Textbox(label="Info", interactive=False)

                    rv_gen_btn.click(
                        fn=generate_replicate_video,
                        inputs=[rv_prompt, rv_model, rv_mode],
                        outputs=[rv_video, rv_info],
                    )

                    gr.Markdown("""
**動画料金の目安:**
| モデル | 品質 | 料金 |
|--------|------|------|
| LTX Video | 良い | ~$0.02 (~¥3) |
| Kling v1.6 | 高い | ~$0.10 (~¥15) |
| Minimax Video-01 | 高い | ~$0.13 (~¥20) |
| Hunyuan Video | 高い | ~$0.32 (~¥48) |
""")

                # ── fal.ai 動画生成 ──
                with gr.Tab("AI動画 (fal.ai)"):
                    gr.Markdown(
                        "**fal.ai動画生成** — Veo 3, Sora 2, Kling, Wan等の動画モデル。\n\n"
                        "**長さ**: モデルにより最大5-20秒。長い動画は「長尺動画」タブで自動連結。"
                    )
                    fv_model = gr.Dropdown(
                        choices=list(FAL_VIDEO_MODELS.keys()),
                        value=list(FAL_VIDEO_MODELS.keys())[0],
                        label="動画モデル",
                    )
                    fv_prompt = gr.Textbox(label="Prompt", lines=3, placeholder="例: A woman walking on the beach at sunset, cinematic, slow motion")
                    fv_neg = gr.Textbox(label="Negative Prompt (任意)", lines=1, placeholder="blurry, low quality, distorted")
                    with gr.Row():
                        fv_duration = gr.Slider(1, 20, value=5, step=1, label="秒数 (モデルの最大値で自動クランプ)")
                        fv_seed = gr.Number(value=-1, label="Seed (-1=random)")
                    with gr.Row():
                        fv_width = gr.Slider(256, 1920, value=1280, step=64, label="Width (対応モデルのみ)")
                        fv_height = gr.Slider(256, 1080, value=720, step=64, label="Height (対応モデルのみ)")
                    fv_mode = gr.Radio(choices=["normal", "adult"], value="normal", label="保存先")
                    fv_gen_btn = gr.Button("動画生成 (fal.ai)", variant="primary", size="lg")
                    fv_video = gr.Video(label="Generated Video", height=400)
                    fv_info = gr.Textbox(label="Info", interactive=False)

                    # Show model info on selection
                    fv_model_info = gr.Markdown("")
                    def show_video_model_info(key):
                        m = FAL_VIDEO_MODELS.get(key, {})
                        nsfw = "NSFW OK" if m.get("nsfw") else "SFWのみ"
                        return (
                            f"**最大秒数**: {m.get('max_duration', '?')}秒 | "
                            f"**解像度変更**: {'対応' if m.get('supports_resolution') else '固定'} | "
                            f"**{nsfw}** | コスト: {m.get('cost', '?')}"
                        )
                    fv_model.change(fn=show_video_model_info, inputs=[fv_model], outputs=[fv_model_info])

                    fv_gen_btn.click(
                        fn=generate_fal_video,
                        inputs=[fv_prompt, fv_model, fv_mode, fv_duration, fv_width, fv_height, fv_neg, fv_seed],
                        outputs=[fv_video, fv_info],
                    )

                # ── img2vid (画像→動画) ──
                with gr.Tab("画像→動画 (img2vid)"):
                    gr.Markdown(
                        "**画像→動画変換** — Face Swap画像や生成画像を動画化。fal.aiクラウド処理。\n\n"
                        "**使い方:** Face Swapタブで作った画像をここにドラッグ → 動画化"
                    )
                    iv_model = gr.Dropdown(
                        choices=list(FAL_IMG2VID_MODELS.keys()),
                        value=list(FAL_IMG2VID_MODELS.keys())[0],
                        label="動画モデル",
                        info="Veo3/Sora2=最高品質 / Kling=高品質 / Wan=NSFW OK / LTX=高速安い",
                    )
                    iv_image = gr.Image(label="入力画像（Face Swap結果やAI生成画像）", type="filepath")
                    iv_prompt = gr.Textbox(
                        label="動きの指示（任意）",
                        lines=2,
                        placeholder="例: gentle smile, hair blowing in wind, slow camera zoom in",
                    )
                    iv_duration = gr.Slider(1, 20, value=5, step=1, label="秒数 (モデルの最大値で自動クランプ)")
                    with gr.Row():
                        iv_mode = gr.Radio(choices=["normal", "adult"], value="normal", label="保存先", scale=1)
                        iv_btn = gr.Button("画像→動画 生成", variant="primary", size="lg", scale=2)
                    iv_video = gr.Video(label="Generated Video", height=400)
                    iv_info = gr.Textbox(label="Info", interactive=False)

                    iv_btn.click(
                        fn=generate_fal_img2vid,
                        inputs=[iv_image, iv_prompt, iv_model, iv_mode, iv_duration],
                        outputs=[iv_video, iv_info],
                    )

                    gr.Markdown("""
**img2vid料金の目安:**
| モデル | 品質 | 料金 |
|--------|------|------|
| Kling 2.5 Turbo Pro | 最高 | ~$0.10 (~¥15) |
| Wan 2.1 | 高い | ~$0.05 (~¥8) |
| LTX 2.3 | 良い(高速) | ~$0.02 (~¥3) |
""")

                # ── Dezgo 動画生成 ──
                with gr.Tab("AI動画 (Dezgo)"):
                    gr.Markdown("**Dezgo動画生成** — 完全無検閲。Wan 2.6モデル。")
                    dv_model = gr.Dropdown(
                        choices=list(DEZGO_VIDEO_MODELS.keys()),
                        value=list(DEZGO_VIDEO_MODELS.keys())[0],
                        label="動画モデル",
                    )
                    dv_prompt = gr.Textbox(label="Prompt", lines=3, placeholder="例: Cinematic scene of a samurai in rain, dramatic lighting")
                    dv_mode = gr.Radio(choices=["normal", "adult"], value="normal", label="保存先")
                    dv_gen_btn = gr.Button("動画生成 (Dezgo)", variant="primary", size="lg")
                    dv_video = gr.Video(label="Generated Video", height=400)
                    dv_info = gr.Textbox(label="Info", interactive=False)

                    dv_gen_btn.click(
                        fn=generate_dezgo_video,
                        inputs=[dv_prompt, dv_model, dv_mode],
                        outputs=[dv_video, dv_info],
                    )

                # ── RunPod Flux (予備) ──
                with gr.Tab("RunPod Flux (予備)"):
                    gr.Markdown("**RunPod経由のFlux生成** — Replicateが使えない場合の予備。Backend を `runpod` に切り替えてください。")

                    f_prompt = gr.Textbox(label="Prompt", lines=4, placeholder="Flux prompt...")
                    with gr.Row():
                        f_width = gr.Number(value=1024, label="Width", precision=0)
                        f_height = gr.Number(value=1024, label="Height", precision=0)
                        f_seed = gr.Number(value=-1, label="Seed", precision=0)
                        f_mode = gr.Radio(choices=["normal", "adult"], value="normal", label="保存先")

                    # Hidden RunPod defaults
                    f_unet = gr.Textbox(value=FLUX_DEFAULTS["unet"], visible=False)
                    f_clip_l = gr.Textbox(value=FLUX_DEFAULTS["clip_l"], visible=False)
                    f_t5xxl = gr.Textbox(value=FLUX_DEFAULTS["t5xxl"], visible=False)
                    f_vae = gr.Textbox(value=FLUX_DEFAULTS["vae"], visible=False)
                    f_steps = gr.Number(value=20, visible=False, precision=0)
                    f_guidance = gr.Number(value=3.5, visible=False)
                    f_sampler = gr.Textbox(value="euler", visible=False)
                    f_scheduler = gr.Textbox(value="simple", visible=False)
                    f_batch = gr.Number(value=1, visible=False, precision=0)
                    f_dtype = gr.Textbox(value="fp8_e4m3fn", visible=False)
                    f_lora = gr.Textbox(value="None", visible=False)
                    f_lora_str = gr.Number(value=0.8, visible=False)

                    f_gen_btn = gr.Button("生成 (RunPod Flux)", variant="primary", size="lg")
                    with gr.Row():
                        f_gallery = gr.Gallery(label="Generated Images", columns=2, height=512)
                    f_info = gr.Textbox(label="Info", interactive=False)

                    f_gen_btn.click(
                        fn=generate_flux,
                        inputs=[f_prompt, f_unet, f_clip_l, f_t5xxl, f_vae, f_lora, f_lora_str,
                                f_width, f_height, f_steps, f_guidance, f_sampler, f_scheduler,
                                f_seed, f_batch, f_dtype, f_mode],
                        outputs=[f_gallery, f_info],
                    )

        # ── CivitAI Tab ──
        with gr.Tab("CivitAI"):
            gr.Markdown("**CivitAI連携** — クラウド生成 (NSFW OK)・モデル検索・ダウンロード")

            with gr.Group():
                gr.Markdown("### Cloud Generation (クラウド生成) — NSFW OK・ダウンロード不要")
                gr.Markdown("CivitAI のGPUで生成。好きなモデルをそのまま使えます。NSFW制限なし。")
                with gr.Row():
                    with gr.Column(scale=2):
                        cg_prompt = gr.Textbox(label="Prompt", lines=3, placeholder="masterpiece, best quality, RAW photo, beautiful woman...")
                        cg_negative = gr.Textbox(label="Negative Prompt", lines=2, value="worst quality, low quality, blurry, deformed, ugly, bad anatomy")
                    with gr.Column(scale=1):
                        _cg_icloud = get_icloud_only_models()
                        _cg_icloud_prefixed = [f"[iCloud] {m}" for m in _cg_icloud]
                        cg_model = gr.Dropdown(
                            choices=list(CIVITAI_GENERATION_MODELS.keys()) + _cg_icloud_prefixed,
                            value="Juggernaut XL Ragnarok (SDXL 最高品質)",
                            label="Model (プリセット / iCloud)",
                            info="SDXL系が高品質。NSFW→Pony系推奨。[iCloud]はCivitAIでURN自動解決",
                        )
                        cg_version_id = gr.Textbox(
                            label="Version ID (検索結果から・空欄ならプリセット使用)",
                            placeholder="検索結果のVersion IDをコピペ",
                        )
                with gr.Row():
                    cg_w = gr.Slider(256, 1024, value=1024, step=64, label="Width")
                    cg_h = gr.Slider(256, 1024, value=1024, step=64, label="Height")
                    cg_steps = gr.Slider(10, 50, value=30, step=1, label="Steps")
                    cg_cfg = gr.Slider(1, 15, value=5, step=0.5, label="CFG")
                with gr.Row():
                    cg_seed = gr.Number(value=-1, label="Seed (-1=random)", precision=0)
                    cg_qty = gr.Slider(1, 4, value=1, step=1, label="枚数")
                    cg_gen_btn = gr.Button("Generate (CivitAI Cloud)", variant="primary", size="lg")

                with gr.Row():
                    cg_gallery = gr.Gallery(label="Generated Images", columns=2, height=512)
                cg_info = gr.Textbox(label="Info", interactive=False)

                cg_gen_btn.click(
                    fn=civitai_generate_cloud,
                    inputs=[cg_prompt, cg_negative, cg_model, cg_version_id,
                            cg_w, cg_h, cg_steps, cg_cfg, cg_seed, cg_qty],
                    outputs=[cg_gallery, cg_info],
                )

            with gr.Group():
                gr.Markdown("### Model Search (モデル検索)")
                gr.Markdown("キーワード検索はCivitAI側の制限で不安定です。空欄で人気順ブラウズが確実。")
                with gr.Row():
                    c_query = gr.Textbox(label="検索キーワード (空欄=人気順ブラウズ)", placeholder="空欄推奨。キーワードは結果0件の場合あり")
                    c_type = gr.Dropdown(
                        choices=["Checkpoint", "LORA", "TextualInversion", "VAE", "ControlNet", "Upscaler"],
                        value="Checkpoint",
                        label="Type",
                    )
                    c_sort = gr.Dropdown(
                        choices=["Highest Rated", "Most Downloaded", "Newest"],
                        value="Highest Rated",
                        label="Sort",
                    )
                    c_nsfw = gr.Checkbox(label="NSFW含む", value=True)
                with gr.Row():
                    c_search_btn = gr.Button("検索", variant="primary")
                c_results = gr.Textbox(label="検索結果", lines=15, interactive=False)
                c_search_btn.click(
                    fn=civitai_search,
                    inputs=[c_query, c_type, c_sort, c_nsfw],
                    outputs=[c_results],
                )

            with gr.Group():
                gr.Markdown("### Download (モデルDL)")
                gr.Markdown("検索結果の **Version ID** を入力してダウンロード。modelsフォルダに自動保存されます。")
                with gr.Row():
                    c_version_id = gr.Textbox(label="Version ID", placeholder="検索結果のVersion IDをコピペ")
                    c_dest_type = gr.Dropdown(
                        choices=["Checkpoint", "LoRA", "VAE", "ControlNet", "Upscaler", "Embedding"],
                        value="Checkpoint",
                        label="保存先タイプ",
                    )
                    c_dl_btn = gr.Button("ダウンロード", variant="primary")
                c_dl_status = gr.Textbox(label="ダウンロード状況", lines=4, interactive=False)
                # c_dl_btn event wired below after c_model_summary is defined

            with gr.Group():
                gr.Markdown("### Direct URL Download (URL直接ダウンロード)")
                gr.Markdown("HuggingFace等から直接ダウンロード。CivitAI以外のモデルソースに対応。")
                with gr.Row():
                    c_url = gr.Textbox(
                        label="Download URL",
                        placeholder="https://huggingface.co/xxx/resolve/main/model.safetensors",
                        scale=3,
                    )
                    c_url_type = gr.Dropdown(
                        choices=["Checkpoint", "LoRA", "VAE", "ControlNet", "Upscaler", "Embedding", "UNET / Diffusion Model", "CLIP / Text Encoder"],
                        value="Checkpoint",
                        label="保存先タイプ",
                        scale=1,
                    )
                with gr.Row():
                    c_url_filename = gr.Textbox(label="ファイル名 (空欄=URLから自動)", placeholder="", scale=2)
                    c_url_dl_btn = gr.Button("URLからダウンロード", variant="primary", scale=1)
                c_url_dl_status = gr.Textbox(label="ダウンロード状況", lines=3, interactive=False)
                # c_url_dl_btn event wired below after c_model_summary is defined

            with gr.Group():
                gr.Markdown("### Model Manager (インストール済みモデル)")
                c_model_summary = gr.Textbox(label="モデル一覧", lines=15, interactive=False, value=get_model_summary())
                with gr.Row():
                    c_refresh_summary_btn = gr.Button("モデル一覧を更新")
                    c_open_models_btn = gr.Button("Models フォルダを開く")
                c_refresh_summary_btn.click(fn=get_model_summary, outputs=[c_model_summary])
                c_open_models_btn.click(fn=lambda: open_folder(config["models_dir"]), outputs=[c_url_dl_status])

            with gr.Group():
                gr.Markdown("### Vault (クラウドモデル保管庫)")
                gr.Markdown(
                    "CivitAI Vaultに保存したモデルの一覧表示・ダウンロード。\n"
                    "Supporter以上のプランで利用可能。ダウンロード先はGoogle Driveに自動保存されます。"
                )
                with gr.Row():
                    vault_query = gr.Textbox(label="検索 (空欄=全件)", placeholder="モデル名で検索", scale=2)
                    vault_list_btn = gr.Button("Vault一覧を取得", variant="primary", scale=1)
                vault_results = gr.Textbox(label="Vaultアイテム一覧", lines=12, interactive=False)
                with gr.Row():
                    vault_version_id = gr.Textbox(label="Version ID (一覧からコピペ)", placeholder="ダウンロードしたいモデルのVersion ID", scale=2)
                    vault_dest_type = gr.Dropdown(
                        choices=["Checkpoint", "LoRA", "VAE", "ControlNet", "Upscaler", "Embedding"],
                        value="Checkpoint", label="保存先タイプ", scale=1,
                    )
                    vault_dl_btn = gr.Button("Vaultからダウンロード", variant="primary", scale=1)
                vault_status = gr.Textbox(label="Vault状況", lines=3, interactive=False)

                def list_vault_items(query):
                    try:
                        info = civitai.vault_get()
                        storage_used = info.get("usedStorageKb", 0) / 1024 / 1024
                        storage_limit = info.get("storageKb", 0) / 1024 / 1024
                        header = f"Vault: {storage_used:.1f}GB / {storage_limit:.1f}GB 使用中\n{'='*50}\n"

                        items_data = civitai.vault_list(limit=60, query=query or "")
                        items = items_data.get("items", []) if isinstance(items_data, dict) else []
                        if not items:
                            return header + "Vaultにアイテムがありません。"
                        lines = [header]
                        for i, item in enumerate(items, 1):
                            name = item.get("modelName", "Unknown")
                            version = item.get("versionName", "")
                            vid = item.get("modelVersionId", "?")
                            model_type = item.get("type", "")
                            files = item.get("files", [])
                            size = sum(f.get("sizeKB", 0) for f in files) / 1024 / 1024 if files else 0
                            lines.append(f"[{i}] {name} ({version})\n    Type: {model_type} | Size: {size:.1f}GB | Version ID: {vid}")
                        return "\n".join(lines)
                    except Exception as e:
                        return f"Vaultエラー: {e}\n\nAPI Keyが正しく設定されているか確認してください。\nVaultはSupporter以上のプランで利用可能です。"

                vault_list_btn.click(fn=list_vault_items, inputs=[vault_query], outputs=[vault_results])
                # vault_dl_btn event wired below after _all_model_outputs is defined

            with gr.Group():
                gr.Markdown("### Upload (画像アップロード)")
                gr.Markdown("生成した画像をCivitAIに投稿します。(API Key必須)")
                with gr.Row():
                    c_upload_file = gr.Textbox(label="画像パス", placeholder="outputs/normal/img_20260306_xxxx.png")
                    c_upload_nsfw = gr.Checkbox(label="NSFW", value=False)
                c_upload_btn = gr.Button("CivitAIにアップロード")
                c_upload_status = gr.Textbox(label="アップロード状況", interactive=False)
                c_upload_btn.click(
                    fn=civitai_upload_image,
                    inputs=[c_upload_file, gr.Textbox(visible=False, value=""), c_upload_nsfw],
                    outputs=[c_upload_status],
                )

            with gr.Group():
                gr.Markdown("### Training (モデルトレーニング)")
                gr.Markdown(
                    "CivitAIのトレーニング機能を使ってカスタムLoRAを作成できます。\n\n"
                    "- **SD1.5 LoRA**: ~500-1000 Buzz\n"
                    "- **SDXL LoRA**: ~1000-2000 Buzz\n"
                    "- トレーニングはCivitAIのGPUで実行されます（ローカルGPU不要）\n\n"
                    "**[CivitAI Training Page](https://civitai.com/models/train)** で直接トレーニングを開始できます。"
                )

            # Wire download buttons with auto-refresh of all model dropdowns
            _all_model_outputs = [
                c_dl_status, q_model, q_lora, q_vae, a_model, a_lora, a_vae,
                v_model, v_lora, v_vae, v_motion,
                i2v_model, i2v_vae, i2v_motion,
                v2v_model, v2v_vae, v2v_motion,
                c_model_summary,
            ]
            c_dl_btn.click(
                fn=civitai_download,
                inputs=[c_version_id, c_dest_type],
                outputs=_all_model_outputs,
            )
            _url_model_outputs = [
                c_url_dl_status, q_model, q_lora, q_vae, a_model, a_lora, a_vae,
                v_model, v_lora, v_vae, v_motion,
                i2v_model, i2v_vae, i2v_motion,
                v2v_model, v2v_vae, v2v_motion,
                c_model_summary,
            ]
            c_url_dl_btn.click(
                fn=download_from_url,
                inputs=[c_url, c_url_type, c_url_filename],
                outputs=_url_model_outputs,
            )
            # Vault download button (defined above, wired here after _all_model_outputs)
            _vault_model_outputs = [
                vault_status, q_model, q_lora, q_vae, a_model, a_lora, a_vae,
                v_model, v_lora, v_vae, v_motion,
                i2v_model, i2v_vae, i2v_motion,
                v2v_model, v2v_vae, v2v_motion,
                c_model_summary,
            ]
            vault_dl_btn.click(
                fn=civitai_download,
                inputs=[vault_version_id, vault_dest_type],
                outputs=_vault_model_outputs,
            )

            # ── vast.ai ComfyUI Direct Install ──
            gr.Markdown("---")
            gr.Markdown("### vast.ai ComfyUI にモデルをインストール")
            gr.Markdown(
                "CivitAI Early Access等、Novita.aiに未掲載のモデルを\n"
                "vast.ai ComfyUIインスタンスに直接ダウンロード。\n"
                "NSFW生成にCivitAIモデルを使いたい場合に。"
            )
            with gr.Row():
                vast_dl_input = gr.Textbox(
                    label="CivitAI Version ID または ダウンロードURL",
                    placeholder="例: 1234567 or https://civitai.com/api/download/models/1234567",
                    scale=2,
                )
                vast_dl_type = gr.Dropdown(
                    choices=["Checkpoint", "LoRA", "VAE", "Embedding", "ControlNet"],
                    value="LoRA", label="モデルタイプ", scale=1,
                )
                vast_dl_btn = gr.Button("vast.aiにインストール", variant="primary", scale=1)
            vast_dl_status = gr.Textbox(label="インストール状況", lines=3, interactive=False)

            vast_dl_btn.click(
                fn=install_to_vastai,
                inputs=[vast_dl_input, vast_dl_type],
                outputs=[vast_dl_status],
            )

        # ══════════════════════════════════════════════
        # NEW ADVANCED TABS (ai-studio parity)
        # ══════════════════════════════════════════════

        # ── Style Transfer Tab ──
        with gr.Tab("Style Transfer"):
            gr.Markdown(
                "**Style Transfer** — 画像にアートスタイルを適用。fal.aiクラウド処理\n\n"
                "1. 元画像をアップロード → 2. スタイル選択 → 3. 「スタイル適用」"
            )
            with gr.Row():
                with gr.Column(scale=2):
                    st_image = gr.Image(label="元画像", type="filepath")
                with gr.Column(scale=1):
                    st_style = gr.Dropdown(
                        choices=list(STYLE_PRESETS.keys()),
                        value=list(STYLE_PRESETS.keys())[0],
                        label="スタイル",
                    )
                    st_strength = gr.Slider(0.3, 1.0, value=0.75, step=0.05, label="強度 (高い=スタイル強め)")
                    st_custom = gr.Textbox(label="追加プロンプト (任意)", placeholder="描写の補足...", lines=2)
                    st_mode = gr.Radio(choices=["normal", "adult"], value="normal", label="保存先")
                    st_btn = gr.Button("スタイル適用", variant="primary", size="lg")
            st_result = gr.Image(label="結果", height=512)
            st_info = gr.Textbox(label="Info", interactive=False)

            st_btn.click(
                fn=generate_style_transfer,
                inputs=[st_image, st_style, st_custom, st_strength, st_mode],
                outputs=[st_result, st_info],
            )

        # ── Inpaint Tab ──
        with gr.Tab("Inpaint"):
            gr.Markdown(
                "**Inpaint** — 画像の一部をマスクで指定して再生成\n\n"
                "1. 元画像 + マスク画像（白=編集エリア）をアップロード\n"
                "2. 編集内容をプロンプトで記述 → 「インペイント実行」"
            )
            with gr.Row():
                ip_image = gr.Image(label="元画像", type="filepath")
                ip_mask = gr.Image(label="マスク（白=編集エリア）", type="filepath")
            ip_prompt = gr.Textbox(label="プロンプト (編集内容)", placeholder="例: blue eyes, smiling face", lines=2)
            ip_neg = gr.Textbox(label="Negative Prompt", value="worst quality, blurry, deformed", lines=1)
            with gr.Row():
                ip_w = gr.Slider(256, 2048, value=1024, step=64, label="Width")
                ip_h = gr.Slider(256, 2048, value=1024, step=64, label="Height")
            with gr.Row():
                ip_steps = gr.Slider(1, 50, value=28, step=1, label="Steps")
                ip_guidance = gr.Slider(1.0, 10.0, value=3.5, step=0.5, label="Guidance")
                ip_seed = gr.Number(value=-1, label="Seed (-1=random)")
            ip_mode = gr.Radio(choices=["normal", "adult"], value="normal", label="保存先")
            ip_btn = gr.Button("インペイント実行", variant="primary", size="lg")
            ip_result = gr.Image(label="結果", height=512)
            ip_info = gr.Textbox(label="Info", interactive=False)

            ip_btn.click(
                fn=generate_inpaint,
                inputs=[ip_image, ip_mask, ip_prompt, ip_neg, ip_w, ip_h, ip_steps, ip_guidance, ip_seed, ip_mode],
                outputs=[ip_result, ip_info],
            )

        # ── Remove BG Tab ──
        with gr.Tab("Remove BG"):
            gr.Markdown(
                "**Background Removal** — AIで背景を自動除去（透明PNG出力）\n\n"
                "画像をアップロードして「背景除去」を押すだけ。"
            )
            with gr.Row():
                rmbg_image = gr.Image(label="入力画像", type="filepath")
                rmbg_result = gr.Image(label="結果（透明背景）", height=512)
            rmbg_mode = gr.Radio(choices=["normal", "adult"], value="normal", label="保存先")
            rmbg_btn = gr.Button("背景除去", variant="primary", size="lg")
            rmbg_info = gr.Textbox(label="Info", interactive=False)

            rmbg_btn.click(
                fn=generate_remove_bg,
                inputs=[rmbg_image, rmbg_mode],
                outputs=[rmbg_result, rmbg_info],
            )

        # ── Upscale Tab ──
        with gr.Tab("Upscale"):
            gr.Markdown(
                "**AI Upscale** — AIによる高品質アップスケーリング\n\n"
                "画像をアップロード → 倍率選択 → 「アップスケール実行」\n"
                "2x=Aura SR (高速)、4x=Clarity Upscaler (高品質)"
            )
            with gr.Row():
                us_image = gr.Image(label="入力画像", type="filepath")
                us_result = gr.Image(label="結果", height=512)
            with gr.Row():
                us_scale = gr.Radio(choices=["2", "4"], value="2", label="倍率")
                us_mode = gr.Radio(choices=["normal", "adult"], value="normal", label="保存先")
            us_btn = gr.Button("アップスケール実行", variant="primary", size="lg")
            us_info = gr.Textbox(label="Info", interactive=False)

            us_btn.click(
                fn=generate_upscale,
                inputs=[us_image, us_scale, us_mode],
                outputs=[us_result, us_info],
            )

        # ── ControlNet Cloud Tab ──
        with gr.Tab("ControlNet"):
            gr.Markdown(
                "**ControlNet (Cloud)** — 参照画像の構図・ポーズ・輪郭を維持して新しい画像を生成\n\n"
                "1. コントロール画像（エッジ/深度/ポーズ/スケッチ）をアップロード\n"
                "2. コントロールタイプ選択 + プロンプト → 「生成」"
            )
            with gr.Row():
                with gr.Column(scale=2):
                    cn_image = gr.Image(label="コントロール画像", type="filepath")
                with gr.Column(scale=1):
                    cn_type = gr.Dropdown(
                        choices=list(CONTROLNET_TYPES.keys()),
                        value=list(CONTROLNET_TYPES.keys())[0],
                        label="コントロールタイプ",
                    )
                    cn_strength = gr.Slider(0.1, 2.0, value=0.7, step=0.1, label="コントロール強度")
            cn_prompt = gr.Textbox(label="プロンプト", placeholder="例: beautiful anime girl, detailed...", lines=2)
            cn_neg = gr.Textbox(label="Negative Prompt", value="worst quality, blurry, deformed", lines=1)
            with gr.Row():
                cn_w = gr.Slider(256, 2048, value=1024, step=64, label="Width")
                cn_h = gr.Slider(256, 2048, value=1024, step=64, label="Height")
            with gr.Row():
                cn_steps = gr.Slider(1, 50, value=28, step=1, label="Steps")
                cn_guidance = gr.Slider(1.0, 10.0, value=3.5, step=0.5, label="Guidance")
                cn_seed = gr.Number(value=-1, label="Seed")
            cn_mode = gr.Radio(choices=["normal", "adult"], value="normal", label="保存先")
            cn_btn = gr.Button("ControlNet生成", variant="primary", size="lg")
            cn_result = gr.Image(label="結果", height=512)
            cn_info = gr.Textbox(label="Info", interactive=False)

            cn_btn.click(
                fn=generate_controlnet,
                inputs=[cn_image, cn_prompt, cn_neg, cn_type, cn_strength, cn_w, cn_h, cn_steps, cn_guidance, cn_seed, cn_mode],
                outputs=[cn_result, cn_info],
            )

        # ── Vid2Vid Cloud Tab ──
        with gr.Tab("Vid2Vid Cloud"):
            gr.Markdown(
                "**Video-to-Video (Cloud)** — 動画のスタイル変換・リメイク\n\n"
                "元動画をアップロード → プロンプトでスタイル指定 → クラウドで変換\n"
                "モーションを維持しながらスタイルを変更できます。"
            )
            v2v_video = gr.Video(label="元動画")
            v2v_prompt = gr.Textbox(label="プロンプト (変換後のスタイル)", placeholder="例: anime style, studio ghibli, colorful...", lines=2)
            with gr.Row():
                v2v_model = gr.Dropdown(
                    choices=list(FAL_VID2VID_MODELS.keys()),
                    value=list(FAL_VID2VID_MODELS.keys())[0],
                    label="モデル",
                    scale=2,
                )
                v2v_strength = gr.Slider(0.2, 0.9, value=0.6, step=0.05, label="変換強度 (高い=大きく変化)")
            v2v_mode = gr.Radio(choices=["normal", "adult"], value="normal", label="保存先")
            v2v_btn = gr.Button("Vid2Vid 実行", variant="primary", size="lg")
            v2v_result = gr.Video(label="結果")
            v2v_info = gr.Textbox(label="Info", interactive=False)

            v2v_btn.click(
                fn=generate_vid2vid_cloud,
                inputs=[v2v_video, v2v_prompt, v2v_model, v2v_strength, v2v_mode],
                outputs=[v2v_result, v2v_info],
            )

        # ── Art Presets Tab ──
        with gr.Tab("Art Presets"):
            gr.Markdown(
                "**芸術プリセット** — ワンクリックで芸術性の高い画像を生成\n\n"
                "プロンプト + プリセット選択 → 最適なモデル・設定で自動生成\n"
                "NSFWプリセットも含みます。"
            )
            with gr.Tabs():
                with gr.Tab("芸術プリセット"):
                    ap_prompt = gr.Textbox(label="プロンプト (被写体・シーン)", placeholder="例: a woman standing in autumn forest", lines=3)
                    ap_preset = gr.Dropdown(
                        choices=list(ART_PRESETS.keys()),
                        value=list(ART_PRESETS.keys())[0],
                        label="プリセット",
                    )
                    with gr.Row():
                        ap_seed = gr.Number(value=-1, label="Seed (-1=random)")
                        ap_mode = gr.Radio(choices=["normal", "adult"], value="normal", label="保存先")
                    ap_btn = gr.Button("生成", variant="primary", size="lg")
                    ap_gallery = gr.Gallery(label="結果", columns=2, height=512)
                    ap_info = gr.Textbox(label="Info", interactive=False)

                    # Show preset details
                    ap_details = gr.Markdown("")
                    def show_preset_details(key):
                        p = ART_PRESETS.get(key, {})
                        return (
                            f"**Model**: {p.get('model', '?')}\n"
                            f"**Size**: {p.get('width', '?')}x{p.get('height', '?')}\n"
                            f"**Steps**: {p.get('steps', '?')} | **CFG**: {p.get('cfg', '?')}\n"
                            f"**Suffix**: {p.get('prompt_suffix', '')[:100]}..."
                        )
                    ap_preset.change(fn=show_preset_details, inputs=[ap_preset], outputs=[ap_details])

                    ap_btn.click(
                        fn=generate_with_art_preset,
                        inputs=[ap_prompt, ap_preset, ap_seed, ap_mode],
                        outputs=[ap_gallery, ap_info],
                    )

                with gr.Tab("NSFW 画像"):
                    gr.Markdown(
                        "**NSFW画像生成** — AV品質 / グラビア / アニメ / フェティッシュ\n\n"
                        "fal.ai → Dezgo → Novita の自動フォールバックでブロックされません。\n"
                        "自動アップスケールで超高画質出力も可能。"
                    )
                    nsfw_preset_sel = gr.Dropdown(
                        choices=list(NSFW_PRESETS.keys()),
                        value=list(NSFW_PRESETS.keys())[0],
                        label="NSFWプリセット",
                    )
                    nsfw_prompt_input = gr.Textbox(
                        label="追加プロンプト (被写体・ポーズ・シーン)",
                        placeholder="例: beautiful japanese woman, lying on bed, looking at camera, seductive smile",
                        lines=3,
                    )
                    with gr.Row():
                        nsfw_seed_input = gr.Number(value=-1, label="Seed (-1=random)")
                        nsfw_upscale_check = gr.Checkbox(value=False, label="自動2xアップスケール (超高画質)")
                    nsfw_gen_btn = gr.Button("NSFW画像を生成 (自動フォールバック)", variant="primary", size="lg")
                    nsfw_gallery_out = gr.Gallery(label="結果", columns=2, height=512)
                    nsfw_info_out = gr.Textbox(label="Info", lines=3, interactive=False)

                    # Preset details
                    nsfw_detail_md = gr.Markdown("")
                    def show_nsfw_detail(key):
                        p = NSFW_PRESETS.get(key, {})
                        fb = p.get("fallback_provider", "dezgo")
                        return (
                            f"**推奨モデル**: {', '.join(p.get('recommended_models', []))}\n"
                            f"**推奨LoRA** (ローカル): {', '.join(p.get('recommended_loras', []))}\n"
                            f"**サイズ**: {p.get('settings', {}).get('width', '?')}x{p.get('settings', {}).get('height', '?')} | "
                            f"Steps: {p.get('settings', {}).get('steps', '?')} | "
                            f"フォールバック: {fb}\n"
                            f"**Base prompt**: {p.get('prompt_base', '')[:120]}..."
                        )
                    nsfw_preset_sel.change(fn=show_nsfw_detail, inputs=[nsfw_preset_sel], outputs=[nsfw_detail_md])

                    nsfw_gen_btn.click(
                        fn=generate_nsfw_with_fallback,
                        inputs=[nsfw_prompt_input, nsfw_preset_sel, nsfw_seed_input, nsfw_upscale_check],
                        outputs=[nsfw_gallery_out, nsfw_info_out],
                    )

                with gr.Tab("NSFW 動画"):
                    gr.Markdown(
                        "**NSFW動画生成** — Wan (NSFW OK) で無検閲動画を生成\n\n"
                        "txt2vid (プロンプトから) または img2vid (画像から動画化) が選べます。"
                    )
                    nv_preset = gr.Dropdown(
                        choices=list(NSFW_VIDEO_PRESETS.keys()),
                        value=list(NSFW_VIDEO_PRESETS.keys())[0],
                        label="動画プリセット",
                    )
                    nv_prompt = gr.Textbox(
                        label="追加プロンプト (動きや詳細を記述)",
                        placeholder="例: slowly removing clothes, looking at camera, soft smile",
                        lines=3,
                    )
                    nv_image = gr.Image(label="入力画像 (img2vid の場合 - 任意)", type="filepath")
                    nv_btn = gr.Button("NSFW動画を生成", variant="primary", size="lg")
                    nv_video = gr.Video(label="結果", height=400)
                    nv_info = gr.Textbox(label="Info", interactive=False)

                    nv_btn.click(
                        fn=generate_nsfw_video_preset,
                        inputs=[nv_prompt, nv_preset, nv_image],
                        outputs=[nv_video, nv_info],
                    )

                with gr.Tab("HQ 高品質生成"):
                    gr.Markdown(
                        "**最高品質画像生成** — 最大パラメータ + 自動アップスケールで超高画質\n\n"
                        "生成後に自動で2xアップスケールを適用し、最大2432x2432の画像を出力。"
                    )
                    hq_prompt = gr.Textbox(label="プロンプト", lines=3, placeholder="詳細なプロンプトほど高品質")
                    hq_neg = gr.Textbox(label="Negative Prompt", value="ugly, deformed, blurry, low quality, worst quality, bad anatomy, extra fingers, watermark, text, logo, jpeg artifacts", lines=2)
                    hq_model = gr.Dropdown(
                        choices=list(FAL_MODELS.keys()),
                        value="Flux Realism (フォトリアル・NSFW OK)",
                        label="モデル",
                    )
                    with gr.Row():
                        hq_w = gr.Slider(512, 2048, value=1216, step=64, label="Width")
                        hq_h = gr.Slider(512, 2048, value=1216, step=64, label="Height")
                    with gr.Row():
                        hq_steps = gr.Slider(20, 50, value=40, step=1, label="Steps (高い=高品質)")
                        hq_guidance = gr.Slider(1.0, 10.0, value=4.0, step=0.5, label="Guidance")
                    with gr.Row():
                        hq_seed = gr.Number(value=-1, label="Seed")
                        hq_batch = gr.Slider(1, 4, value=1, step=1, label="枚数")
                    hq_upscale = gr.Checkbox(value=True, label="自動2xアップスケール (生成後にAI拡大)")
                    hq_mode = gr.Radio(choices=["normal", "adult"], value="normal", label="保存先")
                    hq_btn = gr.Button("最高品質で生成", variant="primary", size="lg")
                    hq_gallery = gr.Gallery(label="結果", columns=2, height=512)
                    hq_info = gr.Textbox(label="Info", lines=3, interactive=False)

                    hq_btn.click(
                        fn=generate_hq_image,
                        inputs=[hq_prompt, hq_neg, hq_model, hq_w, hq_h, hq_steps, hq_guidance, hq_seed, hq_batch, hq_upscale, hq_mode],
                        outputs=[hq_gallery, hq_info],
                    )

        # ── Face Swap Tab ──
        with gr.Tab("Face Swap"):
            gr.Markdown(
                "**Face Swap** — 顔の入れ替え。fal.aiクラウド処理（Mac負荷ゼロ）\n\n"
                "1. ベース画像（体・シーン）をアップロード\n"
                "2. 顔画像（入れ替えたい顔）をアップロード\n"
                "3. 「Face Swap 実行」を押す"
            )
            with gr.Row():
                fs_base = gr.Image(label="ベース画像（体・シーン）", type="filepath")
                fs_face = gr.Image(label="顔画像（入れ替える顔）", type="filepath")
            with gr.Row():
                fs_mode = gr.Radio(choices=["normal", "adult"], value="normal", label="保存先", scale=1)
                fs_btn = gr.Button("Face Swap 実行", variant="primary", size="lg", scale=2)
            fs_result = gr.Image(label="結果", height=512)
            fs_info = gr.Textbox(label="Info", interactive=False)

            fs_btn.click(
                fn=generate_face_swap,
                inputs=[fs_base, fs_face, fs_mode],
                outputs=[fs_result, fs_info],
            )

        # ── AI Assistant Tab ──
        with gr.Tab("AI Assistant"):
            gr.Markdown("**AI アシスタント** — 戦略相談・プロンプト提案・モデル選択・トレーニング計画")

            with gr.Row():
                ai_provider = gr.Dropdown(
                    choices=list(PROVIDERS.keys()),
                    value="Claude (高品質アドバイス)",
                    label="AI プロバイダー",
                    scale=2,
                )
                ai_model_override = gr.Dropdown(
                    choices=["auto", "claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001", "gpt-4.1", "gpt-4.1-mini", "grok-3", "grok-3-mini"],
                    value="auto",
                    label="モデル (auto=プロバイダーのデフォルト)",
                    scale=2,
                )

            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(label="AI Advisor", height=500)
                    with gr.Row():
                        chat_input = gr.Textbox(
                            label="質問・相談",
                            placeholder="例: リアル系ポートレートのプロンプトを提案して / CivitAIで売れるLoRAのアイデアは？",
                            lines=2,
                            scale=4,
                        )
                        chat_btn = gr.Button("送信", variant="primary", scale=1)
                    clear_btn = gr.Button("会話クリア", size="sm")

                with gr.Column(scale=1):
                    gr.Markdown("### クイック質問")
                    for i, q in enumerate(QUICK_QUESTIONS):
                        quick_btn = gr.Button(q, size="sm")
                        quick_btn.click(
                            fn=lambda msg, h, prov, mo, qq=q: (
                                h + [(qq, chat_with_ai(qq, h, config, prov, mo))],
                                "",
                            ),
                            inputs=[chat_input, chatbot, ai_provider, ai_model_override],
                            outputs=[chatbot, chat_input],
                        )

            def chat_respond(message, history, provider, model_override):
                if not message.strip():
                    return history, ""
                response = chat_with_ai(message, history, config, provider, model_override)
                history.append((message, response))
                return history, ""

            chat_btn.click(fn=chat_respond, inputs=[chat_input, chatbot, ai_provider, ai_model_override], outputs=[chatbot, chat_input])
            chat_input.submit(fn=chat_respond, inputs=[chat_input, chatbot, ai_provider, ai_model_override], outputs=[chatbot, chat_input])
            clear_btn.click(fn=lambda: ([], ""), outputs=[chatbot, chat_input])

        # ── Prompt Generator Tab ──
        with gr.Tab("Prompt Generator"):
            gr.Markdown("**プロンプト生成** — タグビルダーで組み立て or AIに自動生成させる")

            with gr.Tabs():
                # ─ Tag Builder ─
                with gr.Tab("Tag Builder (タグビルダー)"):
                    gr.Markdown("カテゴリからタグを選んでプロンプトを組み立てます。選択後「プロンプト組み立て」を押してください。")

                    with gr.Row():
                        with gr.Column():
                            tb_quality = gr.CheckboxGroup(
                                label="品質",
                                choices=["masterpiece", "best quality", "highly detailed", "ultra detailed",
                                         "photorealistic", "raw photo", "8k uhd", "film grain",
                                         "sharp focus", "professional photography"],
                            )
                            tb_subject = gr.CheckboxGroup(
                                label="被写体",
                                choices=["1girl", "1boy", "1woman", "1man", "couple", "group",
                                         "landscape", "cityscape", "still life", "animal", "fantasy creature"],
                            )
                            tb_face = gr.CheckboxGroup(
                                label="顔・表情",
                                choices=["beautiful face", "detailed face", "perfect face", "natural skin texture",
                                         "smile", "serious", "looking at camera", "looking away",
                                         "closed eyes", "expressive eyes", "detailed eyes"],
                            )
                        with gr.Column():
                            tb_hair = gr.CheckboxGroup(
                                label="髪",
                                choices=["long hair", "short hair", "medium hair", "ponytail", "twin tails",
                                         "braid", "black hair", "brown hair", "blonde hair", "red hair",
                                         "white hair", "blue hair", "pink hair", "gradient hair"],
                            )
                            tb_body = gr.CheckboxGroup(
                                label="体型・服装",
                                choices=["slender", "curvy", "muscular", "petite",
                                         "dress", "casual outfit", "school uniform", "kimono",
                                         "swimsuit", "lingerie", "armor", "suit",
                                         "nude", "topless", "bare shoulders"],
                            )
                            tb_pose = gr.CheckboxGroup(
                                label="ポーズ・アクション",
                                choices=["standing", "sitting", "lying down", "walking", "running",
                                         "dancing", "fighting", "leaning", "stretching",
                                         "arms up", "hand on hip", "peace sign"],
                            )
                        with gr.Column():
                            tb_background = gr.CheckboxGroup(
                                label="背景・場所",
                                choices=["simple background", "white background", "detailed background",
                                         "outdoor", "indoor", "beach", "forest", "city street",
                                         "classroom", "bedroom", "temple", "castle",
                                         "space", "underwater", "flower garden"],
                            )
                            tb_lighting = gr.CheckboxGroup(
                                label="照明・雰囲気",
                                choices=["natural lighting", "golden hour", "sunset", "studio lighting",
                                         "dramatic lighting", "soft lighting", "backlighting",
                                         "neon lights", "candlelight", "moonlight",
                                         "cinematic", "moody", "warm tones", "cool tones"],
                            )
                            tb_camera = gr.CheckboxGroup(
                                label="カメラ・構図",
                                choices=["close-up", "portrait", "upper body", "cowboy shot", "full body",
                                         "from above", "from below", "from side", "dutch angle",
                                         "wide angle", "85mm lens", "bokeh", "depth of field",
                                         "shallow depth of field"],
                            )
                            tb_style = gr.CheckboxGroup(
                                label="スタイル",
                                choices=["anime style", "illustration", "cel shading", "watercolor",
                                         "oil painting", "digital art", "concept art", "pop art",
                                         "cyberpunk", "steampunk", "gothic", "art nouveau",
                                         "ukiyo-e", "vaporwave"],
                            )

                    with gr.Row():
                        tb_custom = gr.Textbox(label="追加タグ (カンマ区切り)", placeholder="custom tag 1, custom tag 2...", lines=1)

                    tb_build_btn = gr.Button("プロンプト組み立て", variant="primary", size="lg")

                    tb_result = gr.Textbox(label="生成されたプロンプト", lines=4, interactive=True)
                    tb_neg_result = gr.Textbox(
                        label="推奨 Negative Prompt",
                        value="worst quality, low quality, blurry, deformed, ugly, bad anatomy, bad hands, extra fingers, missing fingers, bad face, distorted face, watermark, text, signature",
                        lines=2, interactive=True,
                    )

                    def build_prompt_from_tags(quality, subject, face, hair, body, pose, bg, light, camera, style, custom):
                        tags = []
                        for group in [quality, subject, face, hair, body, pose, bg, light, camera, style]:
                            if group:
                                tags.extend(group)
                        if custom and custom.strip():
                            tags.extend([t.strip() for t in custom.split(",") if t.strip()])
                        return ", ".join(tags)

                    tb_build_btn.click(
                        fn=build_prompt_from_tags,
                        inputs=[tb_quality, tb_subject, tb_face, tb_hair, tb_body, tb_pose,
                                tb_background, tb_lighting, tb_camera, tb_style, tb_custom],
                        outputs=[tb_result],
                    )

                # ─ AI Prompt Generator ─
                with gr.Tab("AI Prompt (AI自動生成)"):
                    gr.Markdown("イメージを日本語で入力すると、AIが最適なプロンプトを英語で生成します。")

                    with gr.Row():
                        with gr.Column(scale=3):
                            pg_idea = gr.Textbox(
                                label="イメージ・アイデア（日本語OK）",
                                lines=3,
                                placeholder="例: 桜の下で着物を着た美しい女性、春の陽光、写真風\n例: サイバーパンクな東京の夜景、ネオンが反射する雨の路面\n例: かわいいアニメキャラ、猫耳、メイド服",
                            )
                            pg_style_hint = gr.Radio(
                                choices=["リアル写真風", "アニメ・イラスト", "アート・絵画風", "NSFW リアル", "NSFW アニメ", "指定なし"],
                                value="指定なし",
                                label="スタイル方向",
                            )
                        with gr.Column(scale=1):
                            pg_provider = gr.Dropdown(
                                choices=list(PROVIDERS.keys()),
                                value="Claude (高品質アドバイス)",
                                label="AI プロバイダー",
                            )
                            pg_gen_btn = gr.Button("プロンプト生成", variant="primary", size="lg")

                    pg_prompt_out = gr.Textbox(label="生成されたプロンプト (Prompt)", lines=4, interactive=True)
                    pg_neg_out = gr.Textbox(label="生成された Negative Prompt", lines=2, interactive=True)
                    pg_settings_out = gr.Textbox(label="推奨設定", lines=3, interactive=False)

                    def generate_prompt_with_ai(idea, style_hint, provider):
                        if not idea.strip():
                            return "", "", "イメージを入力してください"

                        system = """あなたはStable Diffusion用プロンプトの専門家です。
ユーザーのアイデアを受け取り、最適なプロンプトを生成してください。

回答フォーマット（必ずこの形式で）:
PROMPT: (英語のプロンプト、カンマ区切り)
NEGATIVE: (英語のネガティブプロンプト、カンマ区切り)
SETTINGS: Model=xxx | Steps=xx | CFG=x.x | Size=WxH | Sampler=xxx | Scheduler=xxx

ルール:
- プロンプトは英語で、具体的なタグをカンマ区切りで
- 品質タグを先頭に、被写体→外見→背景→照明の順
- ネガティブプロンプトも適切に
- 設定はスタイルに最適な値を提案
- 余計な説明は不要、上記フォーマットのみ"""

                        if style_hint != "指定なし":
                            idea = f"[スタイル: {style_hint}] {idea}"

                        prov = PROVIDERS.get(provider)
                        if not prov:
                            return "", "", f"不明なプロバイダー: {provider}"
                        api_key = config.get(prov["key_name"], "")
                        if not api_key:
                            return "", "", f"{provider} の API Key が未設定です"

                        try:
                            messages = [{"role": "user", "content": idea}]
                            result = prov["call"](api_key, system, messages)

                            prompt_text = ""
                            neg_text = ""
                            settings_text = ""
                            for line in result.strip().split("\n"):
                                line = line.strip()
                                if line.startswith("PROMPT:"):
                                    prompt_text = line[7:].strip()
                                elif line.startswith("NEGATIVE:"):
                                    neg_text = line[9:].strip()
                                elif line.startswith("SETTINGS:"):
                                    settings_text = line[9:].strip()

                            if not prompt_text:
                                prompt_text = result.strip()

                            return prompt_text, neg_text, settings_text
                        except Exception as e:
                            return "", "", f"エラー: {e}"

                    pg_gen_btn.click(
                        fn=generate_prompt_with_ai,
                        inputs=[pg_idea, pg_style_hint, pg_provider],
                        outputs=[pg_prompt_out, pg_neg_out, pg_settings_out],
                    )

        # ── Guide Tab ──
        with gr.Tab("Guide"):
            gr.Markdown("**ガイド** — プロンプトの書き方・モデル選択・設定・収益化のコツ")

            with gr.Tabs():
                for key, section in GUIDE_SECTIONS.items():
                    with gr.Tab(section["title"]):
                        gr.Markdown(section["content"])

                with gr.Tab("プロンプトテンプレート"):
                    gr.Markdown("## プロンプトテンプレート\n「適用」ボタンでQuick/Adultタブに直接反映されます。")
                    for key, tmpl in PROMPT_TEMPLATES.items():
                        with gr.Group():
                            gr.Markdown(f"### {tmpl['name']}")
                            gr.Textbox(value=tmpl["prompt"], label="Prompt", lines=2, interactive=False)
                            gr.Textbox(value=tmpl["negative"], label="Negative Prompt", lines=1, interactive=False)
                            settings_str = f"Steps: {tmpl['settings']['steps']} | CFG: {tmpl['settings']['cfg']} | Size: {tmpl['settings']['width']}x{tmpl['settings']['height']} | Sampler: {tmpl['settings']['sampler']}"
                            gr.Markdown(f"**設定**: {settings_str}")
                            gr.Markdown(f"**推奨モデル**: {', '.join(tmpl['recommended_models'])}")
                            with gr.Row():
                                _apply_q = gr.Button("→ Quickに適用", size="sm")
                                _apply_a = gr.Button("→ Adultに適用", size="sm")
                            _apply_q.click(
                                fn=lambda t=tmpl: (
                                    gr.update(value=t["prompt"]),
                                    gr.update(value=t["negative"]),
                                    gr.update(value=t["settings"]["steps"]),
                                    gr.update(value=t["settings"]["cfg"]),
                                    gr.update(value=t["settings"]["sampler"]),
                                    gr.update(value=t["settings"]["width"]),
                                    gr.update(value=t["settings"]["height"]),
                                ),
                                inputs=[],
                                outputs=[q_prompt, q_neg, q_steps, q_cfg, q_sampler, q_w, q_h],
                            )
                            _apply_a.click(
                                fn=lambda t=tmpl: (
                                    gr.update(value=t["prompt"]),
                                    gr.update(value=t["negative"]),
                                    gr.update(value=t["settings"]["steps"]),
                                    gr.update(value=t["settings"]["cfg"]),
                                    gr.update(value=t["settings"]["sampler"]),
                                    gr.update(value=t["settings"]["width"]),
                                    gr.update(value=t["settings"]["height"]),
                                ),
                                inputs=[],
                                outputs=[a_prompt, a_neg, a_steps, a_cfg, a_sampler, a_w, a_h],
                            )

        # ── Storyboard Tab ──
        with gr.Tab("Storyboard"):
            gr.Markdown("**Storyboard Studio** — 複数シーンを順次生成してAI映画を作る。Cinema Preset + Color Grading + ナレーション対応。")

            with gr.Row():
                sb_project_name = gr.Textbox(value="Untitled Project", label="Project Name", scale=2)
                sb_model = gr.Dropdown(
                    choices=_get_models_for_backend() + list(FAL_MODELS.keys()),
                    value=list(FAL_MODELS.keys())[0] if FAL_MODELS else None,
                    label="Video/Image Model",
                    scale=2,
                )
                sb_total_info = gr.Textbox(label="Status", interactive=False, scale=1)

            # Scene inputs (up to 6 scenes)
            sb_scenes = []
            for i in range(6):
                with gr.Accordion(f"Scene {i+1}", open=(i == 0)):
                    with gr.Row():
                        sp = gr.Textbox(label=f"Scene {i+1} Prompt", lines=2, placeholder="A woman walks through neon-lit Tokyo...", scale=3)
                        sc = gr.Dropdown(choices=list(CINEMA_PRESETS.keys()), value="Cinematic Blockbuster", label="Camera", scale=1)
                        sg = gr.Dropdown(choices=list(COLOR_GRADE_PRESETS.keys()), value="(なし)", label="Color Grade", scale=1)
                    with gr.Row():
                        sn = gr.Textbox(label="Narration (optional)", placeholder="The city never sleeps...", scale=3)
                        sd = gr.Slider(3, 15, value=5, step=1, label="Duration (sec)", scale=1)
                    sb_scenes.append({"prompt": sp, "cinema": sc, "grade": sg, "narration": sn, "duration": sd})

            sb_gallery = gr.Gallery(label="Generated Scenes", columns=3, height=400)
            sb_output_info = gr.Textbox(label="Output", interactive=False, lines=3)

            with gr.Row():
                sb_generate_btn = gr.Button("Generate All Scenes", variant="primary", size="lg")
                sb_stitch_btn = gr.Button("Stitch Video (ffmpeg)", variant="secondary", size="lg")

            def generate_storyboard(project_name, model, *scene_args):
                """Generate all storyboard scenes sequentially."""
                # Unpack scene args (5 per scene: prompt, cinema, grade, narration, duration)
                all_images = []
                output_lines = [f"Project: {project_name}"]
                scene_count = len(scene_args) // 5

                for i in range(scene_count):
                    base = i * 5
                    s_prompt = scene_args[base]
                    s_cinema = scene_args[base + 1]
                    s_grade = scene_args[base + 2]
                    s_narration = scene_args[base + 3]
                    s_duration = scene_args[base + 4]

                    if not s_prompt or not s_prompt.strip():
                        continue

                    # Apply cinema preset
                    final_prompt = s_prompt
                    if s_cinema and s_cinema != "(なし)" and s_cinema in CINEMA_PRESETS:
                        final_prompt += CINEMA_PRESETS[s_cinema]

                    output_lines.append(f"Scene {i+1}: generating...")

                    try:
                        images, info = generate_image(
                            final_prompt, config["default_negative_prompt"],
                            model, "None", 0.8, "None",
                            config["default_width"], config["default_height"],
                            25, 7, "dpmpp_2m_sde", "karras", -1, 1,
                            False, 1.5, 0.5, 15, "None", "normal",
                            False, 0.4, 512,
                        )
                        # Apply color grade
                        if s_grade and s_grade != "(なし)" and images:
                            images = [apply_color_grade(img, s_grade) if isinstance(img, Image.Image) else img for img in images]

                        all_images.extend(images)
                        output_lines.append(f"Scene {i+1}: done ({info})")
                    except Exception as e:
                        output_lines.append(f"Scene {i+1}: FAILED - {e}")

                status = f"{len(all_images)} scenes generated"
                return all_images, "\n".join(output_lines), status

            # Collect all scene inputs
            all_scene_inputs = []
            for s in sb_scenes:
                all_scene_inputs.extend([s["prompt"], s["cinema"], s["grade"], s["narration"], s["duration"]])

            sb_generate_btn.click(
                fn=generate_storyboard,
                inputs=[sb_project_name, sb_model] + all_scene_inputs,
                outputs=[sb_gallery, sb_output_info, sb_total_info],
            )

            def stitch_scenes():
                """Stitch generated scene images into a video slideshow using ffmpeg."""
                output_dir = config["output_dir_normal"]
                # Get latest images
                images = sorted(
                    [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith((".png", ".jpg"))],
                    key=os.path.getmtime, reverse=True
                )[:6]  # Last 6 images

                if len(images) < 2:
                    return "Need at least 2 generated scenes to stitch."

                output_path = os.path.join(output_dir, f"storyboard_{int(time.time())}.mp4")
                # Create ffmpeg slideshow (5 seconds per image with fade transitions)
                list_file = os.path.join(output_dir, "concat_list.txt")
                with open(list_file, "w") as f:
                    for img in images:
                        f.write(f"file '{img}'\nduration 5\n")
                    f.write(f"file '{images[-1]}'\n")  # Last image needs no duration

                try:
                    subprocess.run([
                        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file,
                        "-vf", "scale=1024:768:force_original_aspect_ratio=decrease,pad=1024:768:-1:-1:color=black",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30",
                        output_path
                    ], capture_output=True, text=True, timeout=60)
                    os.remove(list_file)
                    return f"Video saved: {output_path}"
                except Exception as e:
                    return f"ffmpeg error: {e}"

            sb_stitch_btn.click(fn=stitch_scenes, outputs=[sb_output_info])

        # ── Settings Tab ──
        with gr.Tab("Settings"):
            gr.Markdown("**設定** — バックエンド・保存先・Google Drive連携・RunPod・CivitAI")

            with gr.Group():
                gr.Markdown("### Backend (バックエンド切り替え)")
                with gr.Row():
                    s_backend = gr.Radio(
                        choices=["fal", "vast", "replicate", "together", "dezgo", "novita", "civitai", "local"],
                        value=config.get("backend", "local"),
                        label="実行先",
                        info="fal=Flux推奨 / vast=CivitAIモデル自由 / replicate=万能 / together=LoRA / dezgo=無検閲 / novita=最安 / civitai=SDXL / local=Mac",
                    )
                    backend_status = gr.Textbox(label="切り替え状況", interactive=False)
                s_backend.change(fn=lambda choice: switch_backend(choice)[0], inputs=[s_backend], outputs=[backend_status])

            with gr.Group():
                gr.Markdown("### RunPod Cloud GPU")
                gr.Markdown(
                    "RunPodのAPI KeyはRunPodダッシュボードの Settings > API Keys から取得してください。\n\n"
                    "GPU を選んで「Cloud GPU 起動」を押すとComfyUI付きPodが作成されます。\n"
                    "起動後、上部の Backend を `runpod` に切り替えてください。"
                )
                s_runpod_key = gr.Textbox(
                    label="RunPod API Key",
                    value=config.get("runpod_api_key", ""),
                    type="password",
                )
                runpod_template_select = gr.Dropdown(
                    choices=list(RunPodManager.TEMPLATES.keys()),
                    value="ComfyUI + Flux (推奨)",
                    label="テンプレート",
                    info="Flux付き=モデルDL不要。SD+ComfyUI=従来のSD系モデル対応。",
                )
                runpod_gpu_select = gr.Dropdown(
                    choices=[
                        "NVIDIA RTX A5000",         # 24GB $0.16/hr - コスパ最高
                        "NVIDIA GeForce RTX 3090",  # 24GB $0.22/hr
                        "NVIDIA GeForce RTX 4080",  # 16GB $0.27/hr
                        "NVIDIA GeForce RTX 4090",  # 24GB $0.44/hr - 最速
                        "NVIDIA RTX A4000",          # 16GB $0.17/hr
                        "NVIDIA RTX A6000",          # 48GB $0.33/hr - 大VRAM
                    ],
                    value="NVIDIA RTX A5000",
                    label="GPU タイプ",
                    info="A5000(24GB,$0.16/hr)がコスパ最高。4090が最速。A6000(48GB)はFlux FP16向き。",
                )
                runpod_status = gr.Textbox(label="RunPod Status (コスト情報含む)", lines=6, interactive=False)
                with gr.Row():
                    runpod_start_btn = gr.Button("Cloud GPU 起動", variant="primary")
                    runpod_stop_btn = gr.Button("Cloud GPU 停止", variant="stop")
                    runpod_check_btn = gr.Button("状態確認 / 接続")
                with gr.Row():
                    runpod_avail_btn = gr.Button("空きGPU確認")
                    runpod_retry_btn = gr.Button("空き待ちリトライ (最大10分)")

                def check_available_gpus():
                    if not runpod.api_key:
                        return "API Key が未設定です"
                    try:
                        gpus = runpod.check_gpu_availability()
                        lines = ["現在利用可能なGPU (16GB+VRAM, 安い順):"]
                        lines.append(f"{'GPU名':35s} {'VRAM':>5s} {'Community':>10s} {'Secure':>10s}")
                        lines.append("-" * 65)
                        for g in gpus[:15]:
                            comm = f"${g['communityPrice']:.2f}/hr" if g.get('communityPrice') else "N/A"
                            sec = f"${g['securePrice']:.2f}/hr" if g.get('securePrice') else "N/A"
                            lines.append(f"{g['displayName']:35s} {g['memoryInGb']:>3.0f}GB {comm:>10s} {sec:>10s}")
                        if not gpus:
                            lines.append("利用可能なGPUが見つかりません。")
                        return "\n".join(lines)
                    except Exception as e:
                        return f"確認エラー: {e}"

                def runpod_retry_until_available(gpu_choice, template_choice):
                    """Keep retrying pod creation every 30s for up to 10 minutes."""
                    if not runpod.api_key:
                        return "API Key が未設定です"
                    max_attempts = 20  # 20 * 30s = 10 minutes
                    for attempt in range(1, max_attempts + 1):
                        try:
                            new_pod = runpod.create_pod(
                                gpu_type_id=gpu_choice,
                                auto_fallback=True,
                                template_key=template_choice,
                            )
                            config["runpod_pod_id"] = new_pod["id"]
                            save_config(config)
                            actual_gpu = new_pod.get("machine", {}).get("gpuDisplayName", gpu_choice)
                            # Wait for ready
                            for _ in range(40):
                                time.sleep(3)
                                pod = runpod.get_pod(new_pod["id"])
                                if pod and pod["desiredStatus"] == "RUNNING":
                                    url = runpod.get_comfyui_url(pod)
                                    if url:
                                        config["runpod_comfyui_url"] = url
                                        config["comfyui_url"] = url
                                        config["backend"] = "runpod"
                                        client.server_url = url
                                        save_config(config)
                                        cost_info = format_pod_cost(pod)
                                        return (
                                            f"GPU確保成功! ({attempt}回目の試行)\n"
                                            f"GPU: {actual_gpu}\n"
                                            f"{cost_info}\n"
                                            f"URL: {url}\n"
                                            f"Backend を runpod に自動切替しました。"
                                        )
                            return f"Pod作成済み（ID: {new_pod['id']}）。「状態確認」で接続してください。"
                        except RuntimeError as e:
                            if "在庫切れ" in str(e):
                                if attempt < max_attempts:
                                    time.sleep(30)
                                    continue
                                return (
                                    f"10分間リトライしましたがGPU確保できませんでした。\n"
                                    f"RunPodが非常に混雑しています。\n\n"
                                    f"対処法:\n"
                                    f"・しばらく待って再試行\n"
                                    f"・日本の朝〜昼（米国深夜）が空きやすい\n"
                                    f"・ローカル (Mac MPS) で画像生成は可能"
                                )
                            return f"エラー: {e}"
                    return "リトライ上限に達しました"

                runpod_start_btn.click(fn=runpod_start, inputs=[runpod_gpu_select, runpod_template_select], outputs=[runpod_status])
                runpod_stop_btn.click(fn=runpod_stop, outputs=[runpod_status])
                runpod_check_btn.click(fn=runpod_check_status, outputs=[runpod_status])
                runpod_avail_btn.click(fn=check_available_gpus, outputs=[runpod_status])
                runpod_retry_btn.click(fn=runpod_retry_until_available, inputs=[runpod_gpu_select, runpod_template_select], outputs=[runpod_status])

            with gr.Group():
                gr.Markdown("### Vast.ai Cloud GPU (CivitAIモデル自由使用)")
                gr.Markdown(
                    "**Vast.ai** — RunPodより安い。CivitAIモデルを自由にDLして使える。\n\n"
                    "1. [vast.ai](https://vast.ai/) でアカウント作成 → API Key取得\n"
                    "2. 下でGPU検索 → インスタンス作成\n"
                    "3. モデルパック選択で推奨NSFWモデルが自動DL\n"
                    "4. Backend を `vast` に切り替え"
                )
                s_vast_key = gr.Textbox(
                    label="Vast.ai API Key",
                    value=config.get("vast_api_key", ""),
                    type="password",
                    info="https://cloud.vast.ai/api/ から取得",
                )

                with gr.Row():
                    vast_max_price = gr.Slider(0.1, 2.0, value=0.5, step=0.05, label="最大料金 ($/hr)")
                    vast_min_vram = gr.Slider(8, 48, value=16, step=4, label="最小VRAM (GB)")

                vast_offers_display = gr.Textbox(label="利用可能なGPU", lines=8, interactive=False)
                vast_search_btn = gr.Button("GPU検索", variant="secondary")

                def vast_search_gpus(max_price, min_vram):
                    if not vastai.api_key:
                        return "Vast.ai API Key が未設定です"
                    try:
                        offers = vastai.search_offers(min_gpu_ram=int(min_vram), max_price=float(max_price))
                        return vastai.format_offers(offers)
                    except Exception as e:
                        return f"検索エラー: {e}"

                vast_search_btn.click(fn=vast_search_gpus, inputs=[vast_max_price, vast_min_vram], outputs=[vast_offers_display])

                gr.Markdown("#### インスタンス作成")
                with gr.Row():
                    vast_offer_id = gr.Number(label="Offer ID (上の検索結果から)", precision=0)
                    vast_image = gr.Dropdown(
                        choices=list(COMFYUI_IMAGES.keys()),
                        value=list(COMFYUI_IMAGES.keys())[0],
                        label="Docker Image",
                    )
                    vast_model_pack = gr.Radio(
                        choices=["starter", "full", "none"],
                        value="starter",
                        label="モデルパック",
                        info="starter=RealVisXL+基本(7GB) / full=全推奨モデル(27GB) / none=モデルDLなし",
                    )
                vast_disk = gr.Slider(20, 100, value=50, step=10, label="ディスク容量 (GB)")

                vast_status = gr.Textbox(label="Vast.ai Status", lines=6, interactive=False)
                with gr.Row():
                    vast_create_btn = gr.Button("インスタンス作成", variant="primary")
                    vast_stop_btn = gr.Button("停止", variant="stop")
                    vast_start_btn = gr.Button("再開")
                    vast_destroy_btn = gr.Button("削除 (永久)", variant="stop")
                vast_check_btn = gr.Button("状態確認 / 接続")

                def vast_create_instance(offer_id, image_key, model_pack, disk):
                    if not vastai.api_key:
                        return "Vast.ai API Key が未設定です"
                    if not offer_id or offer_id <= 0:
                        return "Offer IDを入力してください（GPU検索結果の右端の数字）"
                    try:
                        result = vastai.create_instance(
                            offer_id=int(offer_id),
                            image_key=image_key,
                            disk_gb=int(disk),
                            model_pack=model_pack,
                        )
                        inst_id = result.get("new_contract")
                        if inst_id:
                            config["vast_instance_id"] = inst_id
                            save_config(config)
                            return (
                                f"インスタンス作成成功! ID: {inst_id}\n"
                                f"モデルパック: {model_pack}\n"
                                f"起動中... 「状態確認/接続」で接続してください。\n"
                                f"（モデルDL含め3-10分かかります）"
                            )
                        return f"作成結果: {json.dumps(result, indent=2)}"
                    except Exception as e:
                        return f"作成エラー: {e}"

                def vast_check_status():
                    if not vastai.api_key:
                        return "Vast.ai API Key が未設定です"
                    try:
                        instances = vastai.get_instances()
                        if not instances:
                            return "稼働中のインスタンスはありません。"

                        lines = [format_cost_summary(instances), ""]
                        for inst in instances:
                            lines.append(format_instance_status(inst))
                            lines.append("")

                            # Try to connect to ComfyUI
                            if inst.get("actual_status") == "running":
                                url = vastai.get_comfyui_url(inst)
                                if url:
                                    try:
                                        urllib.request.urlopen(f"{url}/system_stats", timeout=5)
                                        config["vast_comfyui_url"] = url
                                        config["comfyui_url"] = url
                                        config["vast_instance_id"] = inst.get("id")
                                        client.server_url = url
                                        save_config(config)
                                        lines.append(f"ComfyUI接続済み: {url}")
                                    except Exception:
                                        lines.append(f"ComfyUI起動待ち... URL: {url}")

                        return "\n".join(lines)
                    except Exception as e:
                        return f"確認エラー: {e}"

                def vast_stop():
                    inst_id = config.get("vast_instance_id")
                    if not inst_id:
                        return "インスタンスIDが不明です。状態確認を先に実行してください。"
                    try:
                        vastai.stop_instance(inst_id)
                        return f"インスタンス {inst_id} を停止しました。"
                    except Exception as e:
                        return f"停止エラー: {e}"

                def vast_start():
                    inst_id = config.get("vast_instance_id")
                    if not inst_id:
                        return "インスタンスIDが不明です。"
                    try:
                        vastai.start_instance(inst_id)
                        return f"インスタンス {inst_id} を再開しました。接続は「状態確認」で。"
                    except Exception as e:
                        return f"再開エラー: {e}"

                def vast_destroy():
                    inst_id = config.get("vast_instance_id")
                    if not inst_id:
                        return "インスタンスIDが不明です。"
                    try:
                        vastai.destroy_instance(inst_id)
                        config.pop("vast_instance_id", None)
                        config.pop("vast_comfyui_url", None)
                        save_config(config)
                        return f"インスタンス {inst_id} を完全削除しました。"
                    except Exception as e:
                        return f"削除エラー: {e}"

                vast_create_btn.click(fn=vast_create_instance,
                                      inputs=[vast_offer_id, vast_image, vast_model_pack, vast_disk],
                                      outputs=[vast_status])
                vast_check_btn.click(fn=vast_check_status, outputs=[vast_status])
                vast_stop_btn.click(fn=vast_stop, outputs=[vast_status])
                vast_start_btn.click(fn=vast_start, outputs=[vast_status])
                vast_destroy_btn.click(fn=vast_destroy, outputs=[vast_status])

                gr.Markdown("#### 推奨CivitAIモデル一覧")
                _model_info_lines = []
                for name, info in VAST_MODELS.items():
                    _model_info_lines.append(f"- **{name}** ({info['type']}, {info['size']}) — {info['desc']}")
                gr.Markdown("\n".join(_model_info_lines))

                gr.Markdown("#### カスタムCivitAIモデルDL")
                gr.Markdown("CivitAI URLからモデルをDLするwgetコマンドを生成。SSHでインスタンスに接続して実行。")
                with gr.Row():
                    vast_civitai_url = gr.Textbox(label="CivitAI Download URL", placeholder="https://civitai.com/api/download/models/...")
                    vast_model_type = gr.Dropdown(
                        choices=["checkpoint", "lora", "vae", "embedding", "controlnet", "upscaler"],
                        value="checkpoint", label="モデルタイプ",
                    )
                vast_dl_cmd_output = gr.Textbox(label="実行コマンド (SSHで実行)", lines=2, interactive=False)

                def gen_dl_cmd(url, mtype):
                    if not url:
                        return ""
                    return vastai.generate_civitai_download_command(url, mtype)

                vast_civitai_url.change(fn=gen_dl_cmd, inputs=[vast_civitai_url, vast_model_type], outputs=[vast_dl_cmd_output])
                vast_model_type.change(fn=gen_dl_cmd, inputs=[vast_civitai_url, vast_model_type], outputs=[vast_dl_cmd_output])

            with gr.Group():
                gr.Markdown("### API Keys")
                s_replicate_key = gr.Textbox(
                    label="Replicate API Key (推奨 - Flux/SDXL/動画生成)",
                    value=config.get("replicate_api_key", ""),
                    type="password",
                    info="https://replicate.com/account/api-tokens",
                )
                s_fal_key = gr.Textbox(
                    label="fal.ai API Key (Flux高品質・NSFW OK)",
                    value=config.get("fal_api_key", ""),
                    type="password",
                    info="https://fal.ai/dashboard/keys",
                )
                s_together_key = gr.Textbox(
                    label="Together.ai API Key (Flux・LoRA・NSFW OK)",
                    value=config.get("together_api_key", ""),
                    type="password",
                    info="https://api.together.ai/settings/api-keys",
                )
                s_dezgo_key = gr.Textbox(
                    label="Dezgo API Key (完全無検閲・画像+動画)",
                    value=config.get("dezgo_api_key", ""),
                    type="password",
                    info="https://dezgo.com/account",
                )
                s_novita_key = gr.Textbox(
                    label="Novita.ai API Key (無検閲モデル・最安)",
                    value=config.get("novita_api_key", ""),
                    type="password",
                    info="https://novita.ai/dashboard/key",
                )
                s_civitai_key = gr.Textbox(
                    label="CivitAI API Key",
                    value=config.get("civitai_api_key", ""),
                    type="password",
                )
                s_anthropic_key = gr.Textbox(
                    label="Anthropic API Key (Claude - 高品質アドバイス)",
                    value=config.get("anthropic_api_key", ""),
                    type="password",
                    info="https://console.anthropic.com/",
                )
                s_openai_key = gr.Textbox(
                    label="OpenAI API Key (GPT - 一般相談)",
                    value=config.get("openai_api_key", ""),
                    type="password",
                    info="https://platform.openai.com/api-keys",
                )
                s_xai_key = gr.Textbox(
                    label="xAI API Key (Grok - NSFW対応)",
                    value=config.get("xai_api_key", ""),
                    type="password",
                    info="https://console.x.ai/",
                )

            with gr.Group():
                gr.Markdown("### Output Directories")
                s_output_normal = gr.Textbox(label="Normal 保存先", value=config["output_dir_normal"])
                s_output_adult = gr.Textbox(label="Adult 保存先", value=config["output_dir_adult"])
                with gr.Row():
                    open_normal_btn = gr.Button("Normal フォルダを開く")
                    open_adult_btn = gr.Button("Adult フォルダを開く")
                    open_models_btn = gr.Button("Models フォルダを開く")

            with gr.Group():
                gr.Markdown("### Google Drive 連携")
                gr.Markdown(
                    "Google Driveデスクトップアプリをインストール後、\n"
                    "Google Drive内にモデル用フォルダを作成し、パスを指定してください。\n\n"
                    "例: `/Users/koachmedia/Library/CloudStorage/GoogleDrive-xxx/My Drive/AI-models`"
                )
                s_gdrive = gr.Textbox(label="Google Drive Models Path", value=config.get("google_drive_models_dir", ""))
                gdrive_link_btn = gr.Button("Google Drive モデルをリンク")
                gdrive_status = gr.Textbox(label="リンク状況", interactive=False)

            with gr.Group():
                gr.Markdown("### iCloud Models")
                gr.Markdown(
                    "iCloudにダウンロードしたCivitAIモデルを直接読み込みます。\n"
                    "フォルダにsafetensorsファイルを入れるだけでOK。\n\n"
                    "例: `~/Library/Mobile Documents/com~apple~CloudDocs/civitai-model`"
                )
                s_icloud = gr.Textbox(label="iCloud Models Path", value=config.get("icloud_models_dir", ""))
                icloud_link_btn = gr.Button("iCloud モデルをリンク")
                icloud_status = gr.Textbox(label="リンク状況", interactive=False)

            with gr.Group():
                gr.Markdown("### Server")
                s_comfyui_url_display = gr.Textbox(label="ComfyUI Server URL (自動設定)", value=config["comfyui_url"], interactive=False)

            with gr.Group():
                gr.Markdown("### Developer Tools (コード編集)")
                gr.Markdown(
                    "アプリのコードを直接編集・カスタマイズしたい場合に使います。\n"
                    "Claude Code または Codex でAIにコード変更を指示できます。"
                )
                dev_status = gr.Textbox(label="", interactive=False)
                with gr.Row():
                    claude_code_btn = gr.Button("Claude Code を起動", variant="primary")
                    codex_btn = gr.Button("Codex (OpenAI) を起動")
                    open_code_btn = gr.Button("VS Code で開く")

                def launch_claude_code():
                    try:
                        app_dir = os.path.dirname(os.path.abspath(__file__))
                        base_dir = os.path.dirname(app_dir)
                        subprocess.Popen(
                            ["osascript", "-e",
                             f'tell application "Terminal" to do script "cd \\"{base_dir}\\" && claude"'],
                        )
                        return "Claude Code を新しいターミナルで起動しました"
                    except Exception as e:
                        return f"起動エラー: {e}\n\n手動で起動:\ncd ~/Desktop/アプリ開発プロジェクト/AI-diffusion && claude"

                def launch_codex():
                    try:
                        app_dir = os.path.dirname(os.path.abspath(__file__))
                        base_dir = os.path.dirname(app_dir)
                        subprocess.Popen(
                            ["osascript", "-e",
                             f'tell application "Terminal" to do script "cd \\"{base_dir}\\" && codex"'],
                        )
                        return "Codex を新しいターミナルで起動しました"
                    except Exception as e:
                        return f"起動エラー: {e}\n\n手動で起動:\ncd ~/Desktop/アプリ開発プロジェクト/AI-diffusion && codex"

                def launch_vscode():
                    try:
                        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                        subprocess.Popen(["code", base_dir])
                        return "VS Code を起動しました"
                    except Exception as e:
                        return f"起動エラー: {e}\n\nVS Codeがインストールされていない場合:\nbrew install --cask visual-studio-code"

                claude_code_btn.click(fn=launch_claude_code, outputs=[dev_status])
                codex_btn.click(fn=launch_codex, outputs=[dev_status])
                open_code_btn.click(fn=launch_vscode, outputs=[dev_status])

            save_btn = gr.Button("設定を保存", variant="primary")
            save_status = gr.Textbox(label="", interactive=False)

            save_btn.click(
                fn=save_settings,
                inputs=[s_output_normal, s_output_adult, s_gdrive, s_icloud, s_comfyui_url_display, s_runpod_key, s_backend, s_replicate_key, s_fal_key, s_together_key, s_dezgo_key, s_novita_key, s_civitai_key, s_anthropic_key, s_openai_key, s_xai_key, s_vast_key],
                outputs=[save_status],
            )
            gdrive_link_btn.click(fn=link_google_drive, inputs=[s_gdrive], outputs=[gdrive_status])
            icloud_link_btn.click(fn=link_icloud_models, inputs=[s_icloud], outputs=[icloud_status])
            open_normal_btn.click(fn=lambda: open_folder(config["output_dir_normal"]), outputs=[save_status])
            open_adult_btn.click(fn=lambda: open_folder(config["output_dir_adult"]), outputs=[save_status])
            open_models_btn.click(fn=lambda: open_folder(config["models_dir"]), outputs=[save_status])

    # ── Session & Shutdown (tabs外、画面下部) ──
    gr.Markdown("---")
    with gr.Row():
        with gr.Column(scale=3):
            with gr.Row():
                session_save_btn = gr.Button("作業を保存", variant="secondary", size="sm")
                session_restore_btn = gr.Button("前回の作業を復元", variant="secondary", size="sm")
            session_status = gr.Textbox(label="", interactive=False, max_lines=1)
        with gr.Column(scale=1):
            shutdown_btn = gr.Button("アプリを終了", variant="stop", size="lg")

    # Session save: capture key inputs from Quick and Adult tabs
    def do_save_session(q_p, q_n, q_m, q_s, q_c, a_p, a_n, a_m, a_s, a_c):
        data = {
            "saved_at": datetime.datetime.now().isoformat(),
            "quick": {"prompt": q_p, "negative": q_n, "model": q_m, "steps": q_s, "cfg": q_c},
            "adult": {"prompt": a_p, "negative": a_n, "model": a_m, "steps": a_s, "cfg": a_c},
        }
        if save_session(data):
            return f"保存完了 ({data['saved_at'][:19]})"
        return "保存に失敗しました"

    def do_restore_session():
        data = load_session()
        if not data:
            return ("", "", None, 25, 7.0, "", "", None, 25, 7.0, "保存データがありません")
        q = data.get("quick", {})
        a = data.get("adult", {})
        saved = data.get("saved_at", "不明")[:19]
        return (
            q.get("prompt", ""), q.get("negative", ""), q.get("model"), q.get("steps", 25), q.get("cfg", 7.0),
            a.get("prompt", ""), a.get("negative", ""), a.get("model"), a.get("steps", 25), a.get("cfg", 7.0),
            f"復元完了 (保存時刻: {saved})",
        )

    def do_shutdown():
        shutdown_app()
        return "シャットダウン中..."

    session_save_btn.click(
        fn=do_save_session,
        inputs=[q_prompt, q_neg, q_model, q_steps, q_cfg,
                a_prompt, a_neg, a_model, a_steps, a_cfg],
        outputs=[session_status],
    )

    session_restore_btn.click(
        fn=do_restore_session,
        inputs=[],
        outputs=[q_prompt, q_neg, q_model, q_steps, q_cfg,
                 a_prompt, a_neg, a_model, a_steps, a_cfg,
                 session_status],
    )

    shutdown_btn.click(fn=do_shutdown, outputs=[session_status])


if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
        allowed_paths=[
            config.get("output_dir_normal", "./outputs/normal"),
            config.get("output_dir_adult", "./outputs/adult"),
            os.path.dirname(os.path.abspath(__file__)),
        ],
    )
