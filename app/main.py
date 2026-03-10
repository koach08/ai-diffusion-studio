"""AI-diffusion Studio - Custom UI with ComfyUI backend."""
import datetime
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
    level=logging.ERROR,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("ai-diffusion")

from config import load_config, save_config, get_available_models, get_available_loras, get_available_vaes, get_available_motion_models, get_available_upscale_models, get_available_unet_models, get_available_clip_models, SAMPLERS, SCHEDULERS
from comfyui_api import ComfyUIClient, build_txt2img_workflow, build_animatediff_workflow, build_img2vid_workflow, build_vid2vid_workflow, build_flux_workflow
from runpod_manager import RunPodManager, format_pod_status, format_pod_cost
from replicate_api import ReplicateClient, MODELS as REPLICATE_MODELS, VIDEO_MODELS as REPLICATE_VIDEO_MODELS, download_url_to_pil
from civitai_api import CivitAIClient, format_search_results, CIVITAI_GENERATION_MODELS
from fal_api import FalClient, FAL_MODELS, FAL_VIDEO_MODELS, download_fal_image
from together_api import TogetherClient, TOGETHER_MODELS, decode_together_image
from dezgo_api import DezgoClient, DEZGO_IMAGE_MODELS, DEZGO_VIDEO_MODELS, decode_dezgo_image
from novita_api import NovitaClient, NOVITA_MODELS, download_novita_image
from guide import GUIDE_SECTIONS, PROMPT_TEMPLATES
from ai_assistant import chat_with_ai, QUICK_QUESTIONS, PROVIDERS

config = load_config()
client = ComfyUIClient(config["comfyui_url"])
runpod = RunPodManager(config.get("runpod_api_key", ""))
civitai = CivitAIClient(config.get("civitai_api_key", ""))
replicate = ReplicateClient(config.get("replicate_api_key", ""))
fal = FalClient(config.get("fal_api_key", ""))
together = TogetherClient(config.get("together_api_key", ""))
dezgo = DezgoClient(config.get("dezgo_api_key", ""))
novita = NovitaClient(config.get("novita_api_key", ""))

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def refresh_models():
    models = get_available_models(config["models_dir"])
    loras = ["None"] + get_available_loras(config["models_dir"])
    vaes = ["None"] + get_available_vaes(config["models_dir"])
    return (
        gr.update(choices=models, value=models[0] if models else None),
        gr.update(choices=loras, value="None"),
        gr.update(choices=vaes, value="None"),
    )


def refresh_all_model_dropdowns():
    """Refresh model/lora/vae dropdowns across ALL tabs after a download."""
    models = get_available_models(config["models_dir"])
    loras = ["None"] + get_available_loras(config["models_dir"])
    vaes = ["None"] + get_available_vaes(config["models_dir"])
    motion = get_available_motion_models()
    m = gr.update(choices=models)
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
                   hires_steps=15, upscale_model="", mode="normal"):
    """Generate image - routes to CivitAI / Replicate API / ComfyUI based on backend."""
    backend = config.get("backend", "local")

    # ── fal.ai backend: Flux quality, NSFW OK ──
    if backend == "fal":
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
            model_key = list(FAL_MODELS.keys())[0]

        try:
            urls = fal.generate_image(
                model_key=model_key,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=w, height=h,
                num_images=int(batch_size),
                seed=int(seed),
                safety_checker=False,
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

        # Pick model: use selected model name if it matches a CivitAI model, else auto
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
            return images, f"[CivitAI: {civitai_model_key}] コスト: {cost}/枚\n保存先: {', '.join(saved_paths)}"
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

    # ── Local / RunPod backend: use ComfyUI ──
    if not client.is_server_running():
        raise gr.Error(
            "ComfyUI Server が起動していません。\n"
            "💡 Replicate バックエンドなら ComfyUI 不要で即生成できます。\n"
            "Settings → Backend を 'replicate' に切り替えてください。"
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
    )

    timeout = 1200 if hires_fix else 900
    images = client.generate(workflow, timeout=timeout)

    output_dir = config["output_dir_adult"] if mode == "adult" else config["output_dir_normal"]
    saved_paths = []
    for img in images:
        path = save_image_to_dir(img, output_dir, prefix=mode)
        saved_paths.append(path)

    hires_info = f" [Hires Fix: {hires_scale}x, denoise={hires_denoise}]" if hires_fix else ""
    return images, f"保存先: {', '.join(saved_paths)}{hires_info}"


# ──────────────────────────────────────────────
# Generate functions for each tab
# ──────────────────────────────────────────────

def generate_normal(prompt, neg, model, lora, lora_str, vae, w, h, steps, cfg, sampler, sched, seed, batch,
                    hires_fix, hires_scale, hires_denoise, hires_steps, upscale_model):
    images, info = generate_image(prompt, neg, model, lora, lora_str, vae, w, h, steps, cfg, sampler, sched, seed, batch,
                                  hires_fix, hires_scale, hires_denoise, hires_steps, upscale_model, "normal")
    return images, info


def generate_adult(prompt, neg, model, lora, lora_str, vae, w, h, steps, cfg, sampler, sched, seed, batch,
                   hires_fix, hires_scale, hires_denoise, hires_steps, upscale_model):
    images, info = generate_image(prompt, neg, model, lora, lora_str, vae, w, h, steps, cfg, sampler, sched, seed, batch,
                                  hires_fix, hires_scale, hires_denoise, hires_steps, upscale_model, "adult")
    return images, info


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
                           frame_count, fps, output_format, mode="normal"):
    """Generate video from text prompt - routes to cloud API or AnimateDiff."""
    backend = config.get("backend", "local")

    # ── Cloud backends: use fal.ai for video ──
    if backend != "local":
        if not prompt.strip():
            raise gr.Error("プロンプトを入力してください。")
        if fal.api_key:
            video_path, info = generate_fal_video(prompt, "LTX 2.3 (高速・安い)", mode)
            return video_path, [], info
        else:
            raise gr.Error(
                "動画生成には fal.ai API Key が必要です。\n"
                "Settingsタブで設定してください。"
            )

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
    return video_path, frames[:4] if frames else [], info


def generate_video_img2vid(image, prompt, neg, model, motion_model, vae,
                           w, h, steps, cfg, sampler, sched, seed,
                           frame_count, fps, denoise, output_format, mode="normal"):
    """Generate video from input image using AnimateDiff img2vid."""
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
    return video_path, frames[:4] if frames else [], info


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


def generate_fal_video(prompt, model_key, mode="normal"):
    """Generate video via fal.ai API."""
    if not fal.api_key:
        raise gr.Error("fal.ai API Key が未設定です。Settingsタブで設定してください。")
    if not prompt.strip():
        raise gr.Error("プロンプトを入力してください。")

    try:
        video_url = fal.generate_video(model_key, prompt)
        if not video_url:
            raise RuntimeError("動画URLが取得できませんでした")

        output_dir = config["output_dir_adult"] if mode == "adult" else config["output_dir_normal"]
        os.makedirs(output_dir, exist_ok=True)
        import datetime as dt
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(output_dir, f"fal_video_{ts}.mp4")
        req = urllib.request.Request(video_url, headers={"User-Agent": "AI-diffusion/1.0"})
        resp = urllib.request.urlopen(req, timeout=120)
        with open(video_path, "wb") as f:
            f.write(resp.read())

        from fal_api import FAL_VIDEO_MODELS as FVM
        cost = FVM.get(model_key, {}).get("cost", "?")
        return video_path, f"[fal.ai {model_key}] コスト: {cost}\n保存先: {video_path}"
    except Exception as e:
        logger.error(f"fal.ai動画生成エラー: {traceback.format_exc()}")
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
        return video_path, f"[Dezgo {model_key}] コスト: {cost}\n保存先: {video_path}"
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
        return video_path, f"[Replicate {model_key}] コスト: {cost}\n保存先: {video_path}"
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

def save_settings(output_normal, output_adult, gdrive_path, comfyui_url, runpod_api_key, backend_choice,
                   replicate_api_key, fal_api_key, together_api_key, dezgo_api_key, novita_api_key,
                   civitai_api_key, anthropic_api_key, openai_api_key, xai_api_key):
    config["output_dir_normal"] = output_normal
    config["output_dir_adult"] = output_adult
    config["google_drive_models_dir"] = gdrive_path
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
    save_config(config)
    client.server_url = comfyui_url.rstrip("/")
    runpod.api_key = runpod_api_key
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


def civitai_download(version_id, model_type_dest):
    """Download model from CivitAI by version ID. Returns (status, *refreshed_dropdowns)."""
    if not version_id:
        return ("Version ID を入力してください（検索結果に表示されます）",) + refresh_all_model_dropdowns()

    dest_map = {
        "Checkpoint": "checkpoints",
        "LoRA": "loras",
        "VAE": "vae",
        "ControlNet": "controlnet",
        "Upscaler": "upscale_models",
        "Embedding": "embeddings",
    }
    dest_dir = os.path.join(config["models_dir"], dest_map.get(model_type_dest, "checkpoints"))
    os.makedirs(dest_dir, exist_ok=True)

    try:
        info = civitai.get_download_url(int(version_id))
        if not info:
            return ("ダウンロードURLが見つかりません",) + refresh_all_model_dropdowns()
        size_gb = info["size_kb"] / 1024 / 1024
        filepath = civitai.download_model(int(version_id), dest_dir)
        msg = f"✅ ダウンロード完了!\nFile: {os.path.basename(filepath)}\nSize: {size_gb:.1f}GB\nPath: {filepath}\n\n全タブのモデルリストを自動更新しました"
        return (msg,) + refresh_all_model_dropdowns()
    except Exception as e:
        return (f"ダウンロードエラー: {e}",) + refresh_all_model_dropdowns()


def download_from_url(url, model_type_dest, custom_filename):
    """Download model from direct URL (HuggingFace, etc.). Returns (status, *refreshed_dropdowns)."""
    if not url or not url.startswith("http"):
        return ("有効なURLを入力してください",) + refresh_all_model_dropdowns()

    dest_map = {
        "Checkpoint": "checkpoints",
        "LoRA": "loras",
        "VAE": "vae",
        "ControlNet": "controlnet",
        "Upscaler": "upscale_models",
        "Embedding": "embeddings",
        "UNET / Diffusion Model": "diffusion_models",
        "CLIP / Text Encoder": "clip",
    }
    dest_dir = os.path.join(config["models_dir"], dest_map.get(model_type_dest, "checkpoints"))
    os.makedirs(dest_dir, exist_ok=True)

    # Determine filename
    if custom_filename and custom_filename.strip():
        filename = custom_filename.strip()
    else:
        filename = url.split("/")[-1].split("?")[0]
        if not filename or "." not in filename:
            filename = "downloaded_model.safetensors"

    filepath = os.path.join(dest_dir, filename)
    if os.path.exists(filepath):
        return (f"既にファイルが存在します: {filepath}",) + refresh_all_model_dropdowns()

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AI-diffusion/1.0"})
        resp = urllib.request.urlopen(req, timeout=600)
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB

        with open(filepath, "wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)

        size_gb = downloaded / (1024 * 1024 * 1024)
        msg = f"✅ ダウンロード完了!\nFile: {filename}\nSize: {size_gb:.2f}GB\nPath: {filepath}\n\n全タブのモデルリストを自動更新しました"
        return (msg,) + refresh_all_model_dropdowns()
    except Exception as e:
        # Clean up partial download
        if os.path.exists(filepath):
            os.remove(filepath)
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

QUALITY_PRESETS = {
    "Quick Test": {"steps": 12, "cfg": 7.0, "hires": False, "batch": 1, "sampler": "euler_ancestral"},
    "Standard": {"steps": 25, "cfg": 7.0, "hires": False, "batch": 1, "sampler": "dpmpp_2m"},
    "High Quality": {"steps": 30, "cfg": 7.0, "hires": True, "batch": 1, "sampler": "dpmpp_2m"},
    "Batch x4": {"steps": 20, "cfg": 7.0, "hires": False, "batch": 4, "sampler": "euler_ancestral"},
}


def build_gen_controls(tab_name):
    """Build the common generation controls for a tab. Returns all input components."""
    models = get_available_models(config["models_dir"])
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
        quality_mode = gr.Radio(
            choices=list(QUALITY_PRESETS.keys()),
            value="Standard",
            label="Quality",
            info="Quick=速い / Standard=通常 / HQ=Hires付き / Batch=4枚",
            scale=2,
        )

    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(label="Prompt", lines=4, placeholder="Describe what you want to generate...")
            negative = gr.Textbox(label="Negative Prompt", lines=2, value=config["default_negative_prompt"])
        with gr.Column(scale=1):
            civitai_model_names = list(CIVITAI_GENERATION_MODELS.keys())
            fal_model_names = list(FAL_MODELS.keys())
            backend = config.get("backend", "local")
            if backend == "fal":
                cloud_choices = fal_model_names + civitai_model_names
                default_val = fal_model_names[0]
                model_info = "fal.ai: モデルを選択 / CivitAIバックエンドならCivitAIモデル"
            elif backend == "civitai":
                cloud_choices = civitai_model_names
                default_val = civitai_model_names[0]
                model_info = "CivitAIクラウドモデルが使われます"
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
            lora = gr.Dropdown(choices=loras, label="LoRA (ローカル専用)", value="None")
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

    with gr.Row():
        generate_btn = gr.Button(f"Generate ({tab_name})", variant="primary", size="lg")
        refresh_btn = gr.Button("Refresh Models", size="sm")

    with gr.Row():
        gallery = gr.Gallery(label="Generated Images", columns=2, height=512)
    info = gr.Textbox(label="Info", interactive=False)

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
        p = QUALITY_PRESETS.get(mode, QUALITY_PRESETS["Standard"])
        return (
            gr.update(value=p["steps"]),
            gr.update(value=p["cfg"]),
            gr.update(value=p["batch"]),
            gr.update(value=p["sampler"]),
            gr.update(value=p["hires"]),
        )

    quality_mode.change(
        fn=apply_quality,
        inputs=[quality_mode],
        outputs=[steps, cfg, batch_size, sampler, hires_fix],
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

    return (prompt, negative, model, lora, lora_strength, vae,
            width, height, steps, cfg, sampler, scheduler, seed, batch_size,
            hires_fix, hires_scale, hires_denoise, hires_steps, upscale_model_dd,
            generate_btn, gallery, info)


# ──────────────────────────────────────────────
# UI Layout
# ──────────────────────────────────────────────

with gr.Blocks(title="AI-diffusion Studio") as app:

    gr.HTML("<h1 class='main-title'>AI-diffusion Studio</h1>")

    with gr.Row():
        status = gr.Textbox(value=check_server_status(), label="Server Status", interactive=False, elem_classes="status-bar", scale=3)
        backend_radio = gr.Radio(
            choices=["fal", "replicate", "together", "dezgo", "novita", "civitai", "local"],
            value=config.get("backend", "local"),
            label="Backend",
            info="fal=Flux推奨 / replicate=万能 / together=LoRA / dezgo=無検閲 / novita=最安 / civitai=SDXL / local=Mac",
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
             q_gen_btn, q_gallery, q_info) = build_gen_controls("Normal")

            q_gen_btn.click(
                fn=generate_normal,
                inputs=[q_prompt, q_neg, q_model, q_lora, q_lora_str, q_vae,
                        q_w, q_h, q_steps, q_cfg, q_sampler, q_sched, q_seed, q_batch,
                        q_hires, q_hires_scale, q_hires_denoise, q_hires_steps, q_upscale_model],
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
            gr.Markdown("**R18 / Adult コンテンツ生成** — 制限なし。出力は専用フォルダに保存されます。")
            (a_prompt, a_neg, a_model, a_lora, a_lora_str, a_vae,
             a_w, a_h, a_steps, a_cfg, a_sampler, a_sched, a_seed, a_batch,
             a_hires, a_hires_scale, a_hires_denoise, a_hires_steps, a_upscale_model,
             a_gen_btn, a_gallery, a_info) = build_gen_controls("Adult")

            a_gen_btn.click(
                fn=generate_adult,
                inputs=[a_prompt, a_neg, a_model, a_lora, a_lora_str, a_vae,
                        a_w, a_h, a_steps, a_cfg, a_sampler, a_sched, a_seed, a_batch,
                        a_hires, a_hires_scale, a_hires_denoise, a_hires_steps, a_upscale_model],
                outputs=[a_gallery, a_info],
            )

        # ── Video Tab ──
        with gr.Tab("Video"):
            gr.Markdown("**動画生成** — AnimateDiff で txt2vid / img2vid / vid2vid。SD1.5モデル対応。")

            # GPU recommendation warning
            video_gpu_warning = gr.Markdown(
                "**GPU推奨**: 動画生成はRunPod (Cloud GPU) での利用を推奨します。"
                "ローカルMacでは1本5-15分かかります。"
                if config.get("backend") == "local" else
                ""
            )

            models = get_available_models(config["models_dir"])
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
                    gr.Markdown("テキストから動画を生成します。SD1.5のチェックポイントを使用してください。")

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
                        v_frames = gr.Slider(8, 32, value=16, step=1, label="Frames")
                        v_fps = gr.Slider(4, 30, value=8, step=1, label="FPS")

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
                                v_frames, v_fps, v_format, v_mode],
                        outputs=[v_video, v_preview, v_info],
                    )

                # ─ img2vid ─
                with gr.Tab("Image to Video (img2vid)"):
                    gr.Markdown("画像をアニメーションに変換します。元画像の構図を保ちつつ動きを加えます。")
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
                        i2v_frames = gr.Slider(8, 32, value=16, step=1, label="Frames")
                        i2v_fps = gr.Slider(4, 30, value=8, step=1, label="FPS")

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
                                i2v_frames, i2v_fps, i2v_denoise, i2v_format, i2v_mode],
                        outputs=[i2v_video, i2v_preview, i2v_info],
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
| Flux Pro 1.1 | 最高 | 普通 | ~$0.05 (~¥8) | OK |
| Flux Dev | 高い | 普通 | ~$0.025 (~¥4) | OK |
| Flux Schnell | 良い | 最速 | ~$0.003 (~¥0.5) | OK |
| Flux Realism | 最高(写真) | 普通 | ~$0.025 (~¥4) | OK |
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
                    gr.Markdown("**fal.ai動画生成** — Kling, Wan, Minimax等の動画モデル。NSFW OK。")
                    fv_model = gr.Dropdown(
                        choices=list(FAL_VIDEO_MODELS.keys()),
                        value=list(FAL_VIDEO_MODELS.keys())[0],
                        label="動画モデル",
                    )
                    fv_prompt = gr.Textbox(label="Prompt", lines=3, placeholder="例: A woman walking on the beach at sunset, cinematic, slow motion")
                    fv_mode = gr.Radio(choices=["normal", "adult"], value="normal", label="保存先")
                    fv_gen_btn = gr.Button("動画生成 (fal.ai)", variant="primary", size="lg")
                    fv_video = gr.Video(label="Generated Video", height=400)
                    fv_info = gr.Textbox(label="Info", interactive=False)

                    fv_gen_btn.click(
                        fn=generate_fal_video,
                        inputs=[fv_prompt, fv_model, fv_mode],
                        outputs=[fv_video, fv_info],
                    )

                # ── Dezgo 動画生成 ──
                with gr.Tab("AI動画 (Dezgo)"):
                    gr.Markdown("**Dezgo動画生成** — 完全無検閲。Wan 2.2モデル。")
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
                        cg_model = gr.Dropdown(
                            choices=list(CIVITAI_GENERATION_MODELS.keys()),
                            value="Juggernaut XL Ragnarok (SDXL 最高品質)",
                            label="Model (プリセット)",
                            info="SDXL系が高品質。NSFW→Pony系推奨。Version IDでも可",
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
                    choices=["auto", "claude-opus-4-20250514", "claude-sonnet-4-20250514", "claude-haiku-4-5-20251001", "gpt-4o", "gpt-4o-mini", "grok-3", "grok-3-mini"],
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
                    gr.Markdown("## プロンプトテンプレート\nコピペしてQuick/Adultタブで使えます。")
                    for key, tmpl in PROMPT_TEMPLATES.items():
                        with gr.Group():
                            gr.Markdown(f"### {tmpl['name']}")
                            gr.Textbox(value=tmpl["prompt"], label="Prompt", lines=2, interactive=False)
                            gr.Textbox(value=tmpl["negative"], label="Negative Prompt", lines=1, interactive=False)
                            settings_str = f"Steps: {tmpl['settings']['steps']} | CFG: {tmpl['settings']['cfg']} | Size: {tmpl['settings']['width']}x{tmpl['settings']['height']} | Sampler: {tmpl['settings']['sampler']}"
                            gr.Markdown(f"**設定**: {settings_str}")
                            gr.Markdown(f"**推奨モデル**: {', '.join(tmpl['recommended_models'])}")

        # ── Settings Tab ──
        with gr.Tab("Settings"):
            gr.Markdown("**設定** — バックエンド・保存先・Google Drive連携・RunPod・CivitAI")

            with gr.Group():
                gr.Markdown("### Backend (バックエンド切り替え)")
                with gr.Row():
                    s_backend = gr.Radio(
                        choices=["fal", "replicate", "together", "dezgo", "novita", "civitai", "local"],
                        value=config.get("backend", "local"),
                        label="実行先",
                        info="fal=Flux推奨 / replicate=万能 / together=LoRA / dezgo=無検閲 / novita=最安 / civitai=SDXL / local=Mac",
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
                inputs=[s_output_normal, s_output_adult, s_gdrive, s_comfyui_url_display, s_runpod_key, s_backend, s_replicate_key, s_fal_key, s_together_key, s_dezgo_key, s_novita_key, s_civitai_key, s_anthropic_key, s_openai_key, s_xai_key],
                outputs=[save_status],
            )
            gdrive_link_btn.click(fn=link_google_drive, inputs=[s_gdrive], outputs=[gdrive_status])
            open_normal_btn.click(fn=lambda: open_folder(config["output_dir_normal"]), outputs=[save_status])
            open_adult_btn.click(fn=lambda: open_folder(config["output_dir_adult"]), outputs=[save_status])
            open_models_btn.click(fn=lambda: open_folder(config["models_dir"]), outputs=[save_status])


if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
    )
